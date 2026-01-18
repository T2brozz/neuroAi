from datetime import timedelta, datetime
import numpy as np
import cv2 as cv
import dv_processing as dv
import threading
import queue
import time
from collections import deque

# TODO: Add your model imports here
import tensorflow as tf  # for CNN
from models.snn.predict import load_snn_model, predict_snn  # for SNN
from models.event_types import dv_store_to_numpy, EVENT_DTYPE
from models.preprocessing import events_to_features

from pathlib import Path

label_names = ['mark', 'marvin', 'yannes']

# Log file path for non-uncertain classifications
LOG_PATH = Path(f'logs/{datetime.now().strftime("live_predictions_%Y%m%d_%H%M%S.log")}')
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Open the camera, just use first detected DAVIS camera
camera = dv.io.camera.DAVIS()

# Initialize a single-stream slicer for events only
slicer = dv.EventStreamSlicer()

# Initialize a visualizer for events
visualizer = dv.visualization.EventVisualizer(camera.getEventResolution(), 
                                              dv.visualization.colors.black(),
                                              dv.visualization.colors.white(), 
                                              dv.visualization.colors.white())

# Create a window for image display
cv.namedWindow("Event Stream - Person Classification", cv.WINDOW_NORMAL)

# Initialize noise filter with parameters in constructor
noise_filter = dv.noise.BackgroundActivityNoiseFilter(
    camera.getEventResolution(), 
    backgroundActivityDuration=timedelta(milliseconds=2)  # 2ms background activity duration
)

# Load your trained models
cnn_model = tf.keras.models.load_model('checkpoints/cnn/simple_cnn_large.keras')
snn_model = load_snn_model(Path('checkpoints/snn/snn_params'), Path('checkpoints/snn/best_hparams.env'))

# Threading setup for async classification
classification_queue = queue.Queue(maxsize=1)  # Only keep latest events
result_lock = threading.Lock()
current_classification = {"cnn": "CNN: None", "snn": "SNN: None", "timestamp": time.time()}
classification_running = threading.Event()
classification_running.set()
fps_history = deque(maxlen=30)
last_fps_time = time.time()

def _log_classification(model: str, label: str) -> None:
    """Append a classification to the log file for any model.

    Only logs when label is not "Uncertain" or "None".
    Output format: ISO8601 timestamp, model, label (TSV).
    """
    try:
        if not model or not label or label in ("Uncertain", "None"):
            return
        timestamp = datetime.now().isoformat()
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"{timestamp}\t{model}\t{label}\n")
    except Exception:
        # Silently ignore logging errors to avoid interrupting the stream
        pass

def _parse_prediction(prediction: np.ndarray, confidence_threshold: float = 0.5) -> str:
    """Convert model prediction to class name string"""
    if prediction is None or len(prediction) == 0:
        return "None"
    
    idx = np.argmax(prediction)
    if idx < 0 or idx >= len(label_names):
        return "None"
    
    if prediction[idx] < confidence_threshold:
        return "Uncertain"
    
    return label_names[idx]

def _event_resolution_wh():
    """Return (width, height) for event resolution across dv_processing variants."""
    res = camera.getEventResolution()
    # Some versions return an object with width/height, others return a tuple
    if hasattr(res, "width"):
        return int(res.width), int(res.height)
    # Assume tuple-like (w, h)
    return int(res[0]), int(res[1])

def compute_event_centroid(events):
    """Compute the centroid of events for label placement"""
    if len(events) == 0:
        return None
    
    x_coords = np.array([event.x() for event in events])
    y_coords = np.array([event.y() for event in events])
    
    centroid_x = int(np.mean(x_coords))
    centroid_y = int(np.mean(y_coords))
    
    return (centroid_x, centroid_y)

def preprocess_events_for_cnn(events):
    """Convert events to format suitable for CNN inference.
    
    Uses the same preprocessing as training (from preprocess_data.py):
    - feature_method: 'time_surface'
    - spatial_downsample: False (full 640x480 resolution)
    - tau = 1e5 (100ms time constant)
    
    Output shape: (1, 480, 640, 2) matching the CNN model input.
    """
    # Handle empty / None input
    if events is None or len(events) == 0:
        return None
    
    # Convert dv_processing EventStore to structured numpy
    structured = dv_store_to_numpy(events)
    
    if structured.size == 0:
        return None
    
    # Get camera resolution (full resolution - no spatial downsampling)
    width, height = _event_resolution_wh()
    
    # Create time surface (2 channels: positive and negative events)
    # This matches models/preprocessing.py events_to_features with method='time_surface'
    x_coords = np.clip(structured['x'], 0, width - 1).astype(np.int32)
    y_coords = np.clip(structured['y'], 0, height - 1).astype(np.int32)
    
    surface_pos = np.zeros((height, width), dtype=np.float32)
    surface_neg = np.zeros((height, width), dtype=np.float32)
    
    # Time surface: exponential decay based on event recency
    t_ref = structured['t'][-1]  # Most recent timestamp as reference
    tau = 1e5  # Time constant in microseconds (100ms) - matches training
    
    # Compute decay values
    dt = t_ref - structured['t']
    decay = np.exp(-dt / tau).astype(np.float32)
    
    # Separate by polarity
    pos_mask = structured['p'] == 1
    neg_mask = ~pos_mask
    
    # Use maximum decay value at each pixel (most recent event dominates)
    # This matches the training loop: surface[y, x] = max(surface[y, x], val)
    np.maximum.at(surface_pos, (y_coords[pos_mask], x_coords[pos_mask]), decay[pos_mask])
    np.maximum.at(surface_neg, (y_coords[neg_mask], x_coords[neg_mask]), decay[neg_mask])
    
    # Stack into 2-channel image: (H, W, 2) 
    # Training stores as concat([pos.flatten(), neg.flatten()]), then reshapes to (H, W, 2)
    image = np.stack([surface_pos, surface_neg], axis=-1)
    
    # Z-score normalization (same as training with normalize_features)
    mean = image.mean()
    std = image.std()
    if std > 0:
        image = (image - mean) / std
    
    # Add batch dimension: (1, H, W, 2)
    image = np.expand_dims(image, axis=0)
    
    return image

def preprocess_events_for_snn(events):
    """Convert events to format suitable for SNN inference.
    
    Uses the same preprocessing as training:
    - Spatial downsampling: 640x480 -> 64x48
    - Feature method: histogram with 10 time bins
    - Output shape: (30720,) = 64 * 48 * 10
    """
    from models.preprocessing import subsample_spatial
    
    # Handle empty / None input
    if events is None or len(events) == 0:
        return None

    # Convert dv_processing EventStore to structured numpy
    structured = dv_store_to_numpy(events)

    if structured.size == 0:
        return None

    # Get camera resolution
    orig_width, orig_height = _event_resolution_wh()
    
    # Match training preprocessing parameters
    TARGET_WIDTH = 64
    TARGET_HEIGHT = 48
    N_BINS = 10
    FEATURE_METHOD = "histogram"
    
    # Spatial downsampling (same as training)
    downsampled = subsample_spatial(
        structured,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        orig_width=orig_width,
        orig_height=orig_height,
    )
    
    # Extract features using histogram method (same as training)
    features = events_to_features(
        downsampled,
        method=FEATURE_METHOD,
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT,
        n_bins=N_BINS,
    )
    
    features = features.astype(np.float32, copy=False)
    
    # Normalize features (z-score normalization as in training)
    # For live inference, we use a simple normalization
    mean = features.mean()
    std = features.std()
    if std > 0:
        features = (features - mean) / std

    return features

def classification_worker():
    """Background thread that processes classification requests"""
    print("Classification worker thread started")
    
    while classification_running.is_set():
        try:
            # Get events from queue with timeout
            events = classification_queue.get(timeout=0.1)
            
            if events is None:
                continue
            
            # Perform classification
            cnn_result = "CNN: None"
            snn_result = "SNN: None"
            
            if len(events) > 0:
                # CNN inference
                cnn_input = preprocess_events_for_cnn(events)
                if cnn_input is not None:
                    cnn_prediction = cnn_model.predict(cnn_input, verbose=0)[0]
                    cnn_result = f"CNN: {_parse_prediction(cnn_prediction)}"
                    
                    # Log CNN classification
                    if ":" in cnn_result:
                        model, label = cnn_result.split(":", 1)
                        _log_classification(model.strip(), label.strip())
                
                # SNN inference
                snn_input = preprocess_events_for_snn(events)
                if snn_input is not None:
                    snn_prediction = predict_snn(snn_model, features=snn_input)
                    snn_result = f"SNN: {_parse_prediction(snn_prediction)}"
                    
                    # Log SNN classification
                    if ":" in snn_result:
                        model, label = snn_result.split(":", 1)
                        _log_classification(model.strip(), label.strip())
            
            # Update shared results
            with result_lock:
                current_classification["cnn"] = cnn_result
                current_classification["snn"] = snn_result
                current_classification["timestamp"] = time.time()
            
            classification_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Classification error: {e}")
            continue
    
    print("Classification worker thread stopped")

# Callback method for time based slicing
def process_events(events):
    """OPTIMIZED event processing with async classification"""
    global last_fps_time
    
    # FPS tracking
    current_time = time.time()
    frame_time = current_time - last_fps_time
    if frame_time > 0:
        fps_history.append(1.0 / frame_time)
    last_fps_time = current_time
    avg_fps = np.mean(fps_history) if len(fps_history) > 0 else 0
    
    original_count = len(events)
    
    # Apply noise filter (fast operation)
    noise_filter.accept(events)
    filtered_events = noise_filter.generateEvents()
    filtered_count = len(filtered_events) if filtered_events is not None else 0
    
    # Use filtered events
    events_to_process = filtered_events if filtered_events is not None and len(filtered_events) > 0 else events
    
    # Submit to classification thread (non-blocking)
    # Only submit if queue is empty (drop frames if classifier is busy)
    if classification_queue.empty() and len(events_to_process) > 0:
        try:
            classification_queue.put_nowait(events_to_process)
        except queue.Full:
            pass  # Skip this frame if queue is full
    
    # Generate visualization (fast operation)
    event_image = visualizer.generateImage(events_to_process)
    
    # Get current classification results (cached from background thread)
    with result_lock:
        cnn_result = current_classification["cnn"]
        snn_result = current_classification["snn"]
        classification_age = current_time - current_classification["timestamp"]
    
    # Add text overlays (fixed positions for performance)
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # FPS counter (top left)
    cv.putText(event_image, f"FPS: {avg_fps:.1f}", 
              (10, 30), font, font_scale, (0, 255, 0), thickness)
    
    # Event count (top left, below FPS)
    cv.putText(event_image, f"Events: {filtered_count}/{original_count}", 
              (10, 60), font, 0.6, (255, 255, 255), 1)
    
    # Classification age indicator (top left, below event count)
    if classification_age < 1.0:
        age_color = (0, 255, 0)  # Green = fresh
    elif classification_age < 2.0:
        age_color = (0, 255, 255)  # Yellow = slightly old
    else:
        age_color = (0, 0, 255)  # Red = stale
    cv.putText(event_image, f"Class age: {classification_age:.1f}s", 
              (10, 90), font, 0.5, age_color, 1)
    
    # SNN classification (bottom left)
    cv.putText(event_image, snn_result, 
              (10, event_image.shape[0] - 30), 
              font, font_scale, (0, 0, 255), thickness)
    
    # CNN classification (bottom left, above SNN)
    cv.putText(event_image, cnn_result, 
              (10, event_image.shape[0] - 60), 
              font, font_scale, (0, 255, 0), thickness)
    
    # Display the image
    cv.imshow("Event Stream - Person Classification", event_image)
    
    # Exit on ESC key (non-blocking)
    if cv.waitKey(1) & 0xFF == 27:
        classification_running.clear()  # Signal classification thread to stop
        cv.destroyAllWindows()
        exit(0)

# Start classification worker thread
classification_running.set()
classification_thread = threading.Thread(target=classification_worker, daemon=True)
classification_thread.start()
print("Classification worker thread started")

# Register a job to be performed every 33 milliseconds (30 FPS)
slicer.doEveryTimeInterval(timedelta(milliseconds=33), process_events)

# Continue the loop while camera is connected
try:
    while camera.isRunning():
        events = camera.getNextEventBatch()
        if events is not None:
            slicer.accept(events)
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    # Clean shutdown
    classification_running.clear()
    cv.destroyAllWindows()
    print("Waiting for classification thread to finish...")
    classification_thread.join(timeout=2.0)
    print("Done!")