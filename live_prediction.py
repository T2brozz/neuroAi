from datetime import timedelta, datetime
import numpy as np
import cv2 as cv
import dv_processing as dv

# TODO: Add your model imports here
# import tensorflow as tf  # for CNN
from models.snn.predict import load_snn_model, predict_snn  # for SNN
from models.event_types import dv_store_to_numpy, EVENT_DTYPE
from models.preprocessing import events_to_features

from pathlib import Path

label_names = ['mark', 'marvin', 'yannes']

# Log file path for non-uncertain classifications
LOG_PATH = Path(f"logs/{datetime.now().strftime("live_predictions_%Y%m%d_%H%M%S.log")}")
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

# TODO: Load your trained models
# cnn_model = tf.keras.models.load_model('path/to/cnn_model')
snn_model = load_snn_model(Path('checkpoints/snn/snn_params'), Path('checkpoints/snn/best_hparams.env'))

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
    """Convert events to format suitable for CNN inference"""
    # TODO: Implement preprocessing based on your CNN input requirements
    # This might involve creating event frames, histograms, etc.
    pass

def preprocess_events_for_snn(events):
    """Convert events to format suitable for SNN inference.

    This uses the same time-surface feature extraction as the offline
    preprocessing pipeline (models.preprocessing.events_to_features).
    """
    # Handle empty / None input
    if events is None:
        return None

    # dv.EventStore may not support len() directly in all contexts, so be safe
    if len(events) == 0:
        return None

    # Convert dv_processing EventStore or iterable of events to structured numpy
    structured = dv_store_to_numpy(events)

    if structured.size == 0:
        return None

    # Use camera resolution so features match sensor size
    width, height = _event_resolution_wh()

    # Use same defaults as in models.preprocessing.events_to_features
    features = events_to_features(
        structured,
        method="time_surface",
        width=width,
        height=height,
    )

    features = features.astype(np.float32)

    # Adapt feature length to match trained model if needed
    expected_len = getattr(snn_model, "n_features", None)
    if expected_len is not None and features.size != expected_len:
        half = features.size // 2
        pos = features[:half].reshape((height, width))
        neg = features[half:].reshape((height, width))

        # Compute target dimensions from expected_len (assume 2 channels)
        area = expected_len // 2

        # Prefer common sensors if exact match
        target_w, target_h = None, None
        if area == 640 * 480:
            target_w, target_h = 640, 480
        elif area == 346 * 260:
            target_w, target_h = 346, 260
        else:
            # Find factor pair close to current aspect ratio
            aspect = width / height if height > 0 else 1.0
            best_h, best_w, best_err = None, None, float("inf")
            h_start = int(np.sqrt(area / max(aspect, 1e-6)))
            h_start = max(1, h_start)
            for h_try in range(h_start, 0, -1):
                if area % h_try != 0:
                    continue
                w_try = area // h_try
                err = abs((w_try / h_try) - aspect)
                if err < best_err:
                    best_err, best_h, best_w = err, h_try, w_try
                # Stop early if exact aspect match
                if err < 1e-6:
                    break
            target_h = best_h if best_h is not None else height
            target_w = best_w if best_w is not None else width

        # Resize each polarity channel to target dims
        pos_resized = cv.resize(pos, (int(target_w), int(target_h)), interpolation=cv.INTER_LINEAR)
        neg_resized = cv.resize(neg, (int(target_w), int(target_h)), interpolation=cv.INTER_LINEAR)
        features_resized = np.concatenate([pos_resized.flatten(), neg_resized.flatten()]).astype(np.float32)
        features = features_resized

    return features

def classify_events(events):
    """Run both CNN and SNN classification on events"""
    cnn_result = "CNN: None"  # Default
    snn_result = "SNN: None"  # Default
    
    if len(events) > 0:
        # TODO: Implement actual classification
        # cnn_input = preprocess_events_for_cnn(events)
        # cnn_prediction = cnn_model.predict(cnn_input)
        # cnn_result = f"CNN: {get_class_name(cnn_prediction)}"
        
        snn_input = preprocess_events_for_snn(events)
        if snn_input is not None:
            snn_prediction = predict_snn(snn_model, features=snn_input)
            snn_result = f"SNN: {_parse_prediction(snn_prediction)}"
        
        # Placeholder classifications for testing
        # cnn_result = "CNN: Person"
            
    return cnn_result, snn_result

# Callback method for time based slicing
def process_events(events):
    original_count = len(events)
    print(f"Received {original_count} events")
    
    # Apply dv_processing noise filter using the correct pattern
    noise_filter.accept(events)
    filtered_events = noise_filter.generateEvents()
    filtered_count = len(filtered_events) if filtered_events is not None else 0
    
    # Get filter reduction factor
    reduction_factor = noise_filter.getReductionFactor()
    print(f"After noise filtering: {filtered_count} events (reduction factor: {reduction_factor:.3f})")
    
    # Use filtered events for further processing
    events_to_process = filtered_events if filtered_events is not None and len(filtered_events) > 0 else events
    
    # Generate event visualization
    event_image = visualizer.generateImage(events_to_process)
    
    # Get classification results using filtered events
    cnn_result, snn_result = classify_events(events_to_process)

    # Log any confident classifications (CNN & SNN)
    for result in (cnn_result, snn_result):
        if isinstance(result, str) and ":" in result:
            model, label = result.split(":", 1)
            _log_classification(model.strip(), label.strip())
    
    # Compute centroid for label placement using filtered events
    centroid = compute_event_centroid(events_to_process)
    
    if centroid is not None:
        # Draw classification results at centroid
        cv.circle(event_image, centroid, 5, (255, 255, 255), -1)  # White dot at centroid
        
        # Add text labels
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # CNN result (above centroid)
        cv.putText(event_image, cnn_result, 
                  (centroid[0] - 50, centroid[1] - 20), 
                  font, font_scale, (0, 255, 0), thickness)
        
        # SNN result (below centroid)
        cv.putText(event_image, snn_result, 
                  (centroid[0] - 50, centroid[1] + 20), 
                  font, font_scale, (0, 0, 255), thickness)
    
    # Add filtering statistics
    cv.putText(event_image, f"Raw Events: {original_count}", 
              (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.putText(event_image, f"Filtered: {filtered_count}", 
              (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show filter reduction factor
    cv.putText(event_image, f"Reduction Factor: {reduction_factor:.3f}", 
              (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display the image
    cv.imshow("Event Stream - Person Classification", event_image)
    
    # Exit on ESC key
    if cv.waitKey(1) == 27:
        cv.destroyAllWindows()
        exit(0)

# Register a job to be performed every 33 milliseconds (30 FPS)
slicer.doEveryTimeInterval(timedelta(milliseconds=100), process_events)

# Continue the loop while camera is connected
while camera.isRunning():
    events = camera.getNextEventBatch()
    if events is not None:
        slicer.accept(events)