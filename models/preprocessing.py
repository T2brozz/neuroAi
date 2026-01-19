import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
import tempfile
import gc


def extract_label_from_filename(filepath: Path) -> str:
    """
    Extract label (person name) from filename.
    Example: 'benjamin_dvSave-2025_12_04_11_12_56.bin.npy' -> 'benjamin'
    """
    filename = filepath.stem.replace('.bin', '')
    label = filename.split('_')[0]
    return label


def load_event_file(filepath: Path) -> np.ndarray:
    """Load event data from .npy file."""
    return np.load(filepath, allow_pickle=True)


def filter_events_by_activity(events: np.ndarray,
                              window_ms: int = 100,
                              min_events_per_window: int = 10) -> np.ndarray:
    """
    Filter out inactive periods by removing events from low-activity time windows.
    Useful for removing frames where no person is visible/moving.
    
    Args:
        events: Structured array with fields (t, x, y, p)
        window_ms: Time window size in milliseconds
        min_events_per_window: Minimum events required in a window to keep them
        
    Returns:
        Filtered events array
    """
    if len(events) == 0:
        return events
    
    # Convert window from ms to microseconds (assuming event timestamps are in us)
    window_us = window_ms * 1000
    
    t_min = events['t'].min()
    t_max = events['t'].max()
    
    # Create windows
    num_windows = int(np.ceil((t_max - t_min) / window_us))
    
    # Mark events in active windows
    active_mask = np.zeros(len(events), dtype=bool)
    
    for window_idx in range(num_windows):
        window_start = t_min + window_idx * window_us
        window_end = window_start + window_us
        
        window_mask = (events['t'] >= window_start) & (events['t'] < window_end)
        event_count = np.sum(window_mask)
        
        # Mark events in this window as active if threshold met
        if event_count >= min_events_per_window:
            active_mask[window_mask] = True
    
    filtered_events = events[active_mask]
    
    removed_pct = 100 * (1 - len(filtered_events) / len(events))
    print(f"  Activity filter: Removed {removed_pct:.1f}% of events (window={window_ms}ms, min_events={min_events_per_window})")
    
    return filtered_events


def events_to_features(events: np.ndarray, 
                       method: str = 'histogram',
                       width: int = 640,
                       height: int = 480,
                       n_bins: int = 10) -> np.ndarray:
    """
    Convert raw events to feature representation for SNN.
    
    Args:
        events: Structured array with fields (t, x, y, p)
        method: Feature extraction method ('histogram', 'rate', 'time_surface')
        width: Sensor width
        height: Sensor height
        n_bins: Number of temporal bins for histogram methods
        
    Returns:
        Feature vector
    """
    if len(events) == 0:
        # Return zero features if no events
        if method == 'histogram':
            return np.zeros(width * height * n_bins, dtype=np.float32)
        elif method == 'rate':
            return np.zeros(width * height * 2, dtype=np.float32)
        elif method == 'time_surface':
            return np.zeros(width * height * 2, dtype=np.float32)
    
    if method == 'histogram':
        # Create spatio-temporal histogram
        t_min, t_max = events['t'].min(), events['t'].max()
        
        if t_max == t_min:
            t_bins = np.zeros(len(events), dtype=np.int32)
        else:
            t_norm = (events['t'] - t_min) / (t_max - t_min)
            t_bins = np.clip((t_norm * n_bins).astype(np.int32), 0, n_bins - 1)
        
        x_coords = np.clip(events['x'], 0, width - 1)
        y_coords = np.clip(events['y'], 0, height - 1)
        
        # Create 3D histogram (time, y, x)
        hist = np.zeros((n_bins, height, width), dtype=np.float32)
        polarities = events['p'].astype(np.float32) * 2 - 1
        
        for i in range(len(events)):
            hist[t_bins[i], y_coords[i], x_coords[i]] += polarities[i]
        
        # Flatten to feature vector
        return hist.flatten()
    
    elif method == 'rate':
        # Event rate per pixel, separate channels for each polarity
        x_coords = np.clip(events['x'], 0, width - 1)
        y_coords = np.clip(events['y'], 0, height - 1)
        
        rate_pos = np.zeros((height, width), dtype=np.float32)
        rate_neg = np.zeros((height, width), dtype=np.float32)
        
        pos_mask = events['p'] == 1
        neg_mask = events['p'] == 0
        
        np.add.at(rate_pos, (y_coords[pos_mask], x_coords[pos_mask]), 1)
        np.add.at(rate_neg, (y_coords[neg_mask], x_coords[neg_mask]), 1)
        
        # Normalize by total time
        t_duration = events['t'].max() - events['t'].min()
        if t_duration > 0:
            rate_pos /= (t_duration / 1e6)  # Convert to events per second
            rate_neg /= (t_duration / 1e6)
        
        return np.concatenate([rate_pos.flatten(), rate_neg.flatten()])
    
    elif method == 'time_surface':
        # Time surface: most recent event timestamp per pixel
        x_coords = np.clip(events['x'], 0, width - 1)
        y_coords = np.clip(events['y'], 0, height - 1)
        
        surface_pos = np.zeros((height, width), dtype=np.float32)
        surface_neg = np.zeros((height, width), dtype=np.float32)
        
        t_ref = events['t'][-1]
        tau = 1e5  # Time constant in microseconds
        
        for i in range(len(events)):
            y, x = y_coords[i], x_coords[i]
            dt = t_ref - events['t'][i]
            val = np.exp(-dt / tau)
            
            if events['p'][i]:
                surface_pos[y, x] = max(surface_pos[y, x], val)
            else:
                surface_neg[y, x] = max(surface_neg[y, x], val)
        
        return np.concatenate([surface_pos.flatten(), surface_neg.flatten()])
    
    else:
        raise ValueError(f"Unknown method: {method}")


def subsample_spatial(events: np.ndarray, 
                      target_width: int = 64,
                      target_height: int = 48,
                      orig_width: int = 640,
                      orig_height: int = 480) -> np.ndarray:
    """
    Spatially subsample events to reduce dimensionality.
    """
    if len(events) == 0:
        return events
    
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height
    
    subsampled = events.copy()
    subsampled['x'] = (events['x'] * scale_x).astype(np.uint16)
    subsampled['y'] = (events['y'] * scale_y).astype(np.uint16)
    
    return subsampled


def downsample_features(X: np.ndarray,
                        orig_width: int = 640,
                        orig_height: int = 480,
                        target_width: int = 64,
                        target_height: int = 48,
                        n_channels: int = 2) -> np.ndarray:
    """
    Spatially downsample feature arrays from full resolution to target resolution.
    
    Works with time_surface (2 channels) or rate (2 channels) features.
    Uses area averaging for downsampling.
    
    Args:
        X: Feature array of shape (n_samples, orig_height * orig_width * n_channels)
           or (n_samples, n_features) where n_features = orig_height * orig_width * n_channels
        orig_width: Original width (default 640)
        orig_height: Original height (default 480)
        target_width: Target width after downsampling (default 64)
        target_height: Target height after downsampling (default 48)
        n_channels: Number of channels (2 for time_surface/rate)
        
    Returns:
        Downsampled features of shape (n_samples, target_height * target_width * n_channels)
    """
    n_samples = X.shape[0]
    expected_features = orig_width * orig_height * n_channels
    
    if X.shape[1] != expected_features:
        raise ValueError(
            f"Feature dimension {X.shape[1]} doesn't match expected "
            f"{orig_width}x{orig_height}x{n_channels} = {expected_features}"
        )
    
    # Reshape to (n_samples, n_channels, orig_height, orig_width)
    X_reshaped = X.reshape(n_samples, n_channels, orig_height, orig_width)
    
    # Calculate pooling factors
    pool_h = orig_height // target_height
    pool_w = orig_width // target_width
    
    # Use reshape + mean for area averaging (fast)
    # First crop to exact multiple if needed
    crop_h = target_height * pool_h
    crop_w = target_width * pool_w
    X_cropped = X_reshaped[:, :, :crop_h, :crop_w]
    
    # Reshape to (n_samples, n_channels, target_height, pool_h, target_width, pool_w)
    X_pooled = X_cropped.reshape(
        n_samples, n_channels, target_height, pool_h, target_width, pool_w
    )
    
    # Average over pooling dimensions
    X_downsampled = X_pooled.mean(axis=(3, 5))
    
    # Flatten back to (n_samples, target_height * target_width * n_channels)
    output_features = target_height * target_width * n_channels
    X_out = X_downsampled.reshape(n_samples, output_features)
    
    return X_out.astype(np.float32)


def downsample_features_chunked(X: np.ndarray,
                                 orig_width: int = 640,
                                 orig_height: int = 480,
                                 target_width: int = 64,
                                 target_height: int = 48,
                                 n_channels: int = 2,
                                 chunk_size: int = 100) -> np.ndarray:
    """
    Memory-efficient chunked version of downsample_features.
    
    Processes data in chunks to avoid memory issues with large datasets.
    Works with both regular arrays and memory-mapped arrays.
    
    Args:
        X: Feature array (can be memmap)
        orig_width, orig_height: Original dimensions
        target_width, target_height: Target dimensions
        n_channels: Number of channels
        chunk_size: Samples to process at a time
        
    Returns:
        Downsampled features array
    """
    n_samples = X.shape[0]
    output_features = target_height * target_width * n_channels
    
    # Allocate output array
    X_out = np.zeros((n_samples, output_features), dtype=np.float32)
    
    print(f"  Downsampling {n_samples} samples from {orig_width}x{orig_height} to {target_width}x{target_height}...")
    
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = np.array(X[start:end])  # Load chunk into memory
        
        X_out[start:end] = downsample_features(
            chunk,
            orig_width=orig_width,
            orig_height=orig_height,
            target_width=target_width,
            target_height=target_height,
            n_channels=n_channels
        )
        
        if (end - start) == chunk_size and end < n_samples:
            progress = 100 * end / n_samples
            print(f"    Progress: {progress:.0f}%", end='\r')
    
    print(f"  Downsampling complete: {X.shape[1]} -> {output_features} features")
    return X_out


def split_events_into_windows(events: np.ndarray,
                              window_ms: int) -> List[np.ndarray]:
    """
    Split events into fixed-duration time windows.
    
    Args:
        events: Structured array with fields (t, x, y, p)
        window_ms: Window size in milliseconds
        
    Returns:
        List of event arrays, one per window
    """
    if len(events) == 0:
        return []
    
    window_us = window_ms * 1000
    t_min = events['t'].min()
    t_max = events['t'].max()
    
    windows = []
    window_start = t_min
    
    while window_start < t_max:
        window_end = window_start + window_us
        mask = (events['t'] >= window_start) & (events['t'] < window_end)
        window_events = events[mask]
        
        if len(window_events) > 0:
            windows.append(window_events)
        
        window_start = window_end
    
    return windows


def _get_feature_dim(feature_method: str, width: int, height: int, n_bins: int) -> int:
    """Calculate feature vector dimension based on method and resolution."""
    if feature_method == 'histogram':
        return width * height * n_bins
    elif feature_method in ('rate', 'time_surface'):
        return width * height * 2
    else:
        raise ValueError(f"Unknown feature method: {feature_method}")


def _process_file_to_memmap(
    filepath: Path,
    label_str: str,
    label_to_idx: Dict[str, int],
    feature_method: str,
    n_bins: int,
    filter_activity: bool,
    activity_window_ms: int,
    activity_min_events: int,
    event_window_ms: int | None,
    stale_label: str | None,
    X_mmap: np.memmap,
    y_mmap: np.memmap,
    start_idx: int,
) -> Tuple[int, int, str]:
    """
    Process a single file and write directly to memory-mapped arrays.
    Returns (num_windows_written, stale_count, filename).
    """
    # Load events
    events = load_event_file(filepath)
    
    if len(events) == 0:
        return 0, 0, filepath.name
    
    # Filter low-activity periods (only if not using stale_label)
    if filter_activity and not stale_label:
        events = filter_events_by_activity(
            events,
            window_ms=activity_window_ms,
            min_events_per_window=activity_min_events
        )
    
    # Split into time windows if specified
    if event_window_ms is not None:
        event_windows = split_events_into_windows(events, event_window_ms)
    else:
        event_windows = [events]
    
    # Free the original events array
    del events
    gc.collect()
    
    # Process each window and write directly to memmap
    write_idx = start_idx
    stale_count = 0
    width, height = 640, 480
    
    for window_events in event_windows:
        if len(window_events) == 0:
            continue
        
        # Determine label
        if stale_label and len(window_events) < activity_min_events:
            window_label = stale_label
            stale_count += 1
        else:
            window_label = label_str
        
        # Extract features and write directly to memmap
        features = events_to_features(
            window_events,
            method=feature_method,
            width=width,
            height=height,
            n_bins=n_bins
        )
        
        X_mmap[write_idx] = features
        y_mmap[write_idx] = label_to_idx[window_label]
        write_idx += 1
        
        # Free memory
        del features
    
    # Flush to disk
    X_mmap.flush()
    y_mmap.flush()
    
    return write_idx - start_idx, stale_count, filepath.name


def _count_windows_in_file(
    filepath: Path,
    event_window_ms: int | None,
) -> int:
    """Count how many windows a file will produce (for pre-allocation)."""
    events = load_event_file(filepath)
    if len(events) == 0:
        return 0
    
    if event_window_ms is None:
        return 1
    
    window_us = event_window_ms * 1000
    t_min = events['t'].min()
    t_max = events['t'].max()
    duration = t_max - t_min
    
    return max(1, int(np.ceil(duration / window_us)))


def load_dataset(data_dir: Path,
                feature_method: str = 'histogram',
                n_bins: int = 10,
                filter_activity: bool = False,
                activity_window_ms: int = 100,
                activity_min_events: int = 10,
                event_window_ms: int | None = None,
                stale_label: str | None = None,
                temp_dir: Path | None = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all event files and convert to features using memory-mapped arrays.
    
    Processes files sequentially and writes to disk incrementally to avoid
    running out of memory with large datasets.
    
    Args:
        data_dir: Directory containing .npy event files
        feature_method: Method for feature extraction
        n_bins: Number of temporal bins
        filter_activity: Whether to filter out low-activity periods
        activity_window_ms: Time window for activity filtering (milliseconds)
        activity_min_events: Minimum events required per window to keep
        event_window_ms: Split events into time windows (milliseconds), None to disable
        stale_label: If set, windows below activity_min_events get this label instead of being filtered
        temp_dir: Directory for temporary memmap files (default: system temp)
        
    Returns:
        Tuple of (features, labels, class_names)
    """
    event_files = sorted(list(data_dir.glob('*.npy')))
    event_files = [f for f in event_files if f.stem != '.gitkeep']
    
    if len(event_files) == 0:
        raise ValueError(f"No .npy files found in {data_dir}")
    
    # Extract labels and create mapping
    labels_str = [extract_label_from_filename(f) for f in event_files]
    unique_labels = sorted(set(labels_str))
    
    # Add stale_label to unique labels if specified
    if stale_label and stale_label not in unique_labels:
        unique_labels.append(stale_label)
    
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    print(f"Found {len(event_files)} files with {len(unique_labels)} classes:")
    for label in unique_labels:
        count = labels_str.count(label)
        print(f"  {label}: {count} files")
    if stale_label:
        print(f"  ('{stale_label}' label will be assigned to low-activity windows)")
    
    # Count total windows for pre-allocation
    print(f"\nCounting windows in {len(event_files)} files...")
    total_windows = 0
    for filepath in event_files:
        total_windows += _count_windows_in_file(filepath, event_window_ms)
    
    # Calculate feature dimensions
    width, height = 640, 480
    feature_dim = _get_feature_dim(feature_method, width, height, n_bins)
    
    print(f"  Total windows: {total_windows}")
    print(f"  Feature dim: {feature_dim}")
    estimated_gb = (total_windows * feature_dim * 4) / (1024**3)
    print(f"  Estimated size: {estimated_gb:.2f} GB")
    
    # Create memory-mapped arrays for incremental writing
    if temp_dir is None:
        temp_dir = Path(tempfile.gettempdir())
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    X_path = temp_dir / 'X_temp.mmap'
    y_path = temp_dir / 'y_temp.mmap'
    
    print(f"\nCreating memory-mapped arrays in {temp_dir}...")
    X_mmap = np.memmap(X_path, dtype=np.float32, mode='w+', shape=(total_windows, feature_dim))
    y_mmap = np.memmap(y_path, dtype=np.int32, mode='w+', shape=(total_windows,))
    
    # Process files sequentially
    print(f"\nProcessing {len(event_files)} files sequentially...")
    current_idx = 0
    total_stale = 0
    
    for filepath, label_str in zip(event_files, labels_str):
        n_written, stale_count, filename = _process_file_to_memmap(
            filepath=filepath,
            label_str=label_str,
            label_to_idx=label_to_idx,
            feature_method=feature_method,
            n_bins=n_bins,
            filter_activity=filter_activity,
            activity_window_ms=activity_window_ms,
            activity_min_events=activity_min_events,
            event_window_ms=event_window_ms,
            stale_label=stale_label,
            X_mmap=X_mmap,
            y_mmap=y_mmap,
            start_idx=current_idx,
        )
        
        if n_written == 0:
            print(f"  Warning: {filename} has no events, skipping")
        else:
            if event_window_ms is not None:
                print(f"  {filename}: {n_written} windows", end="")
                if stale_count > 0:
                    print(f" ({stale_count} stale)")
                else:
                    print()
        
        current_idx += n_written
        total_stale += stale_count
        
        # Force garbage collection after each file
        gc.collect()
    
    if stale_label and total_stale > 0:
        print(f"\nTotal windows marked as '{stale_label}': {total_stale}")
    
    # Trim arrays to actual size (in case some windows were empty)
    actual_size = current_idx
    print(f"\nFinalizing arrays (actual size: {actual_size} samples)...")
    
    # Return views of the memmap arrays (don't load into RAM)
    # The caller should handle these as memmap arrays
    X = X_mmap[:actual_size]
    y = y_mmap[:actual_size]
    
    print(f"\nDataset loaded: X.shape={X.shape}, y.shape={y.shape}")
    print(f"  Note: Data stored in memory-mapped files at {temp_dir}")
    print(f"  Files will be cleaned up when arrays are deleted or process exits.")
    
    return X, y, unique_labels


def normalize_features(X: np.ndarray, chunk_size: int = 500) -> np.ndarray:
    """
    Normalize feature vectors using standard (z-score) normalization.
    Works with memory-mapped arrays by processing in chunks.
    
    Args:
        X: Feature array (can be memmap)
        chunk_size: Number of samples to process at a time for stats
        
    Returns:
        Normalized array (in-place if X is memmap)
    """
    n_samples, n_features = X.shape
    print(f"  Computing statistics over {n_samples} samples...")
    
    # Compute mean and std incrementally using Welford's algorithm
    # to avoid loading all data into memory
    mean = np.zeros(n_features, dtype=np.float64)
    M2 = np.zeros(n_features, dtype=np.float64)
    count = 0
    
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = X[start:end].astype(np.float64)
        
        for row in chunk:
            count += 1
            delta = row - mean
            mean += delta / count
            delta2 = row - mean
            M2 += delta * delta2
    
    std = np.sqrt(M2 / count).astype(np.float32)
    mean = mean.astype(np.float32)
    std[std == 0] = 1
    
    # Normalize in-place in chunks
    print(f"  Normalizing in chunks...")
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        X[start:end] = (X[start:end] - mean) / std
        
        if hasattr(X, 'flush'):
            X.flush()
    
    return X


def split_dataset(X: np.ndarray,
                 y: np.ndarray,
                 val_split: float = 0.2,
                 test_split: float = 0.1,
                 random_state: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split dataset into train/val/test using index-based splitting.
    Works with memory-mapped arrays by only shuffling indices.
    """
    n_samples = len(y)
    indices = np.arange(n_samples)
    
    # Load y into memory for stratification (small compared to X)
    y_array = np.array(y) if hasattr(y, 'flush') else y
    
    # Check if stratification is possible
    unique, counts = np.unique(y_array, return_counts=True)
    min_samples = counts.min()
    can_stratify = min_samples >= 2
    
    if not can_stratify:
        print(f"Warning: Some classes have only {min_samples} sample(s). Splitting without stratification.")
        stratify_arg = None
    else:
        stratify_arg = y_array
    
    # First split: separate test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_split,
        random_state=random_state,
        stratify=stratify_arg
    )
    
    # Second split: separate val from train
    val_size_adjusted = val_split / (1 - test_split)
    
    # Check stratification for second split
    if can_stratify:
        y_train_val = y_array[train_val_idx]
        unique_temp, counts_temp = np.unique(y_train_val, return_counts=True)
        can_stratify_temp = counts_temp.min() >= 2
        stratify_temp = y_train_val if can_stratify_temp else None
    else:
        stratify_temp = None
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify_temp
    )
    
    # Return index arrays instead of copying data
    return {
        'train': (train_idx, None),
        'val': (val_idx, None),
        'test': (test_idx, None)
    }, (X, y)


def one_hot_encode(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot encoding."""
    y_onehot = np.zeros((len(y), n_classes), dtype=np.float32)
    y_onehot[np.arange(len(y)), y] = 1
    return y_onehot


def save_preprocessed(splits: Dict[str, Tuple[np.ndarray, None]],
                     data: Tuple[np.ndarray, np.ndarray],
                     save_dir: Path,
                     class_names: List[str],
                     chunk_size: int = 100):
    """
    Save preprocessed dataset, writing in chunks to avoid memory issues.
    
    Args:
        splits: Dict with train/val/test index arrays from split_dataset
        data: Tuple of (X, y) arrays (can be memmap)
        save_dir: Directory to save to
        class_names: List of class names
        chunk_size: Number of samples to write at a time
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    X, y = data
    
    for split_name, (indices, _) in splits.items():
        n_samples = len(indices)
        n_features = X.shape[1]
        n_classes = len(class_names)
        
        print(f"  Saving {split_name}: {n_samples} samples...")
        
        # Create output files
        X_path = save_dir / f'X_{split_name}.npy'
        y_path = save_dir / f'y_{split_name}.npy'
        
        # Get y values for this split (small, can fit in memory)
        y_split = np.array(y[indices], dtype=np.int32)
        y_onehot = one_hot_encode(y_split, n_classes)
        np.save(y_path, y_onehot)
        
        # Save X in chunks using memmap
        # First create the output file with correct shape
        X_out = np.lib.format.open_memmap(
            X_path, mode='w+', dtype=np.float32, shape=(n_samples, n_features)
        )
        
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            batch_indices = indices[i:end]
            X_out[i:end] = X[batch_indices]
            X_out.flush()
        
        del X_out
        gc.collect()
    
    # Save class names
    np.save(save_dir / 'class_names.npy', np.array(class_names))
    
    print(f"\nSaved preprocessed data to {save_dir}")


def load_preprocessed(data_dir: Path) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], List[str]]:
    """
    Load preprocessed dataset.
    
    Returns:
        Tuple of (splits_dict, class_names)
    """
    splits = {}
    
    for split_name in ['train', 'val', 'test']:
        X = np.load(data_dir / Path(f'X_{split_name}.npy'))
        y = np.load(data_dir / Path(f'y_{split_name}.npy'))
        splits[split_name] = (X, y)
        print(f"Loaded {split_name}: X={X.shape}, y={y.shape}")
    
    class_names = np.load(data_dir / Path('class_names.npy'), allow_pickle=True).tolist()

    print(f"Loaded class names: {class_names}")
    
    return splits, class_names
