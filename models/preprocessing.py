import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split


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


def load_dataset(data_dir: Path,
                feature_method: str = 'histogram',
                spatial_downsample: bool = True,
                target_width: int = 64,
                target_height: int = 48,
                n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all event files and convert to features.
    
    Args:
        data_dir: Directory containing .npy event files
        feature_method: Method for feature extraction
        spatial_downsample: Whether to downsample spatially
        target_width: Target width after downsampling
        target_height: Target height after downsampling
        n_bins: Number of temporal bins
        
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
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    print(f"Found {len(event_files)} files with {len(unique_labels)} classes:")
    for label in unique_labels:
        count = labels_str.count(label)
        print(f"  {label}: {count} samples")
    
    features_list = []
    labels_list = []
    
    for filepath, label_str in zip(event_files, labels_str):
        print(f"Processing {filepath.name}...")
        
        # Load events
        events = load_event_file(filepath)
        
        if len(events) == 0:
            print(f"  Warning: {filepath.name} has no events, skipping")
            continue
        
        # Optionally downsample
        if spatial_downsample:
            events = subsample_spatial(events, target_width, target_height)
            width, height = target_width, target_height
        else:
            width, height = 640, 480
        
        # Extract features
        features = events_to_features(
            events, 
            method=feature_method,
            width=width,
            height=height,
            n_bins=n_bins
        )
        
        features_list.append(features)
        labels_list.append(label_to_idx[label_str])
    
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    
    print(f"\nDataset loaded: X.shape={X.shape}, y.shape={y.shape}")
    
    return X, y, unique_labels


def normalize_features(X: np.ndarray) -> np.ndarray:
    """
    Normalize feature vectors using standard (z-score) normalization.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    return (X - mean) / std


def split_dataset(X: np.ndarray,
                 y: np.ndarray,
                 val_split: float = 0.2,
                 test_split: float = 0.1,
                 random_state: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split dataset into train/val/test.
    """
    # Check if stratification is possible
    unique, counts = np.unique(y, return_counts=True)
    min_samples = counts.min()
    can_stratify = min_samples >= 2
    
    if not can_stratify:
        print(f"Warning: Some classes have only {min_samples} sample(s). Splitting without stratification.")
        stratify_arg = None
    else:
        stratify_arg = y
    
    # First split: separate test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_split,
        random_state=random_state,
        stratify=stratify_arg
    )
    
    # Second split: separate val from train
    val_size_adjusted = val_split / (1 - test_split)
    
    # Check stratification for second split
    if can_stratify:
        unique_temp, counts_temp = np.unique(y_temp, return_counts=True)
        can_stratify_temp = counts_temp.min() >= 2
        stratify_temp = y_temp if can_stratify_temp else None
    else:
        stratify_temp = None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify_temp
    )

    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


def one_hot_encode(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot encoding."""
    y_onehot = np.zeros((len(y), n_classes), dtype=np.float32)
    y_onehot[np.arange(len(y)), y] = 1
    return y_onehot


def save_preprocessed(splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
                     save_dir: Path,
                     class_names: List[str]):
    """
    Save preprocessed dataset.
    
    Args:
        splits: Dict with train/val/test splits
        save_dir: Directory to save to
        class_names: List of class names
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, (X, y) in splits.items():
        np.save(save_dir / f'X_{split_name}.npy', X)
        np.save(save_dir / f'y_{split_name}.npy', y)
    
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
