#!/usr/bin/env python3
"""
Preprocess event data for SNN training.
Reads .npy files from data/binary and creates train/val/test splits.

Uses memory-mapped arrays to handle large datasets without running out of RAM.
"""
import gc
import numpy as np
from pathlib import Path
from models.preprocessing import (
    load_dataset,
    normalize_features,
    split_dataset,
    one_hot_encode,
    save_preprocessed,
    split_events_into_windows,
)


def main():
    # Paths
    binary_dir = Path('data/binary')
    save_dir = Path('data/processed')
    
    # Configuration
    config = {
        'feature_method': 'time_surface',  # 'histogram', 'rate', or 'time_surface'
        'n_bins': 10,                       # Only used for histogram/rate features
        'filter_activity': False,           # Disabled - low activity goes to 'stale' label
        'activity_window_ms': 1000,         # Window size for activity check
        'activity_min_events': 75000,       # Threshold: below this → 'stale' label
        'event_window_ms': 1000,            # Split recordings into 1s windows
        'val_split': 0.2,
        'test_split': 0.1,
        'random_seed': 42,
        # Labels to keep - others are mapped to 'other_label'
        'keep_labels': ['mark', 'marvin', 'yannes'],
        # Label for low-activity windows (below activity_min_events)
        'stale_label': 'stale',
        'other_label': 'unknown',
    }
    
    print("=" * 60)
    print("Event Data Preprocessing Pipeline")
    print("=" * 60)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Load and extract features with stale labeling for low-activity windows
    print("Step 1: Loading event data and extracting features...")
    X, y, class_names = load_dataset(
        binary_dir,
        feature_method=config['feature_method'],
        n_bins=config['n_bins'],
        filter_activity=False,  # We handle activity via stale_label instead
        activity_window_ms=config['activity_window_ms'],
        activity_min_events=config['activity_min_events'],
        event_window_ms=config['event_window_ms'],
        stale_label=config['stale_label'],  # Label for low-activity windows
        temp_dir=save_dir / 'temp',  # Use data/processed/temp for memmap files
    )
    
    # Remap labels: keep_labels stay, others become 'other_label'
    print("\nStep 1.5: Remapping labels...")
    keep_labels = config['keep_labels']
    stale_label = config['stale_label']
    other_label = config['other_label']
    
    # Build final class list: keep_labels + other_label + 'stale' (if present)
    final_classes = list(keep_labels)
    if other_label not in final_classes:
        final_classes.append(other_label)
    if stale_label and stale_label not in final_classes:
        final_classes.append(stale_label)
    
    # Create mapping from old class indices to new
    old_to_new = {}
    other_idx = final_classes.index(other_label)
    for old_idx, old_name in enumerate(class_names):
        if old_name in keep_labels:
            old_to_new[old_idx] = final_classes.index(old_name)
        elif old_name == stale_label:
            old_to_new[old_idx] = final_classes.index(stale_label)
        else:
            old_to_new[old_idx] = other_idx
    
    # Apply remapping in-place (works with memmap)
    print("  Remapping labels in-place...")
    for i in range(len(y)):
        y[i] = old_to_new[y[i]]
    if hasattr(y, 'flush'):
        y.flush()
    
    class_names = final_classes
    
    # Count samples per class (load y into memory temporarily for counting)
    y_array = np.array(y)
    print(f"  Final classes: {class_names}")
    for idx, name in enumerate(class_names):
        count = np.sum(y_array == idx)
        print(f"    {idx}: {name} ({count} samples)")
    del y_array
    gc.collect()
    
    # Normalize (in-place for memmap)
    print("\nStep 2: Normalizing features...")
    X = normalize_features(X)
    
    # Shuffle indices (not the data itself)
    print("\nStep 3: Creating shuffled indices...")
    n_samples = len(y)
    rng = np.random.RandomState(config['random_seed'])
    shuffle_idx = rng.permutation(n_samples)
    print(f"  Created shuffle permutation with seed: {config['random_seed']}")
    
    # Split dataset (returns indices, not data copies)
    print("\nStep 4: Splitting dataset...")
    splits, data = split_dataset(
        X, y,
        val_split=config['val_split'],
        test_split=config['test_split'],
        random_state=config['random_seed'],
    )
    
    # Apply shuffle to split indices
    for split_name in splits:
        indices, _ = splits[split_name]
        # Map through shuffle permutation
        splits[split_name] = (shuffle_idx[indices], None)
    
    # Report split sizes
    n_classes = len(class_names)
    for split_name, (indices, _) in splits.items():
        print(f"  {split_name}: {len(indices)} samples")
    
    # Save (handles one-hot encoding internally)
    print("\nStep 5: Saving preprocessed data...")
    save_preprocessed(splits, data, save_dir, class_names)
    
    # Clean up temp files
    print("\nStep 6: Cleaning up temporary files...")
    temp_dir = save_dir / 'temp'
    if temp_dir.exists():
        for f in temp_dir.glob('*.mmap'):
            try:
                f.unlink()
            except Exception:
                pass
        try:
            temp_dir.rmdir()
        except Exception:
            pass
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print(f"\nClass mapping:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")
    
    print(f"\nFeature dimensions: {X.shape[1]}")
    print(f"Number of classes: {n_classes}")
    print(f"Output directory: {save_dir}")


if __name__ == '__main__':
    main()
