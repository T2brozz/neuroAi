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
    one_hot_encode,
    save_preprocessed,
)


def main():
    # Paths
    binary_dir = Path('data/binary')
    save_dir = Path('data/processed')
    
    # Configuration
    config = {
        'feature_method': 'time_surface',   # 'histogram', 'rate', or 'time_surface'
        'n_bins': 10,                       # Only used for histogram/rate features
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
    print("\nStep 2: Remapping labels...")
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
    
    # Balance stale class to match average of other classes
    # We only compute INDICES to keep, not copying actual data
    print("\nStep 3: Balancing stale class...")
    stale_idx = class_names.index(stale_label)
    non_stale_counts = [np.sum(y_array == idx) for idx, name in enumerate(class_names) if name != stale_label]
    avg_non_stale = int(np.mean(non_stale_counts))
    stale_count = np.sum(y_array == stale_idx)
    
    print(f"  Non-stale class counts: {non_stale_counts}")
    print(f"  Average non-stale count: {avg_non_stale}")
    print(f"  Current stale count: {stale_count}")
    
    if stale_count > avg_non_stale:
        # Find indices of stale samples and randomly select subset
        all_indices = np.arange(len(y_array))
        stale_mask = y_array == stale_idx
        non_stale_mask = ~stale_mask
        
        stale_indices = all_indices[stale_mask]
        non_stale_indices = all_indices[non_stale_mask]
        
        # Randomly select stale samples to keep
        rng_balance = np.random.RandomState(config['random_seed'])
        keep_stale_indices = rng_balance.choice(stale_indices, size=avg_non_stale, replace=False)
        
        # Combine and sort to maintain order - these are INDICES into the original arrays
        balanced_indices = np.sort(np.concatenate([non_stale_indices, keep_stale_indices]))
        
        print(f"  Reducing stale class from {stale_count} to {avg_non_stale} samples")
        print(f"  Total samples: {len(y_array)} -> {len(balanced_indices)}")
    else:
        print(f"  Stale class already balanced (≤ average), no changes needed")
        balanced_indices = np.arange(len(y_array))
    
    # Report final class distribution using the balanced indices
    y_balanced = y_array[balanced_indices]
    print(f"\n  Final balanced class distribution:")
    for idx, name in enumerate(class_names):
        count = np.sum(y_balanced == idx)
        print(f"    {idx}: {name} ({count} samples)")
    
    del y_array, y_balanced
    gc.collect()
    
    # Normalize in chunks (keeps data in memmap) and save statistics
    print("\nStep 4: Normalizing features...")
    X, norm_mean, norm_std = normalize_features(X, return_stats=True)
    
    # Save normalization statistics for use in live prediction
    norm_stats_path = save_dir / 'norm_stats.npz'
    np.savez(norm_stats_path, mean=norm_mean, std=norm_std)
    print(f"  Saved normalization statistics to {norm_stats_path}")
    
    # Shuffle the balanced indices (not the data itself)
    print("\nStep 5: Creating shuffled indices...")
    n_samples = len(balanced_indices)
    rng = np.random.RandomState(config['random_seed'])
    shuffle_perm = rng.permutation(n_samples)
    # Apply shuffle to balanced indices
    final_indices = balanced_indices[shuffle_perm]
    print(f"  Created shuffle permutation with seed: {config['random_seed']}")
    print(f"  Final dataset size: {n_samples} samples")
    
    # Split using indices into the final_indices array
    print("\nStep 6: Splitting dataset...")
    from sklearn.model_selection import train_test_split
    
    # Get labels for stratification
    y_for_split = np.array(y)[final_indices]
    split_indices = np.arange(len(final_indices))
    
    # Check stratification
    unique, counts = np.unique(y_for_split, return_counts=True)
    can_stratify = counts.min() >= 2
    stratify_arg = y_for_split if can_stratify else None
    
    # First split: train+val vs test
    trainval_idx, test_idx = train_test_split(
        split_indices,
        test_size=config['test_split'],
        random_state=config['random_seed'],
        stratify=stratify_arg
    )
    
    # Second split: train vs val
    val_size_adj = config['val_split'] / (1 - config['test_split'])
    if can_stratify:
        y_trainval = y_for_split[trainval_idx]
        stratify_trainval = y_trainval if np.unique(y_trainval, return_counts=True)[1].min() >= 2 else None
    else:
        stratify_trainval = None
    
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_size_adj,
        random_state=config['random_seed'],
        stratify=stratify_trainval
    )
    
    # Map split indices back to original data indices
    splits = {
        'train': (final_indices[train_idx], None),
        'val': (final_indices[val_idx], None),
        'test': (final_indices[test_idx], None),
    }
    data = (X, y)
    
    # Report split sizes
    n_classes = len(class_names)
    for split_name, (indices, _) in splits.items():
        print(f"  {split_name}: {len(indices)} samples")
    
    # Save (handles one-hot encoding internally)
    print("\nStep 7: Saving preprocessed data...")
    save_preprocessed(splits, data, save_dir, class_names)
    
    # Clean up temp files
    print("\nStep 8: Cleaning up temporary files...")
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
