#!/usr/bin/env python3
"""
Preprocess event data for SNN training.
Reads .npy files from data/binary and creates train/val/test splits.
"""
from pathlib import Path
from models.preprocessing import (
    load_dataset,
    normalize_features,
    split_dataset,
    one_hot_encode,
    save_preprocessed
)


def main():
    # Paths
    binary_dir = Path('data/binary')
    save_dir = Path('data/processed')
    
    # Configuration
    config = {
        'feature_method': 'histogram',  # 'histogram', 'rate', or 'time_surface'
        'spatial_downsample': True,
        'target_width': 64,
        'target_height': 48,
        'n_bins': 10,
        'normalize': True,
        'norm_method': 'standard',  # 'standard' or 'minmax'
        'val_split': 0.2,
        'test_split': 0.1,
        'random_state': 42
    }
    
    print("=" * 60)
    print("Event Data Preprocessing Pipeline")
    print("=" * 60)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Load and extract features
    print("Step 1: Loading event data and extracting features...")
    X, y, class_names = load_dataset(
        binary_dir,
        feature_method=config['feature_method'],
        spatial_downsample=config['spatial_downsample'],
        target_width=config['target_width'],
        target_height=config['target_height'],
        n_bins=config['n_bins']
    )
    
    # Normalize
    if config['normalize']:
        print(f"\nStep 2: Normalizing features ({config['norm_method']})...")
        X = normalize_features(X, method=config['norm_method'])
    
    # Split dataset
    print("\nStep 3: Splitting dataset...")
    splits = split_dataset(
        X, y,
        val_split=config['val_split'],
        test_split=config['test_split'],
        random_state=config['random_state']
    )
    
    # Convert labels to one-hot for SNN training
    print("\nStep 4: Converting labels to one-hot encoding...")
    n_classes = len(class_names)
    for split_name in splits.keys():
        X_split, y_split = splits[split_name]
        y_onehot = one_hot_encode(y_split, n_classes)
        splits[split_name] = (X_split, y_onehot)
        print(f"  {split_name}: y shape {y_split.shape} -> {y_onehot.shape}")
    
    # Save
    print("\nStep 5: Saving preprocessed data...")
    save_preprocessed(splits, save_dir, class_names)
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print(f"\nClass mapping:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")
    
    print(f"\nFeature dimensions: {X.shape[1]}")
    print(f"Number of classes: {n_classes}")


if __name__ == '__main__':
    main()
