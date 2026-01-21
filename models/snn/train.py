import logging
import os

import numpy as np
import tensorflow as tf
import nengo_dl
from pathlib import Path


def train_model(
    net,
    inp,
    p_out,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    checkpoint_dir: Path = None,
    use_early_stopping: bool = True,
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 0.0,
    label_smoothing: float = 0.1,
    gradient_clip_norm: float = 1.0,
    reduce_lr_patience: int = 7,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    quiet: bool = False,
) -> tuple[tf.keras.callbacks.History, nengo_dl.Simulator]:
    """
    Train a spiking neural network with regularization to prevent overfitting.
    
    Args:
        net: Built Nengo network
        inp: Input node from the network
        p_out: Output probe from the network
        X_train: Training features (n_samples, n_features)
        y_train: Training labels one-hot encoded (n_samples, n_classes)
        X_val: Validation features (n_samples, n_features)
        y_val: Validation labels one-hot encoded (n_samples, n_classes)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization strength (0.0 = no regularization)
        checkpoint_dir: Directory to save checkpoints
        use_early_stopping: Whether to enable early stopping based on validation loss
        early_stopping_patience: Number of epochs with no improvement before stopping
        early_stopping_min_delta: Minimum change in the monitored quantity to qualify as improvement
        label_smoothing: Label smoothing factor (0.0 = no smoothing, 0.1-0.2 recommended)
        gradient_clip_norm: Max gradient norm for clipping (None or 0 to disable)
        reduce_lr_patience: Patience for ReduceLROnPlateau (0 to disable)
        reduce_lr_factor: Factor to reduce LR by
        min_lr: Minimum learning rate
        quiet: Suppress verbose logging
        
    Returns:
        Tuple of (training history, simulator)
    """

    if quiet:
        logging.getLogger("nengo").setLevel(logging.WARNING)
        logging.getLogger("nengo_dl").setLevel(logging.WARNING)

    # Configure nengo_dl settings for efficient training
    with net:
        nengo_dl.configure_settings(stateful=False, use_loop=False)
    
    print("Creating simulator...")
    
    sim = nengo_dl.Simulator(net, minibatch_size=None)

    print("Compiling model...")
    
    # Build optimizer with weight decay and gradient clipping
    optimizer = _build_optimizer(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip_norm=gradient_clip_norm,
    )
    
    # Use label smoothing in loss function to prevent overconfident predictions
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=label_smoothing,
    )
    if label_smoothing > 0:
        print(f"  Using label smoothing: {label_smoothing}")
    
    sim.compile(
        optimizer=optimizer,
        loss={p_out: loss_fn},
        metrics={p_out: ["accuracy"]}
    )

    # Reshape data for nengo_dl: (batch, time, features)
    X_train_reshaped = X_train[:, np.newaxis, :]
    y_train_reshaped = y_train[:, np.newaxis, :] if len(y_train.shape) == 2 else y_train
    
    X_val_reshaped = X_val[:, np.newaxis, :] if X_val is not None else None
    y_val_reshaped = y_val[:, np.newaxis, :] if y_val is not None and len(y_val.shape) == 2 else y_val
    
    val_data = None
    if X_val is not None and y_val is not None:
        val_data = ({inp: X_val_reshaped}, {p_out: y_val_reshaped})

    # Configure callbacks for regularization
    callbacks = _build_callbacks(
        val_data=val_data,
        use_early_stopping=use_early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        reduce_lr_patience=reduce_lr_patience,
        reduce_lr_factor=reduce_lr_factor,
        min_lr=min_lr,
    )
    
    print(f"\nTraining for up to {epochs} epochs...")
    print(f"  Input shape: {X_train_reshaped.shape}")
    print(f"  Label shape: {y_train_reshaped.shape}")
    
    history = sim.fit(
        {inp: X_train_reshaped},
        {p_out: y_train_reshaped},
        validation_data=val_data,
        callbacks=callbacks if callbacks else None,
        epochs=epochs,
    )

    print("\nTraining complete!")
    
    # Save checkpoint if directory provided
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        params_file = checkpoint_dir / "snn_params"
        print(f"Saving model to {params_file}...")
        sim.save_params(str(params_file))

    return history, sim


def _build_optimizer(
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float | None,
) -> tf.keras.optimizers.Optimizer:
    """Build optimizer with weight decay and gradient clipping."""
    
    clip_kwargs = {}
    if gradient_clip_norm is not None and gradient_clip_norm > 0:
        clip_kwargs["clipnorm"] = gradient_clip_norm
        print(f"  Using gradient clipping: norm={gradient_clip_norm}")
    
    if weight_decay > 0.0:
        # Try different AdamW locations based on TensorFlow version
        try:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                **clip_kwargs,
            )
            print(f"  Using AdamW optimizer with weight_decay={weight_decay}")
            return optimizer
        except AttributeError:
            pass
        
        try:
            optimizer = tf.keras.optimizers.experimental.AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                **clip_kwargs,
            )
            print(f"  Using AdamW (experimental) with weight_decay={weight_decay}")
            return optimizer
        except AttributeError:
            print(f"  WARNING: AdamW not available, using Adam without weight decay")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, **clip_kwargs)
    print(f"  Using Adam optimizer (lr={learning_rate})")
    return optimizer


def _build_callbacks(
    val_data,
    use_early_stopping: bool,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    reduce_lr_patience: int,
    reduce_lr_factor: float,
    min_lr: float,
) -> list:
    """Build training callbacks for regularization."""
    
    callbacks = []
    
    if val_data is None:
        return callbacks
    
    # ReduceLROnPlateau - reduce LR when val_loss stops improving
    if reduce_lr_patience > 0:
        print(
            f"  Using ReduceLROnPlateau: patience={reduce_lr_patience}, "
            f"factor={reduce_lr_factor}, min_lr={min_lr}"
        )
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                min_lr=min_lr,
                verbose=1,
            )
        )
    
    # Early stopping - stop when val_loss stops improving
    if use_early_stopping:
        print(
            f"  Using EarlyStopping: patience={early_stopping_patience}, "
            f"min_delta={early_stopping_min_delta}"
        )
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                restore_best_weights=True,
                verbose=1,
            )
        )
    
    return callbacks


def evaluate_model(
    sim: nengo_dl.Simulator,
    inp,
    p_out,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate model on test set.
    
    Args:
        sim: nengo_dl Simulator
        inp: Input node
        p_out: Output probe
        X_test: Test features (n_samples, n_features)
        y_test: Test labels one-hot encoded (n_samples, n_classes)
        
    Returns:
        Dictionary with loss and metrics
    """
    X_test_reshaped = X_test[:, np.newaxis, :]
    y_test_reshaped = y_test[:, np.newaxis, :] if len(y_test.shape) == 2 else y_test
    
    sim.compile(
        loss={p_out: tf.keras.losses.CategoricalCrossentropy(from_logits=True)},
        metrics={p_out: ["accuracy"]}
    )
    
    evl = sim.evaluate(
        {inp: X_test_reshaped},
        {p_out: y_test_reshaped}
    )
    
    return evl
