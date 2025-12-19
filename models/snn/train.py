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
    checkpoint_dir: Path = None,
    use_early_stopping: bool = True,
    early_stopping_patience: int = 3,
    early_stopping_min_delta: float = 0.0,
) -> tuple[tf.keras.callbacks.History, nengo_dl.Simulator]:
    """
    Train a spiking neural network.
    
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
        checkpoint_dir: Directory to save checkpoints
        use_early_stopping: Whether to enable early stopping based on validation loss
        early_stopping_patience: Number of epochs with no improvement before stopping
        early_stopping_min_delta: Minimum change in the monitored quantity to qualify as improvement
        
    Returns:
        Tuple of (training history, simulator)
    """
    print("Creating simulator...")
    actual_batch_size = min(batch_size, len(X_train))
    
    sim = nengo_dl.Simulator(net, minibatch_size=None)

    print("Compiling model...")
    sim.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={p_out: tf.keras.losses.CategoricalCrossentropy(from_logits=True)},
        metrics={p_out: ["accuracy"]},
    )

    # Reshape data for nengo_dl: (batch, time, features)
    X_train_reshaped = X_train[:, np.newaxis, :]  # Add time dimension
    y_train_reshaped = y_train[:, np.newaxis, :] if len(y_train.shape) == 2 else y_train
    
    X_val_reshaped = X_val[:, np.newaxis, :] if X_val is not None else None
    y_val_reshaped = y_val[:, np.newaxis, :] if y_val is not None and len(y_val.shape) == 2 else y_val
    
    val_data = None
    if X_val is not None and y_val is not None:
        val_data = ({inp: X_val_reshaped}, {p_out: y_val_reshaped})

    # Configure callbacks (early stopping if validation data is available)
    callbacks = []
    if use_early_stopping and val_data is not None:
        print(
            f"Using EarlyStopping on 'val_loss' with "
            f"patience={early_stopping_patience}, min_delta={early_stopping_min_delta}"
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
    
    print(f"\nTraining for {epochs} epochs...")
    print(f"  Input shape: {X_train_reshaped.shape}")
    print(f"  Label shape: {y_train_reshaped.shape}")
    
    history = sim.fit(
        {inp: X_train_reshaped},
        {p_out: y_train_reshaped},
        validation_data=val_data,
        callbacks=callbacks if callbacks else None,
        epochs=epochs,
        verbose=1,
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


def evaluate_model(
    sim: nengo_dl.Simulator,
    inp,
    p_out,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """
    Evaluate model on test set.
    
    Args:
        sim: nengo_dl Simulator
        inp: Input node
        p_out: Output probe
        X_test: Test features (n_samples, n_features)
        y_test: Test labels one-hot encoded (n_samples, n_classes)
        
    Returns:
        Test loss
    """
    X_test_reshaped = X_test[:, np.newaxis, :]
    y_test_reshaped = y_test[:, np.newaxis, :] if len(y_test.shape) == 2 else y_test
    
    sim.compile(
        loss={p_out: tf.keras.losses.CategoricalCrossentropy(from_logits=True)},
        metrics={p_out: ["accuracy"]},
    )
    
    evl = sim.evaluate(
        {inp: X_test_reshaped},
        {p_out: y_test_reshaped},
        verbose=0,
    )
    
    return evl["loss"]
