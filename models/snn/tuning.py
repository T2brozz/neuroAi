from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import tensorflow as tf
import ray
from ray import tune

from models.snn.factory import build_snn,build_simple_snn
from models.snn.train import train_model, evaluate_model


def _ray_trainable(config: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
    """Ray Tune trainable that uses the existing train_model function.

    Expects the following in the config:
    - X_train, y_train, X_val, y_val: numpy arrays
    - n_features, n_classes: ints
    - epochs: int (training epochs for this trial)
    - batch_size, learning_rate, n_neurons_hidden: hyperparameters
    """
    n_features: int = config["n_features"]
    n_classes: int = config["n_classes"]

    n_hidden = int(config["n_neurons_hidden"])
    learning_rate = float(config["learning_rate"])
    batch_size = int(config["batch_size"])
    weight_decay = float(config.get("weight_decay", 0.0))
    homogeneous = bool(config.get("homogeneous", True))
    spiking = bool(config.get("spiking", False))
    epochs = int(config.get("epochs", 5))

    print("\n" + "=" * 80)
    print("[Ray Tune] Starting trial with configuration:")
    print(f"  hidden={n_hidden}")
    print(f"  learning_rate={learning_rate:.2e}")
    print(f"  batch_size={batch_size}")
    print(f"  homogeneous={homogeneous}")
    print(f"  spiking={spiking}")
    print(f"  epochs={epochs}")

    # Build network with current hyperparameters
    net, inp, p_out = build_simple_snn(
        n_features=n_features,
        n_classes=n_classes,
        n_neurons_hidden=n_hidden,
        synapse=None,  # No synaptic filtering for single timestep training
        spiking=spiking,  # Use RectifiedLinear for gradient-based training
        homogeneous=homogeneous,  # Test both uniform and diverse neuron parameters
    )

    # Train using the shared training function
    # Disable early stopping and validation data for Ray Tune to avoid serialization issues
    history, sim = train_model(
        net=net,
        inp=inp,
        p_out=p_out,
        X_train=X_train,
        y_train=y_train,
        X_val=None,  # Don't pass validation data to avoid Ray serialization issues
        y_val=None,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_dir=None,
        weight_decay=weight_decay,
        use_early_stopping=True,
        early_stopping_min_delta=0.001,
        early_stopping_patience=50,
    )
    
    # Evaluate on validation set using evaluate_model function
    val_results = evaluate_model(sim, inp, p_out, X_val, y_val)
    
    # Find the accuracy key (it contains 'accuracy' in the name)
    accuracy_key = [k for k in val_results.keys() if 'accuracy' in k][0]
    val_acc = float(val_results[accuracy_key])

    print(f"[Ray Tune] Trial finished with val_accuracy={val_acc:.4f}")

    # Report result to Ray Tune
    tune.report(metrics={ "val_accuracy": float(val_acc) })
    

    sim.close()


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
    n_classes: int,
    max_epochs: int = 5,
    num_samples: int = 20,
    project_name: str = "snn_ray_tune",
) -> Tuple[Dict[str, Any], "ray.tune.ExperimentAnalysis"]:
    """Run Ray Tune-based hyperparameter search and return the best config.

    Returns a tuple of (best_config, analysis).
    """
    # Determine project root (two levels up from this file: models/snn/tuning.py -> project root)
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    
    # Shutdown any existing Ray instance to avoid conflicts
    if ray.is_initialized():
        ray.shutdown()
    
    print("[Ray Tune] Initializing Ray runtime...")
    print(f"  project_root    = {project_root}")
    
    # Configure runtime_env so Ray workers can find the 'models' module
    runtime_env = {
        "working_dir": project_root,
        "env_vars": {
            "PYTHONPATH": project_root,
            "CUDA_VISIBLE_DEVICES": "-1",  # Disable GPU in workers
        },
    }
    
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=True,
        log_to_driver=True,
        runtime_env=runtime_env,
    )

    print("[Ray Tune] Preparing hyperparameter search...")
    print(f"  max_epochs      = {max_epochs}")
    print(f"  num_samples     = {num_samples}")
    print(f"  project_name    = {project_name}")

    # Define search space (simplified for the new architecture)
    search_space = {
        "n_features": n_features,
        "n_classes": n_classes,
        "epochs": max_epochs,
        "n_neurons_hidden": tune.randint(64, 257),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "homogeneous": tune.choice([True, False]),  # Test homogeneous vs heterogeneous
        "spiking": tune.choice([True, False]),
    }

    print("[Ray Tune] Starting tuning run...")

    # Use tune.with_parameters to pass large numpy arrays efficiently
    trainable_with_params = tune.with_parameters(
        _ray_trainable,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    analysis = tune.run(
        trainable_with_params,
        config=search_space,
        num_samples=num_samples,
        metric="val_accuracy",
        mode="max",
        name=project_name,
        max_concurrent_trials=3,
        progress_reporter=None
    )

    best_config = analysis.get_best_config(metric="val_accuracy", mode="max")
    best_result = analysis.get_best_trial(metric="val_accuracy", mode="max").last_result
    best_val = best_result.get("val_accuracy", None)

    print("[Ray Tune] Tuning complete.")
    print("[Ray Tune] Best configuration:")
    for k, v in best_config.items():
        if k in {"X_train", "y_train", "X_val", "y_val"}:
            continue
        print(f"  {k} = {v}")
    if best_val is not None:
        print(f"[Ray Tune] Best val_accuracy = {best_val:.4f}")

    return best_config, analysis
