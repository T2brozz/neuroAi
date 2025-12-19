from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import tensorflow as tf
import ray
from ray import tune

from models.snn.factory import build_snn
from models.snn.train import train_model


def _ray_trainable(config: Dict[str, Any]) -> None:
    """Ray Tune trainable that uses the existing train_model function.

    Expects the following in the config:
    - X_train, y_train, X_val, y_val: numpy arrays
    - n_features, n_classes: ints
    - epochs: int (training epochs for this trial)
    - batch_size, learning_rate, n_neurons_hidden, synapse: hyperparameters
    """
    X_train: np.ndarray = config["X_train"]
    y_train: np.ndarray = config["y_train"]
    X_val: np.ndarray = config["X_val"]
    y_val: np.ndarray = config["y_val"]
    n_features: int = config["n_features"]
    n_classes: int = config["n_classes"]

    n_hidden = int(config["n_neurons_hidden"])
    syn = float(config["synapse"])
    learning_rate = float(config["learning_rate"])
    batch_size = int(config["batch_size"])
    epochs = int(config.get("epochs", 5))

    print("\n" + "=" * 80)
    print("[Ray Tune] Starting trial with configuration:")
    print(f"  hidden={n_hidden}")
    print(f"  synapse={syn}")
    print(f"  learning_rate={learning_rate:.2e}")
    print(f"  batch_size={batch_size}")
    print(f"  epochs={epochs}")

    # Build network with current hyperparameters
    net, inp, p_out = build_snn(
        n_features=n_features,
        n_classes=n_classes,
        n_neurons_hidden=n_hidden,
        synapse=syn,
    )

    # Train using the shared training function
    history, sim = train_model(
        net=net,
        inp=inp,
        p_out=p_out,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_dir=None,
    )

    # Get best validation accuracy if available
    val_acc = None
    for key, values in history.history.items():
        if key.startswith("val_") and "accuracy" in key:
            val_acc = float(max(values))
            break

    if val_acc is None:
        raise RuntimeError(
            f"Could not find a validation accuracy metric in history keys: "
            f"{list(history.history.keys())}"
        )

    print(f"[Ray Tune] Trial finished with best val_accuracy={val_acc:.4f}")

    # Report result to Ray Tune
    tune.report(val_accuracy=val_acc)

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
    if not ray.is_initialized():
        print("[Ray Tune] Initializing Ray runtime...")
        # local_mode=True runs Ray Tune in-process, which is much more
        # stable inside Jupyter notebooks and with libraries that use
        # heavy native code (TensorFlow, nengo_dl).
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            local_mode=True,
            num_cpus=1,
            log_to_driver=True,
        )

    print("[Ray Tune] Preparing hyperparameter search...")
    print(f"  max_epochs      = {max_epochs}")
    print(f"  num_samples     = {num_samples}")
    print(f"  project_name    = {project_name}")

    # Define search space
    search_space = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "n_features": n_features,
        "n_classes": n_classes,
        "epochs": max_epochs,
        "n_neurons_hidden": tune.randint(50, 257),
        "synapse": tune.uniform(0.001, 0.05),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
    }

    print("[Ray Tune] Starting tuning run...")

    analysis = tune.run(
        _ray_trainable,
        config=search_space,
        num_samples=num_samples,
        metric="val_accuracy",
        mode="max",
        name=project_name,
        resources_per_trial={"cpu": 1},
        verbose=1,
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
