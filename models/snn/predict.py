import nengo_dl
import nengo
import numpy as np
from pathlib import Path
from models.snn.factory import build_simple_snn
from typing import NamedTuple


class SNNModel(NamedTuple):
    sim: nengo_dl.Simulator
    inp: nengo.Node
    p_out: nengo.Probe
    n_features: int

def load_snn_model(params_path: Path, hparams_path: Path) -> SNNModel:
    """
    Load SNN model from saved parameters.
    
    Args:
        params_path: Path to saved model parameters.
        hparams_path: Path to hyperparameters file.
    Returns:
        nengo_dl Simulator with loaded parameters.
    """
    print(f"Loading hyperparameters from {hparams_path}")
    hparams = {}
    with hparams_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line and '=' in line:
                key, value = line.split('=', 1)
                hparams[key] = value
    
    best_hidden = int(hparams.get('N_NEURONS_HIDDEN'))
    n_features = int(hparams.get('N_FEATURES'))
    n_classes = int(hparams.get('N_CLASSES'))
    best_homogeneous = bool(int(hparams.get('HOMOGENEOUS')))

    net, inp, p_out = build_simple_snn(
        n_features=n_features,
        n_classes=n_classes,
        n_neurons_hidden=best_hidden,
        spiking=True,
        homogeneous=best_homogeneous   
    )
    
    print(f"Loading SNN model parameters from {params_path}...")

    sim = nengo_dl.Simulator(net)
    sim.load_params(str(params_path))
    
    return SNNModel(sim=sim, inp=inp, p_out=p_out, n_features=n_features)

def predict_snn(model: SNNModel, features: np.ndarray) -> np.ndarray:
    """
    Perform inference using the SNN model.
    
    Args:
        sim: nengo_dl Simulator with loaded SNN model.
        features: Input features for prediction.
    Returns:
        Predicted class probabilities (softmax applied).
    """
    if features is None:
        return None
    
    # Reshape features to match model input (batch, time, features)
    # Training used a single time step, so for 1D feature vectors we add time=1
    if features.ndim == 1:
        input_data = features[np.newaxis, np.newaxis, :]
    elif features.ndim == 2:
        # Assume features is (time, features)
        input_data = features[np.newaxis, :, :]
    else:
        raise ValueError(f"Unsupported features shape {features.shape}; expected 1D or 2D.")
    
    # Run simulation
    output = model.sim.predict({model.inp: input_data})
    
    # Extract output logits
    logits_time = output[model.p_out][0]
    if logits_time.ndim == 2:
        logits = logits_time[-1]
    else:
        logits = logits_time
    
    # Apply softmax to convert logits to probabilities
    # Subtract max for numerical stability
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits)
    
    return probabilities