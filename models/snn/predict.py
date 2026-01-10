import nengo_dl
import nengo
import numpy as np
from pathlib import Path
from models.snn.factory import build_snn
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
    best_syn_fast = float(hparams.get('SYNAPSE_FAST'))
    best_syn_slow = float(hparams.get('SYNAPSE_SLOW'))
    n_features = int(hparams.get('N_FEATURES'))
    n_classes = int(hparams.get('N_CLASSES'))

    net, inp, p_out = build_snn(
        n_features=n_features,
        n_classes=n_classes,
        n_neurons_hidden=best_hidden,
        synapse_fast=best_syn_fast,
        synapse_slow=best_syn_slow,
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
        Predicted class probabilities.
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
    
    # Extract output probabilities
    probs_time = output[model.p_out][0]
    if probs_time.ndim == 2:
        probabilities = probs_time[-1]
    else:
        probabilities = probs_time
    
    return probabilities