import nengo_dl
import numpy as np
from pathlib import Path
from models.snn.factory import build_snn

def load_snn_model(params_path: Path, hparams_path: Path) -> nengo_dl.Simulator:
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

    sim, inp, p_out = build_snn(
        n_features=n_features,
        n_classes=n_classes,
        n_neurons_hidden=best_hidden,
        synapse_fast=best_syn_fast,
        synapse_slow=best_syn_slow,
    )
    
    print(f"Loading SNN model parameters from {params_path}...")
    sim.load_params(str(params_path))
    
    return sim