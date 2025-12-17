import nengo
import numpy as np
from typing import Tuple

def build_snn(
    n_features: int,
    n_classes: int,
    n_neurons_hidden: int = 200,
    synapse: float = 0.01,
) -> Tuple[nengo.Network, nengo.Node, nengo.Probe]:
    """
    Build a simple SNN with dimensionality reduction for high-dimensional input.

    Architecture:
    - Input (n_features) -> Dimensionality reduction layer (128 dims)
    - Hidden recurrent ensemble (256 neurons, 128 dims)
    - Output ensemble (n_classes dims)
    """
    rng = np.random.default_rng()
    
    # For high-dimensional input, use a bottleneck
    reduced_dim = min(128, max(n_features // 10, 64))

    with nengo.Network() as net:
        # Input node - nengo_dl will feed data directly to this
        # size_in=0 means this is a pure input node (for nengo_dl)
        inp = nengo.Node(np.zeros(n_features))

        # Dimensionality reduction layer (fixed random projection)
        reduction_transform = rng.normal(0.0, 1.0 / np.sqrt(n_features), (reduced_dim, n_features))
        reduced = nengo.Ensemble(
            n_neurons=128,
            dimensions=reduced_dim,
            neuron_type=nengo.LIF(),
        )
        # Connect input node directly to reduced ensemble
        nengo.Connection(inp, reduced, synapse=0.01, transform=reduction_transform)

        # Hidden recurrent ensemble for temporal dynamics
        hidden = nengo.Ensemble(
            n_neurons=n_neurons_hidden,
            dimensions=reduced_dim,
            neuron_type=nengo.LIF(),
        )

        # Recurrent connection with small random weights
        recur_transform = rng.normal(0.0, 1.0 / np.sqrt(reduced_dim), (reduced_dim, reduced_dim))
        nengo.Connection(hidden, hidden, synapse=synapse, transform=recur_transform)

        # Reduced to hidden connection
        hidden_transform = rng.normal(0.0, 1.0 / np.sqrt(reduced_dim), (reduced_dim, reduced_dim))
        nengo.Connection(reduced, hidden, synapse=None, transform=hidden_transform)

        # Output ensemble
        out = nengo.Ensemble(
            n_neurons=2 * n_classes,
            dimensions=n_classes,
            neuron_type=nengo.LIF(),
        )

        # Hidden to output with random weights
        hid_out_transform = rng.normal(0.0, np.sqrt(2.0 / (reduced_dim + n_classes)), (n_classes, reduced_dim))
        nengo.Connection(hidden, out, synapse=synapse, transform=hid_out_transform)

        p_out = nengo.Probe(out, synapse=None)

    return net, inp, p_out

