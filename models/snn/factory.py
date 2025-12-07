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
    Build a simple SNN using pure Nengo (no TensorFlow/Nengo-DL).

    Notes:
    - Connections use fixed random transforms (Gaussian) for a baseline.
    - For learning, consider Nengo's PES/Voja rules or offline training.
    """
    rng = np.random.default_rng()

    with nengo.Network() as net:
        inp = nengo.Node(size_in=n_features)

        # Hidden recurrent ensemble for temporal dynamics
        hidden = nengo.Ensemble(
            n_neurons=n_neurons_hidden,
            dimensions=n_features,
            neuron_type=nengo.LIF(),
        )

        # Recurrent connection with small random weights
        recur_transform = rng.normal(0.0, 1.0 / np.sqrt(n_features), (n_features, n_features))
        nengo.Connection(hidden, hidden, synapse=synapse, transform=recur_transform)

        # Input to hidden with random weights
        in_transform = rng.normal(0.0, 1.0 / np.sqrt(n_features), (n_features, n_features))
        nengo.Connection(inp, hidden, synapse=None, transform=in_transform)

        # Output ensemble
        out = nengo.Ensemble(
            n_neurons=2 * n_classes,
            dimensions=n_classes,
            neuron_type=nengo.LIF(),
        )

        # Hidden to output with random weights
        hid_out_transform = rng.normal(0.0, 1.0 / np.sqrt(n_features), (n_classes, n_features))
        nengo.Connection(hidden, out, synapse=synapse, transform=hid_out_transform)

        p_out = nengo.Probe(out, synapse=None)

    return net, inp, p_out

