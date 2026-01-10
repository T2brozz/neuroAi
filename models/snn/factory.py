import nengo
import numpy as np
from typing import Tuple

def _softmax_vec(x: np.ndarray) -> np.ndarray:
    # numerically stable softmax
    z = x - np.max(x)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)

def _softmax_node_fn(t, x):
    return _softmax_vec(x)

def build_snn(
    n_features: int,
    n_classes: int,
    n_neurons_hidden: int = 200,
    synapse_fast: float = 0.005,   # fast integration for event bins
    synapse_slow: float = 0.03,    # slower memory in recurrence
    use_linear_readout: bool = True,
    seed: int | None = 42,
) -> Tuple[nengo.Network, nengo.Node, nengo.Probe]:

    rng = np.random.default_rng(seed)

    reduced_dim = min(128, max(n_features // 10, 64))

    with nengo.Network(seed=seed) as net:
        # Input node: size_in=0 means it's an input (receives data from outside)
        inp = nengo.Node(output=np.zeros(n_features))

        # Random projection (fixed)
        reduction_transform = rng.normal(
            0.0, 1.0 / np.sqrt(n_features), (reduced_dim, n_features)
        )

        reduced = nengo.Ensemble(
            n_neurons=128,
            dimensions=reduced_dim,
            neuron_type=nengo.LIF(),
        )
        nengo.Connection(inp, reduced, synapse=synapse_fast, transform=reduction_transform)

        hidden = nengo.Ensemble(
            n_neurons=n_neurons_hidden,
            dimensions=reduced_dim,
            neuron_type=nengo.LIF(),
        )

        # Reduced -> hidden should usually be filtered for event streams
        hidden_transform = rng.normal(
            0.0, 1.0 / np.sqrt(reduced_dim), (reduced_dim, reduced_dim)
        )
        nengo.Connection(reduced, hidden, synapse=synapse_fast, transform=hidden_transform)

        # Stable-ish recurrence: start with a scaled identity + small noise
        # (acts like leaky memory rather than chaotic reservoir)
        recur = 0.7 * np.eye(reduced_dim) + 0.05 * rng.normal(
            0.0, 1.0 / np.sqrt(reduced_dim), (reduced_dim, reduced_dim)
        )
        nengo.Connection(hidden, hidden, synapse=synapse_slow, transform=recur)

        if use_linear_readout:
            # Linear readout Node (easy to train / interpret)
            out = nengo.Node(size_in=n_classes)
            # Start with small weights; train later (NengoDL / offline regression / PES)
            w = rng.normal(0.0, 0.01, (n_classes, reduced_dim))
            nengo.Connection(hidden, out, synapse=synapse_fast, transform=w)

            # Softmax probabilities
            probs = nengo.Node(_softmax_node_fn, size_in=n_classes, size_out=n_classes)
            nengo.Connection(out, probs, synapse=None)
            p_out = nengo.Probe(probs, synapse=None)
        else:
            # Spiking output (works, but training can be fussier)
            out = nengo.Ensemble(
                n_neurons=2 * n_classes,
                dimensions=n_classes,
                neuron_type=nengo.LIF(),
            )
            hid_out_transform = rng.normal(
                0.0, np.sqrt(2.0 / (reduced_dim + n_classes)), (n_classes, reduced_dim)
            )
            nengo.Connection(hidden, out, synapse=synapse_fast, transform=hid_out_transform)

            # Softmax over filtered spikes for stability
            probs = nengo.Node(_softmax_node_fn, size_in=n_classes, size_out=n_classes)
            nengo.Connection(out, probs, synapse=synapse_fast)
            p_out = nengo.Probe(probs, synapse=None)

    return net, inp, p_out
