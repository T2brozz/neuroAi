import nengo
import numpy as np
from typing import Tuple

# Disable GPU to prevent TensorFlow/nengo_dl crashes when CUDA is unavailable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
print("TensorFlow running on CPU only")

def build_simple_snn(
    n_features: int,
    n_classes: int,
    n_neurons_hidden: int = 128,
    synapse: float = None,
    spiking: bool = False,
    homogeneous: bool = True,
    seed: int | None = 42,
) -> Tuple[nengo.Network, nengo.Node, nengo.Probe]:
    """
    Build a minimal single-hidden-layer SNN for classification.

    Architecture:
        input (n_features) -> hidden neurons (ReLU) -> linear readout (n_classes)

    This architecture follows nengo_dl best practices:
    - Connect directly to .neurons for trainable weights
    - Use RectifiedLinear for gradient-based training
    - Use Glorot initialization via nengo_dl.dists

    Args:
        n_features: Number of input features.
        n_classes: Number of output classes.
        n_neurons_hidden: Number of neurons in hidden layer (default 128).
        synapse: Synaptic filter time constant (None = no filtering).
        spiking: If True, use LIF neurons; if False, use RectifiedLinear.
        homogeneous: If True, all neurons have identical gains/biases (uniform).
                     If False, use diverse neuron parameters (heterogeneous).
        seed: Random seed for reproducibility.

    Returns:
        (network, input_node, output_probe)
    """
    import nengo_dl
    
    # Choose neuron type - RectifiedLinear is differentiable and works with single timestep
    if spiking:
        neuron_type = nengo.LIF()
    else:
        neuron_type = nengo.RectifiedLinear()

    with nengo.Network(seed=seed) as net:
        # Configure neuron parameters based on homogeneous/heterogeneous setting
        if homogeneous:
            # Homogeneous: all neurons have identical gains and biases
            net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
            net.config[nengo.Ensemble].bias = nengo.dists.Choice([0])
        else:
            # Heterogeneous: diverse neuron parameters for richer dynamics
            # Gains sampled from uniform distribution around 1
            net.config[nengo.Ensemble].gain = nengo.dists.Uniform(0.5, 1.5)
            # Biases sampled to create diverse activation thresholds
            net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-0.5, 0.5)
        
        net.config[nengo.Connection].synapse = synapse
        net.config[nengo.Connection].transform = nengo_dl.dists.Glorot()

        # Input node
        inp = nengo.Node(output=np.zeros(n_features))

        # Hidden layer: connect directly to .neurons for trainable weights
        # dimensions=1 is standard when connecting to neurons directly
        hidden = nengo.Ensemble(
            n_neurons=n_neurons_hidden,
            dimensions=1,
            neuron_type=neuron_type,
        )
        nengo.Connection(inp, hidden.neurons)

        # Output layer: linear readout from hidden neurons
        out = nengo.Node(size_in=n_classes)
        nengo.Connection(hidden.neurons, out)

        # Probe raw logits
        p_out = nengo.Probe(out, synapse=None)

    return net, inp, p_out


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

            # Probe raw logits - NO softmax (blocks gradient flow)
            p_out = nengo.Probe(out, synapse=None)
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

            # Probe raw output - NO softmax (blocks gradient flow)
            p_out = nengo.Probe(out, synapse=synapse_fast)

    return net, inp, p_out
