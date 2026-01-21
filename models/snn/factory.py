import nengo
import numpy as np
from typing import Tuple

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