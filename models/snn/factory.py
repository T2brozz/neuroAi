import nengo
import numpy as np
import nengo_dl
from typing import Tuple

def build_snn(n_features: int, n_classes: int, n_neurons_hidden: int = 200, synapse: float = 0.01) -> Tuple[nengo.Network, nengo.Node, nengo.Probe]:
    with nengo.Network() as net:
        nengo_dl.configure_settings(trainable=True)

        inp = nengo.Node(np.zeros(n_features))

        hidden = nengo.Ensemble(
            n_neurons=n_neurons_hidden,
            dimensions=n_features,
            neuron_type=nengo.LIF(),
        )

        out = nengo.Node(size_in=n_classes)

        nengo.Connection(inp, hidden, synapse=None,
                         transform=nengo_dl.dists.Glorot())
        nengo.Connection(hidden, out, synapse=synapse,
                         transform=nengo_dl.dists.Glorot())

        p_out = nengo.Probe(out, synapse=None)

    return net, inp, p_out

