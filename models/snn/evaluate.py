import numpy as np
import tensorflow as tf
import nengo_dl

from models.snn.factory import build_snn

def evaluate_model(n_hidden: int, syn: float, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_features: int, n_classes: int) -> tuple[float, nengo_dl.Simulator, tf.keras.callbacks.History]:
    net, inp, p_out = build_snn(
        n_features=n_features,
        n_classes=n_classes,
        n_neurons_hidden=n_hidden,
        synapse=syn,
    )

    sim = nengo_dl.Simulator(net, minibatch_size=32)

    sim.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={p_out: tf.keras.losses.CategoricalCrossentropy(from_logits=True)},
        metrics={p_out: ["accuracy"]},
    )

    history = sim.fit(
        {inp: X_train},
        {p_out: y_train},
        validation_data=({inp: X_val}, {p_out: y_val}),
        epochs=10,
        verbose=0,
    )

    val_acc = history.history["val_p_out_accuracy"][-1]
    sim.close()

    return val_acc, sim, history