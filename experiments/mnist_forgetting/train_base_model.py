"""Train base MNIST model on phase 1 (classes 0-4) only.

Uses MSE loss with linear final activation — required by QsimovLinearSystem.
Saves to results_dir/base_model.tf.
"""

import os
import argparse
import numpy as np

from experiments.path_utils import get_mnist_forgetting_results_dir
from experiments.mnist_forgetting.preprocess_data import load_data, get_data_dir

BATCH_SIZE = 32
SEED = 42


def make_imports():
    global tf, keras, init_tensorflow, accuracy, build_mnist, AccumulatedEpochTimeTracker

    import tensorflow as tf
    from tensorflow import keras
    from experiments.mnist_speed_loss.tf_keras.utils import (
        init_tensorflow,
        accuracy,
        AccumulatedEpochTimeTracker,
    )
    from experiments.mnist_speed_loss.tf_keras.model_factory import build_mnist


def main(args):
    make_imports()
    tf.keras.utils.set_random_seed(SEED)
    init_tensorflow(tf, args.processor)

    data_dir = get_data_dir()
    data = load_data(data_dir)

    results_dir = get_mnist_forgetting_results_dir("keras", args.processor)
    os.makedirs(results_dir, exist_ok=True)

    image_shape = data["phase1_train_x"].shape[1:]
    model = build_mnist(image_shape, loss="mse")

    time_tracker = AccumulatedEpochTimeTracker()
    model.fit(
        data["phase1_train_x"],
        data["phase1_train_y"],
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        validation_data=(data["test_phase1_x"], data["test_phase1_y"]),
        callbacks=[time_tracker],
    )

    acc_old = float(np.mean(
        np.argmax(model.predict(data["test_phase1_x"], verbose=0), axis=1) ==
        np.argmax(data["test_phase1_y"], axis=1)
    ))
    acc_new = float(np.mean(
        np.argmax(model.predict(data["test_phase2_x"], verbose=0), axis=1) ==
        np.argmax(data["test_phase2_y"], axis=1)
    ))
    print(f"\nBase model — acc_old={acc_old:.4f}  acc_new={acc_new:.4f}")
    print("(acc_new is near-random: model only saw classes 0-4)")

    model_path = os.path.join(results_dir, "base_model.tf")
    model.save(model_path, save_format="tf")
    print(f"Base model saved to {model_path}")


class TrainBaseModelParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="cpu")
        self.add_argument("--epochs", type=int, default=10)


if __name__ == "__main__":
    main(TrainBaseModelParser().parse_args())
