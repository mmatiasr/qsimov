"""Train base LeNet on CIFAR-10 phase 1 (classes 0-4) with MSE + linear output.

MSE with linear final activation is required by QsimovLinearSystem.
initial_layer=-1 targets the last Dense(32,relu)+Dense(10,linear) block,
which gives 33 paths — manageable on CPU.
"""

import os
import argparse
import numpy as np
from tensorflow import keras as kr

from experiments.path_utils import get_cifar10_forgetting_results_dir
from experiments.cifar10_forgetting.preprocess_data import load_data, get_data_dir

BATCH_SIZE = 64
SEED = 42


def accuracy(y_true, y_pred):
    return kr.backend.mean(
        kr.backend.argmax(y_true, axis=1) == kr.backend.argmax(y_pred, axis=1)
    )


def build_lenet_mse(image_shape=(32, 32, 3), n_classes=10):
    """LeNet variant with linear output for QsimovLinearSystem compatibility."""
    model = kr.Sequential([
        kr.layers.Conv2D(20, 5, padding="same", activation="relu", input_shape=image_shape),
        kr.layers.MaxPooling2D(),
        kr.layers.Conv2D(50, 5, padding="same", activation="relu"),
        kr.layers.MaxPooling2D(),
        kr.layers.Flatten(),
        kr.layers.Dense(500, activation="relu"),
        kr.layers.Dense(32, activation="relu"),
        kr.layers.Dense(n_classes, activation="linear"),
    ])
    model.compile(loss="mse", optimizer="adam", metrics=[accuracy])
    model.summary()
    return model


def main(args):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)

    if args.processor == "cpu":
        tf.config.set_visible_devices([], "GPU")

    data_dir = get_data_dir()
    data = load_data(data_dir)
    results_dir = get_cifar10_forgetting_results_dir("keras", args.processor)
    os.makedirs(results_dir, exist_ok=True)

    image_shape = data["phase1_train_x"].shape[1:]
    model = build_lenet_mse(image_shape)

    model.fit(
        data["phase1_train_x"],
        data["phase1_train_y"],
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        validation_data=(data["test_phase1_x"], data["test_phase1_y"]),
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

    model_path = os.path.join(results_dir, "base_model.tf")
    model.save(model_path, save_format="tf")
    print(f"Saved to {model_path}")


class TrainBaseModelParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="cpu")
        self.add_argument("--epochs", type=int, default=20)


if __name__ == "__main__":
    main(TrainBaseModelParser().parse_args())
