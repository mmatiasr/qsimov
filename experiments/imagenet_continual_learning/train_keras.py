"""Train the initial models for the continual learning experiment.

The path selector is trained on the FULL training set (all 100 classes) so
that its frozen convolutional features and dense intermediate layers cover the
entire label space.  Training on only a subset would mean the features cannot
represent classes outside that subset, making the class-incremental continual
learning test unfair.

Three models are saved:

  vgg16_path_selector_softmax_model.tf  — Full-data softmax model.
      Used by QsimovGradient and as starting point for standard fine-tuning.

  vgg16_path_selector_linear_model.tf   — Full-data linear-output model.
      Required by QsimovLinearSystem (check_last_layer_linear() forbids softmax).

  vgg16_standard_model.tf               — Same as softmax model; explicit copy
      for standard fine-tuning baseline to avoid shared-object confusion.
"""

import os
import pickle
import argparse
import multiprocessing as mp
import numpy as np
from experiments.path_utils import get_imagenet_continual_learning_results_dir


def make_imports():
    global tf, keras, init_tensorflow, AccumulatedEpochTimeTracker
    global load_dataset
    global path_selector_vgg16_softmax, path_selector_vgg16_linear, get_optimizer

    import tensorflow as tf
    from tensorflow import keras
    from experiments.mnist_speed_loss.tf_keras.utils import (
        init_tensorflow,
        AccumulatedEpochTimeTracker,
    )
    from experiments.imagenet_continual_learning.preprocess_data import load_dataset
    from experiments.imagenet_continual_learning.keras_model_factory import (
        path_selector_vgg16_softmax,
        path_selector_vgg16_linear,
        get_optimizer,
    )


BATCH_SIZE = 64


def train_model(model, train_x, train_y, test_x, test_y, epochs):
    time_tracker = AccumulatedEpochTimeTracker()
    history = model.fit(
        x=train_x,
        y=train_y,
        validation_data=(test_x, test_y),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[time_tracker],
    )
    history.history["time(s)"] = time_tracker.times
    return history


def execute_logic(results_dir, tag, x_train, y_train, x_test, y_test, epochs):
    from experiments.imagenet_continual_learning.preprocess_data import NUM_LABELS

    if tag == "path_selector_linear":
        model = path_selector_vgg16_linear()
        model.compile(
            loss="mse",
            optimizer=get_optimizer("vgg16"),
            metrics=["mae"],
        )
        # linear model needs one-hot targets
        ty = np.eye(NUM_LABELS, dtype=np.float32)[y_train.astype(int)]
        vy = np.eye(NUM_LABELS, dtype=np.float32)[y_test.astype(int)]
    else:
        model = path_selector_vgg16_softmax()
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=get_optimizer("vgg16"),
            metrics=["accuracy"],
        )
        ty, vy = y_train, y_test

    history = train_model(model, x_train, ty, x_test, vy, epochs)

    model.save(f"{results_dir}/vgg16_{tag}_model.tf", save_format="tf")

    with open(f"{results_dir}/vgg16_{tag}_history.pkl", "wb") as f:
        pickle.dump(history.history, f)

    with open(f"{results_dir}/vgg16_{tag}_model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # Standard baseline is just a copy of the softmax model
    if tag == "path_selector_softmax":
        model.save(f"{results_dir}/vgg16_standard_model.tf", save_format="tf")

    print(f"\nSaved {tag} model to {results_dir}")


def main(args):
    results_dir = get_imagenet_continual_learning_results_dir(args.processor)
    os.makedirs(results_dir, exist_ok=True)

    tags = ["path_selector_softmax", "path_selector_linear"]

    def main_subprocess(tag):
        make_imports()
        init_tensorflow(tf, args.processor)
        # Train on the FULL dataset so that path selector features cover all classes
        x_train, y_train, x_test, y_test = load_dataset()
        execute_logic(results_dir, tag, x_train, y_train, x_test, y_test, args.epochs)

    for tag in tags:
        print(f"\n\nTraining {tag} model on FULL data (all classes)\n")
        p = mp.Process(target=main_subprocess, args=(tag,))
        p.start()
        p.join()


###############################################################################
# CLI Parser
###############################################################################


class TrainModelsParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_arguments()

    def add_arguments(self):
        self.add_argument("--processor", choices=["cpu", "gpu"], default="gpu")
        self.add_argument("--epochs", type=int, required=True)


if __name__ == "__main__":
    main(TrainModelsParser().parse_args())
