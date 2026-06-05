import pickle
import os
from experiments.cifar10_gradient_by_splits.train_keras_models import (
    TrainModelsParser,
    split_to_name,
    BATCH_SIZE,
)
from experiments.path_utils import (
    get_cifar10_gradient_by_splits_results_dir as get_results_dir,
)
import numpy as np
import multiprocessing as mp
from experiments.cifar10_gradient_by_splits.preprocess_data import load_dataset
import argparse


# Possibly slow imports
def make_imports():
    global tf, keras, AccumulatedEpochTimeTracker, init_tensorflow
    global load_dataset, get_optimizer
    global KerasPathSelector, KerasQsimovGradient, KerasQsimovLinearSystem

    import tensorflow as tf
    from tensorflow import keras
    from qsimov.keras_path_selector import KerasPathSelector
    from qsimov.keras_qsimov_gradient import KerasQsimovGradient
    from qsimov.keras_qsimov_linear_system import KerasQsimovLinearSystem
    from experiments.mnist_speed_loss.tf_keras.utils import (
        init_tensorflow,
        AccumulatedEpochTimeTracker,
    )
    from experiments.cifar10_gradient_by_splits.preprocess_data import (
        load_dataset,
    )
    from experiments.cifar10_gradient_by_splits.keras_model_factory import (
        get_optimizer,
    )


###############################################################################
# Logic
###############################################################################

N_CLASSES = 10
QR_SHRINKAGE = 10


def _n_paths(path_selector):
    return int(np.sum(path_selector.output_masks_, axis=1).max())


def save_gradient_results(name, model_name, model, history, results_dir):
    name += "_qsimov"
    with open(f"{results_dir}/{name}_{model_name}_history.pkl", "wb") as f:
        pickle.dump(history.history, f)
    with open(f"{results_dir}/number_of_paths_{model_name}.txt", "w") as f:
        f.write(str(_n_paths(model._path_selector)))


def save_linear_results(name, model_name, results, n_paths, results_dir):
    name += "_qsimov_linear"
    with open(f"{results_dir}/{name}_{model_name}_results.pkl", "wb") as f:
        pickle.dump(results, f)
    with open(f"{results_dir}/number_of_paths_linear_{model_name}.txt", "w") as f:
        f.write(str(n_paths))


def train_gradient_model(model, train_x, train_y, test_x, test_y, args):
    time_tracker = AccumulatedEpochTimeTracker()
    history = model.fit(
        X=train_x,
        Y=train_y,
        validation_data=(test_x, test_y),
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        callbacks=[time_tracker],
    )
    history.history["time(s)"] = time_tracker.times
    return history


def load_base_model(results_dir, model_name, split, linear=False):
    suffix = "_path_selector_linear" if linear else "_path_selector"
    model_file = f"{split_to_name(split)}{suffix}_{model_name}_model.tf"
    return keras.models.load_model(os.path.join(results_dir, model_file))


def make_qsimov_gradient_model(results_dir, split, args):
    path_selector = KerasPathSelector(
        load_base_model(results_dir, args.model_name, split, linear=False),
        args.initial_layer,
    )
    qsimov_gradient = KerasQsimovGradient(path_selector)
    qsimov_gradient.compile(
        loss="categorical_crossentropy",
        optimizer=get_optimizer(model=args.model_name),
        metrics=["accuracy"],
    )
    return qsimov_gradient


def run_qsimov_linear(results_dir, split, args, train_x, train_y, test_x, test_y):
    """Solve QsimovLinearSystem on CIFAR-10 split data."""
    import time

    path_selector = KerasPathSelector(
        load_base_model(results_dir, args.model_name, split, linear=True),
        args.initial_layer,
    )
    n_paths = _n_paths(path_selector)

    qls = KerasQsimovLinearSystem(
        path_selector,
        solver="back_substitution",
        qr_shrinkage_factor=QR_SHRINKAGE,
        absolute_cutoff=1e-6,
        relative_cutoff=1e6,
        verbose=1,
    )

    t0 = time.time()
    qls.fit(train_x, train_y, batch_size=BATCH_SIZE)
    train_time = time.time() - t0

    y_pred = qls.predict(test_x)
    test_acc = float(np.mean(np.argmax(y_pred, axis=1) == np.argmax(test_y, axis=1)))
    y_pred_train = qls.predict(train_x)
    train_acc = float(np.mean(np.argmax(y_pred_train, axis=1) == np.argmax(train_y, axis=1)))

    results = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_time": train_time,
        "n_paths": n_paths,
    }
    save_linear_results(
        split_to_name(split), args.model_name, results, n_paths, results_dir
    )
    print(f"\nLinear system — split={split_to_name(split)}: "
          f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
          f"time={train_time:.1f}s n_paths={n_paths}")


def execute_logic(results_dir, split, args):
    train_x, train_y, test_x, test_y = load_dataset()

    # --- Gradient method ---
    model = make_qsimov_gradient_model(results_dir, split, args)
    history = train_gradient_model(model, train_x, train_y, test_x, test_y, args)
    save_gradient_results(
        split_to_name(split), args.model_name, model, history, results_dir
    )

    # --- Linear system method ---
    linear_model_path = os.path.join(
        results_dir,
        f"{split_to_name(split)}_path_selector_linear_{args.model_name}_model.tf",
    )
    if os.path.exists(linear_model_path):
        run_qsimov_linear(results_dir, split, args, train_x, train_y, test_x, test_y)
    else:
        print(f"\nSkipping linear system for split={split_to_name(split)}: "
              f"linear model not found at {linear_model_path}. "
              "Re-run train_keras_models.py --train-path-selector to generate it.")


def main(args):
    # one process for each split to avoid memory issues
    def main_subprocess(split):
        # make slow imports (those that need to import tensorflow)
        make_imports()

        # init tensorflow
        init_tensorflow(tf, args.processor)

        # train the different combinations of models and fractions of the data
        execute_logic(results_dir, split, args)

    # directories to save the models
    results_dir = get_results_dir("keras", args.processor)
    os.makedirs(results_dir, exist_ok=True)

    # get results for each split in a different process
    for split in args.splits:
        print(f"\n\nTraining model with split: {split_to_name(split)}\n")
        p = mp.Process(target=main_subprocess, args=(split,))
        p.start()
        p.join()


###############################################################################
# CLI Parser
###############################################################################
class TrainQsimovModelsParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_arguments()

    def add_arguments(self):
        TrainModelsParser.add_processor_argument(self)
        TrainModelsParser.add_epochs_argument(self)
        TrainModelsParser.add_splits_argument(self)
        TrainModelsParser.add_model_name_argument(self)
        self.add_initial_layer_argument()

    def add_initial_layer_argument(self):
        self.add_argument(
            "--initial-layer",
            type=int,
            default=-2,
            help="Initial layer of the qsimov model",
        )


if __name__ == "__main__":
    args = TrainQsimovModelsParser().parse_args()
    main(args)
