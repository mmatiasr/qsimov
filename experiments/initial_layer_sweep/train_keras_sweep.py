"""Sweep initial_layer values for the Qsimov PathSelector on ImageNet.

Measures the effect of the initial_layer parameter on:
  - Number of paths (grows exponentially with depth)
  - PathSelector build time
  - QsimovLinearSystem accuracy and training time
  - QsimovGradient accuracy and training time

The initial_layer controls how deep into the network the path-selection
algorithm reaches (φ_R).  Deeper initial_layer → more layers in φ_R →
exponentially more paths → higher memory/compute cost but richer feature space.

At some depth the number of paths becomes infeasible.  This experiment
measures where that threshold lies for the VGG16+ImageNet setup and what
accuracy gain (if any) is achieved by going deeper.

initial_layer values tested: -1, -2, -3
  -1  → φ_R = [Dense(100)]                          (128+1  = 129 paths baseline)
  -2  → φ_R = [Dense(128,relu), Dense(100)]
  -3  → φ_R = [Dense(2048,relu), Dense(128,relu), Dense(100)]

If n_paths > MAX_PATHS the training step is skipped and recorded as
"infeasible" so the sweep always completes without OOM.

Prerequisites
-------------
Requires pre-trained models from imagenet_continual_learning/train_keras.py:
    {continual_learning_results_dir}/vgg16_path_selector_linear_model.tf
    {continual_learning_results_dir}/vgg16_path_selector_softmax_model.tf
"""

import os
import pickle
import time
import argparse
import multiprocessing as mp
import numpy as np
from experiments.path_utils import (
    get_initial_layer_sweep_results_dir,
    get_imagenet_continual_learning_results_dir,
)

INITIAL_LAYERS = [-1, -2, -3]
BATCH_SIZE = 64
EPOCHS = 5
QR_SHRINKAGE = 10
MAX_PATHS = 500_000  # skip training if n_paths exceeds this


def make_imports():
    global tf, keras, init_tensorflow
    global KerasPathSelector, KerasQsimovLinearSystem, KerasQsimovGradient
    global NUM_LABELS, load_dataset

    import tensorflow as tf
    from tensorflow import keras
    from experiments.mnist_speed_loss.tf_keras.utils import init_tensorflow
    from qsimov.keras_path_selector import KerasPathSelector
    from qsimov.keras_qsimov_linear_system import KerasQsimovLinearSystem
    from qsimov.keras_qsimov_gradient import KerasQsimovGradient
    from experiments.imagenet_subset_by_splits.preprocess_data import NUM_LABELS
    from experiments.imagenet_continual_learning.preprocess_data import load_dataset


def _check_prerequisites(cl_results_dir):
    required = [
        "vgg16_path_selector_linear_model.tf",
        "vgg16_path_selector_softmax_model.tf",
    ]
    missing = [
        f for f in required
        if not os.path.exists(os.path.join(cl_results_dir, f))
    ]
    if missing:
        raise FileNotFoundError(
            "Missing pre-trained model(s) in continual_learning results dir:\n"
            + "\n".join(f"  {cl_results_dir}/{f}" for f in missing)
            + "\nRun: python -m experiments.imagenet_continual_learning.train_keras "
            "--epochs 20 --processor gpu"
        )


def accuracy_top1(y_pred, y_true_int):
    return float(np.mean(np.argmax(y_pred, axis=1) == y_true_int))


def run_one_layer(initial_layer, x_train, y_train, x_test, y_test,
                  cl_results_dir, device):
    """Run both linear and gradient methods for one initial_layer value."""
    entry = {"initial_layer": initial_layer}

    # --- Build linear PathSelector ---
    base_linear = keras.models.load_model(
        f"{cl_results_dir}/vgg16_path_selector_linear_model.tf"
    )
    t0 = time.time()
    ps_linear = KerasPathSelector(
        base_linear, initial_layer=initial_layer, device=device, verbose=1
    )
    entry["build_time_linear"] = time.time() - t0
    n_paths = int(ps_linear.output_masks_.sum())
    entry["n_paths"] = n_paths
    print(f"\ninitial_layer={initial_layer}: n_paths={n_paths}, "
          f"build_time={entry['build_time_linear']:.1f}s")

    if n_paths > MAX_PATHS:
        entry["linear"] = {"status": "infeasible", "n_paths": n_paths}
        entry["gradient"] = {"status": "infeasible", "n_paths": n_paths}
        print(f"  Skipping training: n_paths={n_paths} > MAX_PATHS={MAX_PATHS}")
        return entry

    # --- QsimovLinearSystem ---
    y_onehot = np.eye(NUM_LABELS, dtype=np.float32)[y_train.astype(int)]
    qls = KerasQsimovLinearSystem(
        ps_linear,
        solver="back_substitution",
        qr_shrinkage_factor=QR_SHRINKAGE,
        absolute_cutoff=1e-6,
        relative_cutoff=1e6,
        verbose=1,
    )
    t0 = time.time()
    qls.fit(x_train, y_onehot, batch_size=BATCH_SIZE)
    train_time_linear = time.time() - t0

    y_pred = qls.predict(x_test)
    acc_linear = accuracy_top1(y_pred, y_test)
    entry["linear"] = {
        "status": "ok",
        "train_time": train_time_linear,
        "test_accuracy": acc_linear,
    }
    print(f"  Linear: acc={acc_linear:.4f}, time={train_time_linear:.1f}s")

    # --- Build softmax PathSelector (gradient) ---
    base_softmax = keras.models.load_model(
        f"{cl_results_dir}/vgg16_path_selector_softmax_model.tf"
    )
    t0 = time.time()
    ps_softmax = KerasPathSelector(
        base_softmax, initial_layer=initial_layer, device=device, verbose=1
    )
    entry["build_time_gradient"] = time.time() - t0

    # --- QsimovGradient ---
    qg = KerasQsimovGradient(ps_softmax, verbose=1)
    qg.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        device=device,
    )
    t0 = time.time()
    qg.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True)
    train_time_gradient = time.time() - t0

    y_pred_g = qg.predict(x_test)
    acc_gradient = accuracy_top1(y_pred_g, y_test)
    entry["gradient"] = {
        "status": "ok",
        "train_time": train_time_gradient,
        "test_accuracy": acc_gradient,
    }
    print(f"  Gradient: acc={acc_gradient:.4f}, time={train_time_gradient:.1f}s")

    return entry


def execute_logic(results_dir, cl_results_dir, args):
    device = "/gpu:0" if args.processor == "gpu" else "/cpu:0"
    x_train, y_train, x_test, y_test = load_dataset()

    all_results = []
    for il in INITIAL_LAYERS:
        entry = run_one_layer(
            il, x_train, y_train, x_test, y_test, cl_results_dir, device
        )
        all_results.append(entry)

    output_file = os.path.join(results_dir, "sweep_results.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved sweep results to {output_file}")


def main(args):
    results_dir = get_initial_layer_sweep_results_dir(args.processor)
    cl_results_dir = get_imagenet_continual_learning_results_dir(args.processor)
    os.makedirs(results_dir, exist_ok=True)

    _check_prerequisites(cl_results_dir)

    def main_subprocess():
        make_imports()
        init_tensorflow(tf, args.processor)
        execute_logic(results_dir, cl_results_dir, args)

    p = mp.Process(target=main_subprocess)
    p.start()
    p.join()


###############################################################################
# CLI Parser
###############################################################################


class SweepParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="gpu")


if __name__ == "__main__":
    main(SweepParser().parse_args())
