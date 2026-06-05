"""Streaming incremental update experiment on ImageNet subset.

Simulates a class-incremental data stream of N_BATCHES batches,
each introducing CLASSES_PER_BATCH new classes (5 out of 100).

After each batch update, all methods are evaluated on the test samples of
ALL classes seen so far (cumulative test set).  The key metrics are:

  accuracy_on_seen : top-1 accuracy on cumulative test set
  update_time      : wall-clock seconds to process THIS batch only

This allows separating the "fast re-training" claim (constant update_time for
Qsimov vs. growing cost for cumulative baseline) from the "no forgetting" claim
(accuracy_on_seen stays high for qsimov_linear_accum vs. drops for finetune).

Methods
-------
qsimov_linear_accum
    QsimovLinearSystem.fit() without reset_equations().
    Processes only the NEW batch each time — constant update cost.
    Equations from previous batches are retained, so the linear system
    implicitly satisfies all past constraints (no forgetting by design).

qsimov_gradient
    KerasQsimovGradient.fit() on the new batch only (EPOCHS_PER_BATCH epochs).
    Constant cost per batch; gradient drift causes partial forgetting.

standard_finetune
    Adam fine-tunes the top layers on the new batch only (no replay).
    Constant cost but suffers catastrophic forgetting on old classes.

standard_cumulative
    Re-trains from a fresh base model on ALL cumulative data per batch.
    Oracle upper bound in accuracy; update cost grows linearly with batches.

Prerequisites
-------------
Requires pre-trained models saved by:
    experiments/imagenet_continual_learning/train_keras.py --epochs N --processor {processor}

Expected files in {continual_learning_results_dir}/:
    vgg16_path_selector_linear_model.tf   (for qsimov_linear_accum)
    vgg16_path_selector_softmax_model.tf  (for qsimov_gradient)
    vgg16_standard_model.tf               (for standard_finetune, standard_cumulative)
"""

import os
import pickle
import time
import argparse
import multiprocessing as mp
import numpy as np
from experiments.path_utils import (
    get_imagenet_streaming_incremental_results_dir,
    get_imagenet_continual_learning_results_dir,
)
from experiments.imagenet_streaming_incremental.preprocess_data import (
    N_BATCHES,
    load_dataset,
    make_streaming_batches,
)


def make_imports():
    global tf, keras, init_tensorflow
    global KerasPathSelector, KerasQsimovLinearSystem, KerasQsimovGradient
    global NUM_LABELS

    import tensorflow as tf
    from tensorflow import keras
    from experiments.mnist_speed_loss.tf_keras.utils import init_tensorflow
    from qsimov.keras_path_selector import KerasPathSelector
    from qsimov.keras_qsimov_linear_system import KerasQsimovLinearSystem
    from qsimov.keras_qsimov_gradient import KerasQsimovGradient
    from experiments.imagenet_streaming_incremental.preprocess_data import (
        NUM_LABELS,  # re-imported here to avoid module-level TF import
    )


BATCH_SIZE = 64
EPOCHS_PER_BATCH = 3
QR_SHRINKAGE = 10
INITIAL_LAYER = -1


def accuracy_top1(y_pred, y_true_int):
    return float(np.mean(np.argmax(y_pred, axis=1) == y_true_int))


def _check_prerequisites(cl_results_dir):
    required = [
        "vgg16_path_selector_linear_model.tf",
        "vgg16_path_selector_softmax_model.tf",
        "vgg16_standard_model.tf",
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


# ---------------------------------------------------------------------------
# Method A: QsimovLinearSystem (accumulate equations, no forgetting)
# ---------------------------------------------------------------------------

def run_qsimov_linear_accum(batches, cl_results_dir, results_dir, device):
    from experiments.imagenet_subset_by_splits.preprocess_data import NUM_LABELS as NL

    base_model = keras.models.load_model(
        f"{cl_results_dir}/vgg16_path_selector_linear_model.tf"
    )
    path_selector = KerasPathSelector(
        base_model, initial_layer=INITIAL_LAYER, device=device, verbose=1
    )
    n_paths = int(path_selector.output_masks_.sum())

    qls = KerasQsimovLinearSystem(
        path_selector,
        solver="back_substitution",
        qr_shrinkage_factor=QR_SHRINKAGE,
        absolute_cutoff=1e-6,
        relative_cutoff=1e6,
        verbose=1,
    )

    results = {"n_paths": n_paths, "batches": []}

    for b, batch in enumerate(batches):
        tx, ty = batch["train_x"], batch["train_y"]
        ty_onehot = np.eye(NL, dtype=np.float32)[ty.astype(int)]

        t0 = time.time()
        qls.fit(tx, ty_onehot, batch_size=BATCH_SIZE)
        update_time = time.time() - t0

        y_pred = qls.predict(batch["cum_test_x"])
        acc = accuracy_top1(y_pred, batch["cum_test_y"])

        results["batches"].append({
            "batch": b + 1,
            "update_time": update_time,
            "accuracy_on_seen": acc,
            "n_seen_classes": len(batch["classes"]) * (b + 1),
        })
        print(f"[qsimov_linear_accum] batch {b+1}/{N_BATCHES} "
              f"acc={acc:.4f} time={update_time:.1f}s")

    output_file = f"{results_dir}/qsimov_linear_accum_results.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved qsimov_linear_accum results to {output_file}")


# ---------------------------------------------------------------------------
# Method B: QsimovGradient (no replay, gradient drift)
# ---------------------------------------------------------------------------

def run_qsimov_gradient(batches, cl_results_dir, results_dir, device):
    base_model = keras.models.load_model(
        f"{cl_results_dir}/vgg16_path_selector_softmax_model.tf"
    )
    path_selector = KerasPathSelector(
        base_model, initial_layer=INITIAL_LAYER, device=device, verbose=1
    )
    n_paths = int(path_selector.output_masks_.sum())

    qg = KerasQsimovGradient(path_selector, verbose=1)
    qg.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        device=device,
    )

    results = {"n_paths": n_paths, "batches": []}

    for b, batch in enumerate(batches):
        tx, ty = batch["train_x"], batch["train_y"]

        t0 = time.time()
        qg.fit(tx, ty, batch_size=BATCH_SIZE, epochs=EPOCHS_PER_BATCH, shuffle=True)
        update_time = time.time() - t0

        y_pred = qg.predict(batch["cum_test_x"])
        acc = accuracy_top1(y_pred, batch["cum_test_y"])

        results["batches"].append({
            "batch": b + 1,
            "update_time": update_time,
            "accuracy_on_seen": acc,
            "n_seen_classes": len(batch["classes"]) * (b + 1),
        })
        print(f"[qsimov_gradient] batch {b+1}/{N_BATCHES} "
              f"acc={acc:.4f} time={update_time:.1f}s")

    output_file = f"{results_dir}/qsimov_gradient_results.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved qsimov_gradient results to {output_file}")


# ---------------------------------------------------------------------------
# Method C: Standard fine-tuning (no replay, catastrophic forgetting)
# ---------------------------------------------------------------------------

def run_standard_finetune(batches, cl_results_dir, results_dir):
    model = keras.models.load_model(f"{cl_results_dir}/vgg16_standard_model.tf")
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        metrics=["accuracy"],
    )

    results = {"batches": []}

    for b, batch in enumerate(batches):
        tx, ty = batch["train_x"], batch["train_y"]

        t0 = time.time()
        model.fit(tx, ty, batch_size=BATCH_SIZE, epochs=EPOCHS_PER_BATCH,
                  shuffle=True, verbose=1)
        update_time = time.time() - t0

        y_pred = model.predict(batch["cum_test_x"], batch_size=BATCH_SIZE)
        acc = accuracy_top1(y_pred, batch["cum_test_y"])

        results["batches"].append({
            "batch": b + 1,
            "update_time": update_time,
            "accuracy_on_seen": acc,
            "n_seen_classes": len(batch["classes"]) * (b + 1),
        })
        print(f"[standard_finetune] batch {b+1}/{N_BATCHES} "
              f"acc={acc:.4f} time={update_time:.1f}s")

    output_file = f"{results_dir}/standard_finetune_results.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved standard_finetune results to {output_file}")


# ---------------------------------------------------------------------------
# Method D: Cumulative retrain (oracle, growing cost)
# ---------------------------------------------------------------------------

def run_standard_cumulative(batches, cl_results_dir, results_dir):
    cum_x, cum_y = [], []
    results = {"batches": []}

    for b, batch in enumerate(batches):
        cum_x.append(batch["train_x"])
        cum_y.append(batch["train_y"])
        cx = np.concatenate(cum_x)
        cy = np.concatenate(cum_y)

        # Reload fresh base model each time — fair comparison
        model = keras.models.load_model(f"{cl_results_dir}/vgg16_standard_model.tf")
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            metrics=["accuracy"],
        )

        t0 = time.time()
        model.fit(cx, cy, batch_size=BATCH_SIZE, epochs=EPOCHS_PER_BATCH,
                  shuffle=True, verbose=1)
        update_time = time.time() - t0

        y_pred = model.predict(batch["cum_test_x"], batch_size=BATCH_SIZE)
        acc = accuracy_top1(y_pred, batch["cum_test_y"])

        results["batches"].append({
            "batch": b + 1,
            "update_time": update_time,
            "accuracy_on_seen": acc,
            "n_seen_classes": len(batch["classes"]) * (b + 1),
            "n_train_samples": len(cx),
        })
        print(f"[standard_cumulative] batch {b+1}/{N_BATCHES} "
              f"acc={acc:.4f} time={update_time:.1f}s  "
              f"(trained on {len(cx)} samples)")

    output_file = f"{results_dir}/standard_cumulative_results.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved standard_cumulative results to {output_file}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def execute_logic(method, batches, cl_results_dir, results_dir, args):
    device = "/gpu:0" if args.processor == "gpu" else "/cpu:0"

    if method == "qsimov_linear_accum":
        run_qsimov_linear_accum(batches, cl_results_dir, results_dir, device)
    elif method == "qsimov_gradient":
        run_qsimov_gradient(batches, cl_results_dir, results_dir, device)
    elif method == "standard_finetune":
        run_standard_finetune(batches, cl_results_dir, results_dir)
    elif method == "standard_cumulative":
        run_standard_cumulative(batches, cl_results_dir, results_dir)


def main(args):
    results_dir = get_imagenet_streaming_incremental_results_dir(args.processor)
    cl_results_dir = get_imagenet_continual_learning_results_dir(args.processor)
    os.makedirs(results_dir, exist_ok=True)

    _check_prerequisites(cl_results_dir)

    def main_subprocess(method):
        make_imports()
        init_tensorflow(tf, args.processor)
        x_train, y_train, x_test, y_test = load_dataset()
        batches = make_streaming_batches(x_train, y_train, x_test, y_test)
        execute_logic(method, batches, cl_results_dir, results_dir, args)

    methods = [
        "qsimov_linear_accum",
        "qsimov_gradient",
        "standard_finetune",
        "standard_cumulative",
    ]
    for method in methods:
        print(f"\n\nRunning streaming incremental: {method}\n")
        p = mp.Process(target=main_subprocess, args=(method,))
        p.start()
        p.join()


###############################################################################
# CLI Parser
###############################################################################


class TrainStreamingParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="gpu")


if __name__ == "__main__":
    main(TrainStreamingParser().parse_args())
