"""Multi-round class-incremental Qsimov continual learning on ImageNet.

Data split (class-incremental)
--------------------------------
Round k trains and evaluates ONLY on the classes in group k:
  Round 1 → classes  0-24   Round 3 → classes 50-74
  Round 2 → classes 25-49   Round 4 → classes 75-99

This creates a genuine distribution shift between rounds so catastrophic
forgetting is detectable: accuracy on round-1 val must drop for methods that
forget and stay high for the no-forgetting method.

Qsimov variants
---------------
qsimov_linear_accum
    QsimovLinearSystem (linear last layer, MSE).
    fit() accumulates equations across rounds WITHOUT reset_equations().
    The no-forgetting mechanism: Ax=b grows with all seen data, so path
    weights satisfy constraints from every previous round.

qsimov_linear_reset
    Same solver, but reset_equations() is called before each round.
    Forgetting baseline: only the current round's equations are solved.

qsimov_gradient
    QsimovGradient (softmax last layer, sparse CE).
    Fine-tunes from the previous round's weights without data replay.
    Gradient drift causes partial forgetting.

Path selector
-------------
The path selector is always loaded from the model pre-trained on ALL classes.
Its weights are FROZEN — only the path weights (top Dense layer) change.
This is the correct Qsimov design: structural knowledge (paths) is built once;
quantitative knowledge (path weights) is updated per round.

Evaluation
----------
After round K each variant is evaluated on:
  - The full test set (overall accuracy across all 100 classes).
  - The held-out val sets of EVERY past round 1..K (forgetting measurement).

Note on the loss column for linear variants
-------------------------------------------
The linear model outputs raw real values (not calibrated probabilities), so
sparse-CE computed on those values has no semantic meaning.  Only the accuracy
metric (argmax comparison) is reported for linear variants; the 'loss' key in
the result dict is omitted for them.
"""

import os
import pickle
import time
import argparse
import multiprocessing as mp
import numpy as np
from experiments.path_utils import get_imagenet_continual_learning_results_dir
from experiments.imagenet_continual_learning.preprocess_data import N_ROUNDS


def make_imports():
    global tf, keras, init_tensorflow
    global load_dataset, make_round_splits, NUM_LABELS
    global KerasPathSelector, KerasQsimovLinearSystem, KerasQsimovGradient
    global load_model

    import tensorflow as tf
    from tensorflow import keras
    from experiments.mnist_speed_loss.tf_keras.utils import init_tensorflow
    from experiments.imagenet_continual_learning.preprocess_data import (
        load_dataset,
        make_round_splits,
        NUM_LABELS,
    )
    from qsimov.keras_path_selector import KerasPathSelector
    from qsimov.keras_qsimov_linear_system import KerasQsimovLinearSystem
    from qsimov.keras_qsimov_gradient import KerasQsimovGradient
    from experiments.imagenet_continual_learning.keras_model_factory import load_model


BATCH_SIZE = 64
QR_SHRINKAGE = 10


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def accuracy(y_pred, y_true_int):
    """Top-1 accuracy from raw logits or probabilities."""
    return float(np.mean(np.argmax(y_pred, axis=1) == y_true_int))


def evaluate_accuracy(model, x, y_int, batch_size):
    """Return {'accuracy': float} — loss omitted for linear variants."""
    y_pred = model.predict(x, batch_size=batch_size)
    return {"accuracy": accuracy(y_pred, y_int)}


def evaluate_sparse_ce(model, x, y_int, batch_size, n_classes):
    """Return {'accuracy': float, 'loss': float} for softmax models."""
    y_pred = model.predict(x, batch_size=batch_size)
    acc = accuracy(y_pred, y_int)
    eps = 1e-7
    y_clip = np.clip(y_pred, eps, 1 - eps)
    one_hot = np.eye(n_classes)[y_int.astype(int)]
    loss = float(-np.mean(np.sum(one_hot * np.log(y_clip), axis=1)))
    return {"accuracy": acc, "loss": loss}


def collect_per_round_metrics(model, rounds, current_k, batch_size, eval_fn):
    """Evaluate on val sets of all past rounds."""
    per_round = {}
    for prev_k, (_, _, val_x, val_y) in enumerate(rounds[:current_k], 1):
        per_round[f"round_{prev_k}"] = eval_fn(model, val_x, val_y, batch_size)
    return per_round


# ---------------------------------------------------------------------------
# Variant A: linear system, accumulative (no-forgetting)
# ---------------------------------------------------------------------------

def run_qsimov_linear_accum(path_selector, rounds, test_x, test_y):
    qls = KerasQsimovLinearSystem(
        path_selector,
        solver="back_substitution",
        qr_shrinkage_factor=QR_SHRINKAGE,
        absolute_cutoff=1e-6,
        relative_cutoff=1e6,
        verbose=1,
    )

    eval_fn = lambda m, x, y, bs: evaluate_accuracy(m, x, y, bs)
    results = {}
    cumulative_time = 0.0

    for k, (train_x, train_y, _, _) in enumerate(rounds, 1):
        # Linear system requires float one-hot targets
        y_onehot = np.eye(NUM_LABELS, dtype=np.float32)[train_y.astype(int)]

        t0 = time.time()
        # Key: NO reset_equations() — equations from previous rounds are kept
        qls.fit(train_x, y_onehot, batch_size=BATCH_SIZE)
        cumulative_time += time.time() - t0

        round_key = f"after_round_{k}"
        results[round_key] = {
            "time(s)": cumulative_time,
            "overall": eval_fn(qls, test_x, test_y, BATCH_SIZE),
            "per_round_val": collect_per_round_metrics(qls, rounds, k, BATCH_SIZE, eval_fn),
        }

    return results


# ---------------------------------------------------------------------------
# Variant B: linear system, reset per round (forgetting baseline)
# ---------------------------------------------------------------------------

def run_qsimov_linear_reset(path_selector, rounds, test_x, test_y):
    qls = KerasQsimovLinearSystem(
        path_selector,
        solver="back_substitution",
        qr_shrinkage_factor=QR_SHRINKAGE,
        absolute_cutoff=1e-6,
        relative_cutoff=1e6,
        verbose=1,
    )

    eval_fn = lambda m, x, y, bs: evaluate_accuracy(m, x, y, bs)
    results = {}
    cumulative_time = 0.0

    for k, (train_x, train_y, _, _) in enumerate(rounds, 1):
        y_onehot = np.eye(NUM_LABELS, dtype=np.float32)[train_y.astype(int)]

        t0 = time.time()
        # RESET before each round — previous knowledge discarded
        qls.reset_equations()
        qls.fit(train_x, y_onehot, batch_size=BATCH_SIZE)
        cumulative_time += time.time() - t0

        round_key = f"after_round_{k}"
        results[round_key] = {
            "time(s)": cumulative_time,
            "overall": eval_fn(qls, test_x, test_y, BATCH_SIZE),
            "per_round_val": collect_per_round_metrics(qls, rounds, k, BATCH_SIZE, eval_fn),
        }

    return results


# ---------------------------------------------------------------------------
# Variant C: gradient descent (fine-tune per round, no data replay)
# ---------------------------------------------------------------------------

def run_qsimov_gradient(path_selector, rounds, test_x, test_y, device):
    qg = KerasQsimovGradient(path_selector, verbose=1)
    qg.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        device=device,
    )

    eval_fn = lambda m, x, y, bs: evaluate_sparse_ce(m, x, y, bs, NUM_LABELS)
    results = {}
    cumulative_time = 0.0

    for k, (train_x, train_y, _, _) in enumerate(rounds, 1):
        t0 = time.time()
        # Fine-tune from current weights — no data replay
        qg.fit(
            train_x,
            train_y,
            batch_size=BATCH_SIZE,
            epochs=5,
            shuffle=True,
        )
        cumulative_time += time.time() - t0

        round_key = f"after_round_{k}"
        results[round_key] = {
            "time(s)": cumulative_time,
            "overall": eval_fn(qg, test_x, test_y, BATCH_SIZE),
            "per_round_val": collect_per_round_metrics(qg, rounds, k, BATCH_SIZE, eval_fn),
        }

    return results


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def execute_logic(method, results_dir, args):
    device = "/gpu:0" if args.processor == "gpu" else "/cpu:0"

    x_train, y_train, x_test, y_test = load_dataset()
    rounds = make_round_splits(x_train, y_train, n_rounds=N_ROUNDS)

    if method in ("qsimov_linear_accum", "qsimov_linear_reset"):
        base_model = load_model(results_dir, "path_selector_linear")
        path_selector = KerasPathSelector(
            base_model, initial_layer=args.initial_layer, device=device, verbose=1
        )
        if method == "qsimov_linear_accum":
            results = run_qsimov_linear_accum(path_selector, rounds, x_test, y_test)
        else:
            results = run_qsimov_linear_reset(path_selector, rounds, x_test, y_test)
    else:
        base_model = load_model(results_dir, "path_selector_softmax")
        path_selector = KerasPathSelector(
            base_model, initial_layer=args.initial_layer, device=device, verbose=1
        )
        results = run_qsimov_gradient(path_selector, rounds, x_test, y_test, device)

    n_paths = int(np.sum(path_selector.output_masks_, axis=1).max())
    with open(f"{results_dir}/number_of_paths_vgg16.txt", "w") as f:
        f.write(str(n_paths))

    output_file = f"{results_dir}/{method}_results.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results for {method} to {output_file}")


def main(args):
    results_dir = get_imagenet_continual_learning_results_dir(args.processor)
    os.makedirs(results_dir, exist_ok=True)

    methods = ["qsimov_linear_accum", "qsimov_linear_reset", "qsimov_gradient"]

    def main_subprocess(method):
        make_imports()
        init_tensorflow(tf, args.processor)
        execute_logic(method, results_dir, args)

    for method in methods:
        print(f"\n\nRunning continual learning: {method}\n")
        p = mp.Process(target=main_subprocess, args=(method,))
        p.start()
        p.join()


###############################################################################
# CLI Parser
###############################################################################


class TrainQsimovParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="gpu")
        self.add_argument(
            "--initial-layer",
            type=int,
            default=-1,
            help="PathSelector initial layer. -1 = last Dense only (recommended).",
        )


if __name__ == "__main__":
    main(TrainQsimovParser().parse_args())
