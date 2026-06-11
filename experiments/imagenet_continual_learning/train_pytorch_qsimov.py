"""Multi-round class-incremental Qsimov continual learning — PyTorch backend.

Three variants:
  qsimov_linear_accum  — LinearSystem, no reset (no forgetting by design)
  qsimov_linear_reset  — LinearSystem, reset before each round (forgetting baseline)
  qsimov_gradient      — Gradient descent, fine-tune per round (partial forgetting)
"""

import os
import pickle
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from experiments.path_utils import get_imagenet_continual_learning_results_dir
from experiments.imagenet_continual_learning.preprocess_data import (
    load_dataset,
    make_round_splits,
    NUM_LABELS,
    N_ROUNDS,
)
from experiments.imagenet_subset_by_splits.preprocess_data import INPUT_SHAPE_NCHW
from qsimov.pytorch_path_selector import PytorchPathSelector
from qsimov.pytorch_qsimov_linear_system import PytorchQsimovLinearSystem
from qsimov.pytorch_qsimov_gradient import PytorchQsimovGradient

BATCH_SIZE = 64
QR_SHRINKAGE = 10
EPOCHS_PER_ROUND = 5


def _to_nchw_float32(x_uint8_nhwc):
    return np.transpose(x_uint8_nhwc.astype(np.float32), (0, 3, 1, 2))


def accuracy(y_pred, y_true_int):
    return float(np.mean(np.argmax(y_pred, axis=1) == y_true_int))


def evaluate_accuracy(model, x, y_int):
    y_pred = model.predict(x, batch_size=BATCH_SIZE)
    return {"accuracy": accuracy(y_pred, y_int)}


def evaluate_ce(model, x, y_int):
    y_pred = model.predict(x, batch_size=BATCH_SIZE)
    acc = accuracy(y_pred, y_int)
    eps = 1e-7
    y_clip = np.clip(y_pred, eps, 1 - eps)
    one_hot = np.eye(NUM_LABELS)[y_int.astype(int)]
    loss = float(-np.mean(np.sum(one_hot * np.log(y_clip), axis=1)))
    return {"accuracy": acc, "loss": loss}


def collect_per_round_metrics(model, rounds_nchw, current_k, eval_fn):
    per_round = {}
    for prev_k, (_, _, val_x, val_y) in enumerate(rounds_nchw[:current_k], 1):
        per_round[f"round_{prev_k}"] = eval_fn(model, val_x, val_y)
    return per_round


def _convert_rounds(rounds):
    """Convert all round arrays from uint8 NHWC to float32 NCHW."""
    out = []
    for (tx, ty, vx, vy) in rounds:
        out.append((_to_nchw_float32(tx), ty, _to_nchw_float32(vx), vy))
    return out


def run_qsimov_linear_accum(path_selector, rounds_nchw, test_x, test_y):
    qls = PytorchQsimovLinearSystem(
        path_selector,
        solver="back_substitution",
        qr_shrinkage_factor=QR_SHRINKAGE,
        absolute_cutoff=1e-6,
        relative_cutoff=1e6,
        verbose=1,
    )
    eval_fn = evaluate_accuracy
    results = {}
    cumulative_time = 0.0

    for k, (train_x, train_y, _, _) in enumerate(rounds_nchw, 1):
        y_onehot = np.eye(NUM_LABELS, dtype=np.float32)[train_y.astype(int)]
        t0 = time.time()
        qls.fit(train_x, y_onehot, batch_size=BATCH_SIZE)
        cumulative_time += time.time() - t0

        round_key = f"after_round_{k}"
        results[round_key] = {
            "time(s)": cumulative_time,
            "overall": eval_fn(qls, test_x, test_y),
            "per_round_val": collect_per_round_metrics(qls, rounds_nchw, k, eval_fn),
        }
        print(f"[linear_accum] round {k}/{N_ROUNDS}  "
              f"overall_acc={results[round_key]['overall']['accuracy']:.4f}  "
              f"time={cumulative_time:.1f}s")

    return results


def run_qsimov_linear_reset(path_selector, rounds_nchw, test_x, test_y):
    qls = PytorchQsimovLinearSystem(
        path_selector,
        solver="back_substitution",
        qr_shrinkage_factor=QR_SHRINKAGE,
        absolute_cutoff=1e-6,
        relative_cutoff=1e6,
        verbose=1,
    )
    eval_fn = evaluate_accuracy
    results = {}
    cumulative_time = 0.0

    for k, (train_x, train_y, _, _) in enumerate(rounds_nchw, 1):
        y_onehot = np.eye(NUM_LABELS, dtype=np.float32)[train_y.astype(int)]
        t0 = time.time()
        qls.reset_equations()
        qls.fit(train_x, y_onehot, batch_size=BATCH_SIZE)
        cumulative_time += time.time() - t0

        round_key = f"after_round_{k}"
        results[round_key] = {
            "time(s)": cumulative_time,
            "overall": eval_fn(qls, test_x, test_y),
            "per_round_val": collect_per_round_metrics(qls, rounds_nchw, k, eval_fn),
        }
        print(f"[linear_reset] round {k}/{N_ROUNDS}  "
              f"overall_acc={results[round_key]['overall']['accuracy']:.4f}  "
              f"time={cumulative_time:.1f}s")

    return results


def run_qsimov_gradient(path_selector, rounds_nchw, test_x, test_y, device):
    qg = PytorchQsimovGradient(path_selector, verbose=1)
    eval_fn = lambda m, x, y: evaluate_ce(m, x, y)
    results = {}
    cumulative_time = 0.0

    for k, (train_x, train_y, _, _) in enumerate(rounds_nchw, 1):
        t0 = time.time()
        qg.fit(
            train_x,
            train_y.astype(np.int64),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS_PER_ROUND,
            loss_function=nn.CrossEntropyLoss(),
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-5),
            device=device,
        )
        cumulative_time += time.time() - t0

        round_key = f"after_round_{k}"
        results[round_key] = {
            "time(s)": cumulative_time,
            "overall": eval_fn(qg, test_x, test_y),
            "per_round_val": collect_per_round_metrics(qg, rounds_nchw, k, eval_fn),
        }
        print(f"[gradient] round {k}/{N_ROUNDS}  "
              f"overall_acc={results[round_key]['overall']['accuracy']:.4f}  "
              f"time={cumulative_time:.1f}s")

    return results


def execute_logic(method, results_dir, args, device):
    x_train, y_train, x_test, y_test = load_dataset()
    rounds = make_round_splits(x_train, y_train, n_rounds=N_ROUNDS)
    rounds_nchw = _convert_rounds(rounds)
    test_x = _to_nchw_float32(x_test)

    if method in ("qsimov_linear_accum", "qsimov_linear_reset"):
        base_model = torch.load(
            f"{results_dir}/vgg16_path_selector_linear_model.pt",
            map_location=device,
        )
        path_selector = PytorchPathSelector(
            neural_network=base_model,
            input_shape=INPUT_SHAPE_NCHW,
            initial_layer=args.initial_layer,
            verbose=1,
            device=device,
        )
        if method == "qsimov_linear_accum":
            results = run_qsimov_linear_accum(path_selector, rounds_nchw, test_x, y_test)
        else:
            results = run_qsimov_linear_reset(path_selector, rounds_nchw, test_x, y_test)
    else:
        base_model = torch.load(
            f"{results_dir}/vgg16_path_selector_softmax_model.pt",
            map_location=device,
        )
        path_selector = PytorchPathSelector(
            neural_network=base_model,
            input_shape=INPUT_SHAPE_NCHW,
            initial_layer=args.initial_layer,
            verbose=1,
            device=device,
        )
        results = run_qsimov_gradient(path_selector, rounds_nchw, test_x, y_test, device)

    n_paths = int(np.sum(path_selector.output_masks_, axis=1).max())
    with open(f"{results_dir}/number_of_paths_vgg16.txt", "w") as f:
        f.write(str(n_paths))

    output_file = f"{results_dir}/{method}_results.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results for {method} to {output_file}")


def main(args):
    device = torch.device("cuda" if args.processor == "gpu" else "cpu")
    results_dir = get_imagenet_continual_learning_results_dir(args.processor, framework="pytorch")
    os.makedirs(results_dir, exist_ok=True)

    for method in ("qsimov_linear_accum", "qsimov_linear_reset", "qsimov_gradient"):
        print(f"\n\nRunning continual learning: {method}\n")
        execute_logic(method, results_dir, args, device)


class TrainQsimovParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="gpu")
        self.add_argument("--initial-layer", type=int, default=-1)


if __name__ == "__main__":
    main(TrainQsimovParser().parse_args())
