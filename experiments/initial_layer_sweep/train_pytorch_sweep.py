"""Sweep initial_layer values for Qsimov PathSelector on ImageNet — PyTorch backend.

Measures for each initial_layer in [-2, -3]:
  - Number of paths and PathSelector build time
  - QsimovLinearSystem accuracy and training time
  - QsimovGradient accuracy and training time

Prerequisites: pre-trained models from imagenet_continual_learning/train_pytorch.py:
    {cl_results_dir}/vgg16_path_selector_linear_model.pt
    {cl_results_dir}/vgg16_path_selector_softmax_model.pt
"""

import os
import pickle
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from experiments.path_utils import (
    get_initial_layer_sweep_results_dir,
    get_imagenet_continual_learning_results_dir,
)
from experiments.imagenet_subset_by_splits.preprocess_data import (
    NUM_LABELS,
    INPUT_SHAPE_NCHW,
)
from experiments.imagenet_continual_learning.preprocess_data import load_dataset
from qsimov.pytorch_path_selector import PytorchPathSelector
from qsimov.pytorch_qsimov_linear_system import PytorchQsimovLinearSystem
from qsimov.pytorch_qsimov_gradient import PytorchQsimovGradient

INITIAL_LAYERS = [-2, -3]
BATCH_SIZE = 64
EPOCHS = 5
QR_SHRINKAGE = 10
MAX_PATHS = 500_000


def _to_nchw_float32(x_uint8_nhwc):
    return np.transpose(x_uint8_nhwc.astype(np.float32), (0, 3, 1, 2))


def accuracy_top1(y_pred, y_true_int):
    return float(np.mean(np.argmax(y_pred, axis=1) == y_true_int))


def _check_prerequisites(cl_results_dir):
    required = [
        "vgg16_path_selector_linear_model.pt",
        "vgg16_path_selector_softmax_model.pt",
    ]
    missing = [
        f for f in required
        if not os.path.exists(os.path.join(cl_results_dir, f))
    ]
    if missing:
        raise FileNotFoundError(
            "Missing pre-trained model(s) in continual_learning results dir:\n"
            + "\n".join(f"  {cl_results_dir}/{f}" for f in missing)
            + "\nRun: python3 -m experiments.imagenet_continual_learning.train_pytorch "
            "--epochs 20 --processor gpu"
        )


def run_one_layer(initial_layer, x_train, y_train, x_test, y_test,
                  cl_results_dir, device):
    entry = {"initial_layer": initial_layer}

    base_linear = torch.load(
        f"{cl_results_dir}/vgg16_path_selector_linear_model.pt",
        map_location=device,
    , weights_only=False)
    t0 = time.time()
    try:
        ps_linear = PytorchPathSelector(
            neural_network=base_linear,
            input_shape=INPUT_SHAPE_NCHW,
            initial_layer=initial_layer,
            verbose=1,
            device=device,
        )
    except (ValueError, RuntimeError) as exc:
        reason = str(exc).split("\n")[0]
        entry["linear"] = {"status": "invalid_layer", "reason": reason}
        entry["gradient"] = {"status": "invalid_layer", "reason": reason}
        print(f"\ninitial_layer={initial_layer}: PathSelector failed — {reason}")
        return entry
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

    y_onehot = np.eye(NUM_LABELS, dtype=np.float32)[y_train.astype(int)]
    qls = PytorchQsimovLinearSystem(
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

    y_pred = qls.predict(x_test, batch_size=BATCH_SIZE)
    acc_linear = accuracy_top1(y_pred, y_test)
    entry["linear"] = {
        "status": "ok",
        "train_time": train_time_linear,
        "test_accuracy": acc_linear,
    }
    print(f"  Linear: acc={acc_linear:.4f}, time={train_time_linear:.1f}s")

    base_softmax = torch.load(
        f"{cl_results_dir}/vgg16_path_selector_softmax_model.pt",
        map_location=device,
    , weights_only=False)
    t0 = time.time()
    ps_softmax = PytorchPathSelector(
        neural_network=base_softmax,
        input_shape=INPUT_SHAPE_NCHW,
        initial_layer=initial_layer,
        verbose=1,
        device=device,
    )
    entry["build_time_gradient"] = time.time() - t0

    qg = PytorchQsimovGradient(ps_softmax, verbose=1)
    t0 = time.time()
    y_onehot = np.eye(NUM_LABELS, dtype=np.float32)[y_train.astype(int)]
    qg.fit(
        x_train,
        y_onehot,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-5),
        device=device,
    )
    train_time_gradient = time.time() - t0

    y_pred_g = qg.predict(x_test, batch_size=BATCH_SIZE)
    acc_gradient = accuracy_top1(y_pred_g, y_test)
    entry["gradient"] = {
        "status": "ok",
        "train_time": train_time_gradient,
        "test_accuracy": acc_gradient,
    }
    print(f"  Gradient: acc={acc_gradient:.4f}, time={train_time_gradient:.1f}s")

    return entry


def main(args):
    device = torch.device("cuda" if args.processor == "gpu" else "cpu")
    results_dir = get_initial_layer_sweep_results_dir(args.processor, framework="pytorch")
    cl_results_dir = get_imagenet_continual_learning_results_dir(
        args.processor, framework="pytorch"
    )
    os.makedirs(results_dir, exist_ok=True)
    _check_prerequisites(cl_results_dir)

    x_train_raw, y_train, x_test_raw, y_test = load_dataset()
    x_train = _to_nchw_float32(x_train_raw)
    x_test  = _to_nchw_float32(x_test_raw)

    all_results = []
    for il in INITIAL_LAYERS:
        entry = run_one_layer(il, x_train, y_train, x_test, y_test, cl_results_dir, device)
        all_results.append(entry)

    output_file = os.path.join(results_dir, "sweep_results.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved sweep results to {output_file}")


class SweepParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="gpu")


if __name__ == "__main__":
    main(SweepParser().parse_args())
