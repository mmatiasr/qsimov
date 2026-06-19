"""Streaming incremental update experiment on ImageNet subset — PyTorch backend.

Four methods across N_BATCHES=20 class-incremental streaming batches.

Prerequisites: pre-trained models from imagenet_continual_learning/train_pytorch.py:
    {cl_results_dir}/vgg16_path_selector_linear_model.pt
    {cl_results_dir}/vgg16_path_selector_softmax_model.pt
    {cl_results_dir}/vgg16_standard_model.pt
"""

import os
import pickle
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from experiments.path_utils import (
    get_imagenet_streaming_incremental_results_dir,
    get_imagenet_continual_learning_results_dir,
)
from experiments.imagenet_streaming_incremental.preprocess_data import (
    N_BATCHES,
    load_dataset,
    make_streaming_batches,
)
from experiments.imagenet_subset_by_splits.preprocess_data import (
    NUM_LABELS,
    INPUT_SHAPE_NCHW,
)
from qsimov.pytorch_path_selector import PytorchPathSelector
from qsimov.pytorch_qsimov_linear_system import PytorchQsimovLinearSystem
from qsimov.pytorch_qsimov_gradient import PytorchQsimovGradient

BATCH_SIZE = 64
EPOCHS_PER_BATCH = 3
QR_SHRINKAGE = 10
INITIAL_LAYER = -2


def _to_nchw_float32(x_uint8_nhwc):
    return np.transpose(x_uint8_nhwc.astype(np.float32), (0, 3, 1, 2))


def _convert_batch(batch):
    """Return a copy of the batch dict with all image arrays converted to float32 NCHW."""
    out = dict(batch)
    out["train_x"] = _to_nchw_float32(batch["train_x"])
    out["val_x"]   = _to_nchw_float32(batch["val_x"])
    out["cum_test_x"] = _to_nchw_float32(batch["cum_test_x"])
    return out


def accuracy_top1(y_pred, y_true_int):
    return float(np.mean(np.argmax(y_pred, axis=1) == y_true_int))


def _check_prerequisites(cl_results_dir):
    required = [
        "vgg16_path_selector_linear_model.pt",
        "vgg16_path_selector_softmax_model.pt",
        "vgg16_standard_model.pt",
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


# ---------------------------------------------------------------------------
# Method A: QsimovLinearSystem (accumulate, no forgetting)
# ---------------------------------------------------------------------------

def run_qsimov_linear_accum(batches, cl_results_dir, results_dir, device):
    base_model = torch.load(
        f"{cl_results_dir}/vgg16_path_selector_linear_model.pt",
        map_location=device,
    , weights_only=False)
    path_selector = PytorchPathSelector(
        neural_network=base_model,
        input_shape=INPUT_SHAPE_NCHW,
        initial_layer=INITIAL_LAYER,
        verbose=1,
        device=device,
    )
    n_paths = int(path_selector.output_masks_.sum())

    qls = PytorchQsimovLinearSystem(
        path_selector,
        solver="back_substitution",
        qr_shrinkage_factor=QR_SHRINKAGE,
        absolute_cutoff=1e-6,
        relative_cutoff=1e6,
        verbose=1,
    )

    results = {"n_paths": n_paths, "batches": []}

    for b, batch in enumerate(batches):
        bc = _convert_batch(batch)
        ty_onehot = np.eye(NUM_LABELS, dtype=np.float32)[bc["train_y"].astype(int)]

        t0 = time.time()
        qls.fit(bc["train_x"], ty_onehot, batch_size=BATCH_SIZE)
        update_time = time.time() - t0

        y_pred = qls.predict(bc["cum_test_x"], batch_size=BATCH_SIZE)
        acc = accuracy_top1(y_pred, bc["cum_test_y"])

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
    base_model = torch.load(
        f"{cl_results_dir}/vgg16_path_selector_softmax_model.pt",
        map_location=device,
    , weights_only=False)
    path_selector = PytorchPathSelector(
        neural_network=base_model,
        input_shape=INPUT_SHAPE_NCHW,
        initial_layer=INITIAL_LAYER,
        verbose=1,
        device=device,
    )
    n_paths = int(path_selector.output_masks_.sum())

    qg = PytorchQsimovGradient(path_selector, verbose=1)
    results = {"n_paths": n_paths, "batches": []}

    for b, batch in enumerate(batches):
        bc = _convert_batch(batch)

        t0 = time.time()
        y_onehot = np.eye(NUM_LABELS, dtype=np.float32)[bc["train_y"].astype(int)]
        qg.fit(
            bc["train_x"],
            y_onehot,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS_PER_BATCH,
            loss_function=nn.CrossEntropyLoss(),
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-5),
            device=device,
        )
        update_time = time.time() - t0

        y_pred = qg.predict(bc["cum_test_x"], batch_size=BATCH_SIZE)
        acc = accuracy_top1(y_pred, bc["cum_test_y"])

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

def _train_standard_one_round(model, x_nchw, y_int, device, epochs):
    loss_fn = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=1e-5)
    dl = DataLoader(
        TensorDataset(torch.from_numpy(x_nchw), torch.from_numpy(y_int.astype(np.int64))),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    model.to(device)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()


def _predict_standard(model, x_nchw, device):
    model.eval()
    preds = []
    dl = DataLoader(TensorDataset(torch.from_numpy(x_nchw)), batch_size=BATCH_SIZE)
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def run_standard_finetune(batches, cl_results_dir, results_dir, device):
    model = torch.load(f"{cl_results_dir}/vgg16_standard_model.pt", map_location=device, weights_only=False)
    results = {"batches": []}

    for b, batch in enumerate(batches):
        bc = _convert_batch(batch)

        t0 = time.time()
        _train_standard_one_round(model, bc["train_x"], bc["train_y"], device, EPOCHS_PER_BATCH)
        update_time = time.time() - t0

        y_pred = _predict_standard(model, bc["cum_test_x"], device)
        acc = accuracy_top1(y_pred, bc["cum_test_y"])

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

def run_standard_cumulative(batches, cl_results_dir, results_dir, device):
    cum_x, cum_y = [], []
    results = {"batches": []}

    for b, batch in enumerate(batches):
        bc = _convert_batch(batch)
        cum_x.append(bc["train_x"])
        cum_y.append(bc["train_y"])
        cx = np.concatenate(cum_x)
        cy = np.concatenate(cum_y)

        model = torch.load(f"{cl_results_dir}/vgg16_standard_model.pt", map_location=device, weights_only=False)

        t0 = time.time()
        _train_standard_one_round(model, cx, cy, device, EPOCHS_PER_BATCH)
        update_time = time.time() - t0

        y_pred = _predict_standard(model, bc["cum_test_x"], device)
        acc = accuracy_top1(y_pred, bc["cum_test_y"])

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


def main(args):
    device = torch.device("cuda" if args.processor == "gpu" else "cpu")
    results_dir = get_imagenet_streaming_incremental_results_dir(
        args.processor, framework="pytorch"
    )
    cl_results_dir = get_imagenet_continual_learning_results_dir(
        args.processor, framework="pytorch"
    )
    os.makedirs(results_dir, exist_ok=True)
    _check_prerequisites(cl_results_dir)

    x_train, y_train, x_test, y_test = load_dataset()
    batches = make_streaming_batches(x_train, y_train, x_test, y_test)

    for method in ("qsimov_linear_accum", "qsimov_gradient",
                   "standard_finetune", "standard_cumulative"):
        print(f"\n\nRunning streaming incremental: {method}\n")
        if method == "qsimov_linear_accum":
            run_qsimov_linear_accum(batches, cl_results_dir, results_dir, device)
        elif method == "qsimov_gradient":
            run_qsimov_gradient(batches, cl_results_dir, results_dir, device)
        elif method == "standard_finetune":
            run_standard_finetune(batches, cl_results_dir, results_dir, device)
        elif method == "standard_cumulative":
            run_standard_cumulative(batches, cl_results_dir, results_dir, device)


class TrainStreamingParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="gpu")


if __name__ == "__main__":
    main(TrainStreamingParser().parse_args())
