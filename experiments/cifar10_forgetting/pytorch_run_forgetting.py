"""Class-incremental forgetting experiment on CIFAR-10 — PyTorch backend.

Same 6 methods as run_forgetting.py but using PytorchPathSelector /
PytorchQsimovLinearSystem / PytorchQsimovGradient.
INITIAL_LAYER=-1 targets Dense(32,relu)+Dense(10,linear) — 33 paths.
"""

import os
import pickle
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from experiments.path_utils import get_cifar10_forgetting_results_dir
from experiments.cifar10_forgetting.preprocess_data import load_data, get_data_dir
from experiments.mnist_speed_loss.pytorch.utils import (
    samples_to_channels_first,
    create_dataloader,
    accuracy as torch_accuracy,
)
from qsimov.pytorch_path_selector import PytorchPathSelector
from qsimov.pytorch_qsimov_linear_system import PytorchQsimovLinearSystem
from qsimov.pytorch_qsimov_gradient import PytorchQsimovGradient

BATCH_SIZE = 256
QR_SHRINKAGE = 2
INITIAL_LAYER = -1


def np_accuracy(y_pred, y_true_onehot):
    return float(np.mean(
        np.argmax(y_pred, axis=1) == np.argmax(y_true_onehot, axis=1)
    ))


def eval_splits(y_pred_old, y_pred_new, y_pred_all, data, fit_time):
    return {
        "acc_old": np_accuracy(y_pred_old, data["test_phase1_y"]),
        "acc_new": np_accuracy(y_pred_new, data["test_phase2_y"]),
        "acc_all": np_accuracy(y_pred_all, data["test_y"]),
        "time":    fit_time,
    }


def save_results(method, results, results_dir):
    path = os.path.join(results_dir, f"{method}_results.pkl")
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(
        f"\n[{method}] acc_old={results['acc_old']:.4f}  "
        f"acc_new={results['acc_new']:.4f}  "
        f"acc_all={results['acc_all']:.4f}  "
        f"time={results['time']:.1f}s"
    )


def load_base(base_model_path, device):
    model = torch.load(base_model_path, map_location=device)
    model.eval()
    return model


def model_predict(model, x_nhwc, device, batch_size=BATCH_SIZE):
    x_nchw = samples_to_channels_first(x_nhwc)
    model.eval()
    model.to(device)
    chunks = []
    with torch.no_grad():
        for i in range(0, len(x_nchw), batch_size):
            batch = torch.from_numpy(x_nchw[i:i + batch_size]).to(device)
            chunks.append(model(batch).cpu().numpy())
    return np.concatenate(chunks)


def _torch_train(model, x_nchw, y, device, epochs, batch_size):
    model.train()
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dl = create_dataloader(x_nchw, y, batch_size)
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch + 1}/{epochs}  loss={total_loss / len(dl):.4f}")


# ---------------------------------------------------------------------------
# Method: base
# ---------------------------------------------------------------------------

def run_base(data, results_dir, base_model_path, device):
    model = load_base(base_model_path, device)
    results = eval_splits(
        model_predict(model, data["test_phase1_x"], device),
        model_predict(model, data["test_phase2_x"], device),
        model_predict(model, data["test_x"],        device),
        data, 0.0,
    )
    save_results("base", results, results_dir)


# ---------------------------------------------------------------------------
# Method: linear_accum
# ---------------------------------------------------------------------------

def run_qsimov_linear_accum(data, results_dir, base_model_path, device):
    model = load_base(base_model_path, device)
    input_shape = samples_to_channels_first(data["phase1_train_x"]).shape[1:]
    ps = PytorchPathSelector(
        neural_network=model,
        input_shape=input_shape,
        initial_layer=INITIAL_LAYER,
        verbose=1,
        device=device,
    )
    qls = PytorchQsimovLinearSystem(
        ps, solver="back_substitution", qr_shrinkage_factor=QR_SHRINKAGE, verbose=1
    )

    phase1_x = samples_to_channels_first(data["phase1_train_x"])
    phase2_x = samples_to_channels_first(data["phase2_train_x"])

    qls.fit(phase1_x, data["phase1_train_y"], batch_size=BATCH_SIZE)

    t0 = time.time()
    qls.fit(phase2_x, data["phase2_train_y"], batch_size=BATCH_SIZE)
    update_time = time.time() - t0

    results = eval_splits(
        qls.predict(samples_to_channels_first(data["test_phase1_x"]), batch_size=BATCH_SIZE),
        qls.predict(samples_to_channels_first(data["test_phase2_x"]), batch_size=BATCH_SIZE),
        qls.predict(samples_to_channels_first(data["test_x"]),        batch_size=BATCH_SIZE),
        data, update_time,
    )
    save_results("linear_accum", results, results_dir)


# ---------------------------------------------------------------------------
# Method: linear_new_only
# ---------------------------------------------------------------------------

def run_qsimov_linear_new_only(data, results_dir, base_model_path, device):
    model = load_base(base_model_path, device)
    input_shape = samples_to_channels_first(data["phase1_train_x"]).shape[1:]
    ps = PytorchPathSelector(
        neural_network=model,
        input_shape=input_shape,
        initial_layer=INITIAL_LAYER,
        verbose=1,
        device=device,
    )
    qls = PytorchQsimovLinearSystem(
        ps, solver="back_substitution", qr_shrinkage_factor=QR_SHRINKAGE, verbose=1
    )

    phase2_x = samples_to_channels_first(data["phase2_train_x"])
    t0 = time.time()
    qls.fit(phase2_x, data["phase2_train_y"], batch_size=BATCH_SIZE)
    fit_time = time.time() - t0

    results = eval_splits(
        qls.predict(samples_to_channels_first(data["test_phase1_x"]), batch_size=BATCH_SIZE),
        qls.predict(samples_to_channels_first(data["test_phase2_x"]), batch_size=BATCH_SIZE),
        qls.predict(samples_to_channels_first(data["test_x"]),        batch_size=BATCH_SIZE),
        data, fit_time,
    )
    save_results("linear_new_only", results, results_dir)


# ---------------------------------------------------------------------------
# Method: gradient_new_only
# ---------------------------------------------------------------------------

def run_qsimov_gradient(data, results_dir, base_model_path, device, epochs):
    model = load_base(base_model_path, device)
    input_shape = samples_to_channels_first(data["phase1_train_x"]).shape[1:]
    ps = PytorchPathSelector(
        neural_network=model,
        input_shape=input_shape,
        initial_layer=INITIAL_LAYER,
        verbose=1,
        device=device,
    )
    qg = PytorchQsimovGradient(ps)

    phase2_x = samples_to_channels_first(data["phase2_train_x"])
    t0 = time.time()
    qg.fit(
        phase2_x,
        data["phase2_train_y"],
        batch_size=BATCH_SIZE,
        epochs=epochs,
        loss_function=nn.MSELoss(),
        optimizer=lambda params: torch.optim.Adam(params, lr=0.001),
        metrics=[torch_accuracy],
        device=device,
    )
    fit_time = time.time() - t0

    results = eval_splits(
        qg.predict(samples_to_channels_first(data["test_phase1_x"]), batch_size=BATCH_SIZE, device=device),
        qg.predict(samples_to_channels_first(data["test_phase2_x"]), batch_size=BATCH_SIZE, device=device),
        qg.predict(samples_to_channels_first(data["test_x"]),        batch_size=BATCH_SIZE, device=device),
        data, fit_time,
    )
    save_results("gradient_new_only", results, results_dir)


# ---------------------------------------------------------------------------
# Method: finetune_new_only
# ---------------------------------------------------------------------------

def run_standard_finetune(data, results_dir, base_model_path, device, epochs):
    model = load_base(base_model_path, device)
    phase2_x = samples_to_channels_first(data["phase2_train_x"])

    t0 = time.time()
    _torch_train(model, phase2_x, data["phase2_train_y"], device, epochs, BATCH_SIZE)
    fit_time = time.time() - t0

    results = eval_splits(
        model_predict(model, data["test_phase1_x"], device),
        model_predict(model, data["test_phase2_x"], device),
        model_predict(model, data["test_x"],        device),
        data, fit_time,
    )
    save_results("finetune_new_only", results, results_dir)


# ---------------------------------------------------------------------------
# Method: cumulative
# ---------------------------------------------------------------------------

def run_cumulative(data, results_dir, base_model_path, device, epochs):
    model = load_base(base_model_path, device)

    all_x = np.concatenate([
        samples_to_channels_first(data["phase1_train_x"]),
        samples_to_channels_first(data["phase2_train_x"]),
    ])
    all_y = np.concatenate([data["phase1_train_y"], data["phase2_train_y"]])
    idx = np.random.default_rng(42).permutation(len(all_x))
    all_x, all_y = all_x[idx], all_y[idx]

    t0 = time.time()
    _torch_train(model, all_x, all_y, device, epochs, BATCH_SIZE)
    fit_time = time.time() - t0

    results = eval_splits(
        model_predict(model, data["test_phase1_x"], device),
        model_predict(model, data["test_phase2_x"], device),
        model_predict(model, data["test_x"],        device),
        data, fit_time,
    )
    save_results("cumulative", results, results_dir)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def execute_logic(method, data, results_dir, base_model_path, args, device):
    if method == "base":
        run_base(data, results_dir, base_model_path, device)
    elif method == "linear_accum":
        run_qsimov_linear_accum(data, results_dir, base_model_path, device)
    elif method == "linear_new_only":
        run_qsimov_linear_new_only(data, results_dir, base_model_path, device)
    elif method == "gradient_new_only":
        run_qsimov_gradient(data, results_dir, base_model_path, device, args.epochs)
    elif method == "finetune_new_only":
        run_standard_finetune(data, results_dir, base_model_path, device, args.epochs)
    elif method == "cumulative":
        run_cumulative(data, results_dir, base_model_path, device, args.epochs)


def main(args):
    device = torch.device("cuda" if args.processor == "gpu" else "cpu")
    results_dir = get_cifar10_forgetting_results_dir("pytorch", args.processor)
    data_dir    = get_data_dir()
    base_model_path = os.path.join(results_dir, "base_model.pt")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(base_model_path):
        raise FileNotFoundError(
            f"Base model not found at {base_model_path}. "
            "Run pytorch_train_base_model.py first."
        )

    data = load_data(data_dir)

    methods = [
        "base",
        "linear_accum",
        "linear_new_only",
        "gradient_new_only",
        "finetune_new_only",
        "cumulative",
    ]
    for method in methods:
        print(f"\n\nRunning: {method}\n")
        execute_logic(method, data, results_dir, base_model_path, args, device)


class RunForgettingParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="cpu")
        self.add_argument("--epochs", type=int, default=5)


if __name__ == "__main__":
    main(RunForgettingParser().parse_args())
