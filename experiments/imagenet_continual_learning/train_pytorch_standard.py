"""Standard fine-tuning baselines for the continual learning experiment — PyTorch backend.

standard_finetune     — Sequential fine-tuning without data replay (catastrophic forgetting).
standard_cumulative   — Retrain on all cumulative data each round (oracle upper bound).
"""

import os
import pickle
import time
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from experiments.path_utils import get_imagenet_continual_learning_results_dir
from experiments.imagenet_continual_learning.preprocess_data import (
    load_dataset,
    make_round_splits,
    NUM_LABELS,
    N_ROUNDS,
)

BATCH_SIZE = 64
EPOCHS_PER_ROUND = 5


def _to_nchw_float32(x_uint8_nhwc):
    return np.transpose(x_uint8_nhwc.astype(np.float32), (0, 3, 1, 2))


def _make_dataloader(x_nchw, y_int, batch_size, shuffle):
    x_t = torch.from_numpy(x_nchw)
    y_t = torch.from_numpy(y_int.astype(np.int64))
    return DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=shuffle)


def _train_model_on_data(model, x_nchw, y_int, device, epochs):
    loss_fn = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=1e-5)
    dl = _make_dataloader(x_nchw, y_int, BATCH_SIZE, shuffle=True)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = correct = total = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            correct += (pred.argmax(1) == yb).sum().item()
            total += len(yb)
        print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/len(dl):.4f}  "
              f"acc={correct/total:.4f}")


def accuracy(y_pred, y_true_int):
    return float(np.mean(np.argmax(y_pred, axis=1) == y_true_int))


def evaluate(model, x_nchw, y_int, device):
    model.eval()
    preds = []
    dl = DataLoader(
        TensorDataset(torch.from_numpy(x_nchw)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    y_pred = np.concatenate(preds)
    acc = accuracy(y_pred, y_int)
    eps = 1e-7
    y_clip = np.clip(y_pred, eps, 1 - eps)
    one_hot = np.eye(NUM_LABELS)[y_int.astype(int)]
    loss = float(-np.mean(np.sum(one_hot * np.log(y_clip), axis=1)))
    return {"accuracy": acc, "loss": loss}


def collect_per_round_metrics(model, rounds_nchw, current_k, device):
    per_round = {}
    for prev_k, (_, _, val_x, val_y) in enumerate(rounds_nchw[:current_k], 1):
        per_round[f"round_{prev_k}"] = evaluate(model, val_x, val_y, device)
    return per_round


def _convert_rounds(rounds):
    out = []
    for (tx, ty, vx, vy) in rounds:
        out.append((_to_nchw_float32(tx), ty, _to_nchw_float32(vx), vy))
    return out


def _load_standard_model(results_dir, device):
    return torch.load(f"{results_dir}/vgg16_standard_model.pt", map_location=device)


def execute_logic_finetune(results_dir, rounds_nchw, test_x, test_y, device):
    model = _load_standard_model(results_dir, device)
    results = {}
    cumulative_time = 0.0

    for k, (train_x, train_y, _, _) in enumerate(rounds_nchw, 1):
        t0 = time.time()
        _train_model_on_data(model, train_x, train_y, device, EPOCHS_PER_ROUND)
        cumulative_time += time.time() - t0

        round_key = f"after_round_{k}"
        results[round_key] = {
            "time(s)": cumulative_time,
            "overall": evaluate(model, test_x, test_y, device),
            "per_round_val": collect_per_round_metrics(model, rounds_nchw, k, device),
        }
        print(f"[finetune] round {k}/{N_ROUNDS}  "
              f"overall_acc={results[round_key]['overall']['accuracy']:.4f}  "
              f"time={cumulative_time:.1f}s")

    return results


def execute_logic_cumulative(results_dir, rounds_nchw, test_x, test_y, device):
    results = {}
    cumulative_time = 0.0
    cum_x, cum_y = [], []

    for k, (train_x, train_y, _, _) in enumerate(rounds_nchw, 1):
        cum_x.append(train_x)
        cum_y.append(train_y)
        cx = np.concatenate(cum_x)
        cy = np.concatenate(cum_y)

        model = _load_standard_model(results_dir, device)

        t0 = time.time()
        _train_model_on_data(model, cx, cy, device, EPOCHS_PER_ROUND)
        cumulative_time += time.time() - t0

        round_key = f"after_round_{k}"
        results[round_key] = {
            "time(s)": cumulative_time,
            "overall": evaluate(model, test_x, test_y, device),
            "per_round_val": collect_per_round_metrics(model, rounds_nchw, k, device),
        }
        print(f"[cumulative] round {k}/{N_ROUNDS}  "
              f"overall_acc={results[round_key]['overall']['accuracy']:.4f}  "
              f"time={cumulative_time:.1f}s")

        if k == N_ROUNDS:
            torch.save(model, f"{results_dir}/standard_cumulative_final_model.pt")

    return results


def main(args):
    device = torch.device("cuda" if args.processor == "gpu" else "cpu")
    results_dir = get_imagenet_continual_learning_results_dir(args.processor, framework="pytorch")
    os.makedirs(results_dir, exist_ok=True)

    x_train, y_train, x_test, y_test = load_dataset()
    rounds = make_round_splits(x_train, y_train, n_rounds=N_ROUNDS)
    rounds_nchw = _convert_rounds(rounds)
    test_x = _to_nchw_float32(x_test)

    for method in ("standard_finetune", "standard_cumulative"):
        print(f"\n\nRunning standard continual learning: {method}\n")
        if method == "standard_finetune":
            results = execute_logic_finetune(results_dir, rounds_nchw, test_x, y_test, device)
        else:
            results = execute_logic_cumulative(results_dir, rounds_nchw, test_x, y_test, device)

        output_file = f"{results_dir}/{method}_results.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        print(f"\nSaved results for {method} to {output_file}")


class TrainStandardParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="gpu")


if __name__ == "__main__":
    main(TrainStandardParser().parse_args())
