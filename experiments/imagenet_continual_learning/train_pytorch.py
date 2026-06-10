"""Train the initial models for the continual learning experiment — PyTorch backend.

Three models are saved:
  vgg16_path_selector_softmax_model.pt  — Full-data softmax model.
  vgg16_path_selector_linear_model.pt   — Full-data linear-output model.
  vgg16_standard_model.pt               — Copy of softmax model for standard baselines.
"""

import os
import pickle
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from experiments.path_utils import get_imagenet_continual_learning_results_dir
from experiments.imagenet_continual_learning.preprocess_data import (
    load_dataset,
    NUM_LABELS,
)
from experiments.imagenet_continual_learning.pytorch_model_factory import (
    path_selector_vgg16_softmax,
    path_selector_vgg16_linear,
)

BATCH_SIZE = 64


def _to_nchw_float32(x_uint8_nhwc):
    return np.transpose(x_uint8_nhwc.astype(np.float32), (0, 3, 1, 2))


def _make_dataloader(x_nhwc, y, batch_size, shuffle):
    x_nchw = _to_nchw_float32(x_nhwc)
    x_t = torch.from_numpy(x_nchw)
    y_t = torch.from_numpy(y)
    return DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=shuffle)


def _train_one_epoch(model, dl, loss_fn, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (pred.argmax(1) == (yb if yb.ndim == 1 else yb.argmax(1))).sum().item()
        total += len(yb)
    return total_loss / len(dl), correct / total


def _val_one_epoch(model, dl, loss_fn, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total_loss += loss_fn(pred, yb).item() * len(yb)
            correct += (pred.argmax(1) == (yb if yb.ndim == 1 else yb.argmax(1))).sum().item()
            total += len(yb)
    return total_loss / total, correct / total


def train_model(model, x_train, y_train, x_test, y_test, epochs, device, use_onehot=False):
    if use_onehot:
        loss_fn = nn.MSELoss()
        y_tr = np.eye(NUM_LABELS, dtype=np.float32)[y_train.astype(int)]
        y_te = np.eye(NUM_LABELS, dtype=np.float32)[y_test.astype(int)]
        y_tr_t = torch.from_numpy(y_tr)
        y_te_t = torch.from_numpy(y_te)
    else:
        loss_fn = nn.CrossEntropyLoss()
        y_tr_t = torch.from_numpy(y_train.astype(np.int64))
        y_te_t = torch.from_numpy(y_test.astype(np.int64))

    x_tr_t = torch.from_numpy(_to_nchw_float32(x_train))
    x_te_t = torch.from_numpy(_to_nchw_float32(x_test))
    train_dl = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)
    test_dl  = DataLoader(TensorDataset(x_te_t, y_te_t), batch_size=BATCH_SIZE, shuffle=False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=1e-4)
    model.to(device)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "time(s)": []}
    cumulative_time = 0.0

    for epoch in range(epochs):
        t0 = time.time()
        tl, ta = _train_one_epoch(model, train_dl, loss_fn, opt, device)
        vl, va = _val_one_epoch(model, test_dl, loss_fn, device)
        cumulative_time += time.time() - t0

        history["loss"].append(tl)
        history["accuracy"].append(ta)
        history["val_loss"].append(vl)
        history["val_accuracy"].append(va)
        history["time(s)"].append(cumulative_time)
        print(f"Epoch {epoch+1}/{epochs}  loss={tl:.4f} acc={ta:.4f}  "
              f"val_loss={vl:.4f} val_acc={va:.4f}")

    return history


def execute_logic(results_dir, tag, x_train, y_train, x_test, y_test, epochs, device):
    if tag == "path_selector_linear":
        model = path_selector_vgg16_linear()
        use_onehot = True
    else:
        model = path_selector_vgg16_softmax()
        use_onehot = False

    history = train_model(model, x_train, y_train, x_test, y_test, epochs, device,
                          use_onehot=use_onehot)

    torch.save(model, f"{results_dir}/vgg16_{tag}_model.pt")

    with open(f"{results_dir}/vgg16_{tag}_history.pkl", "wb") as f:
        pickle.dump(history, f)

    with open(f"{results_dir}/vgg16_{tag}_model_summary.txt", "w") as f:
        f.write(str(model))

    if tag == "path_selector_softmax":
        torch.save(model, f"{results_dir}/vgg16_standard_model.pt")

    print(f"\nSaved {tag} model to {results_dir}")


def main(args):
    device = torch.device("cuda" if args.processor == "gpu" else "cpu")
    results_dir = get_imagenet_continual_learning_results_dir(args.processor, framework="pytorch")
    os.makedirs(results_dir, exist_ok=True)

    x_train, y_train, x_test, y_test = load_dataset()

    for tag in ("path_selector_softmax", "path_selector_linear"):
        print(f"\n\nTraining {tag} model on FULL data (all classes)\n")
        execute_logic(results_dir, tag, x_train, y_train, x_test, y_test, args.epochs, device)


class TrainModelsParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="gpu")
        self.add_argument("--epochs", type=int, required=True)


if __name__ == "__main__":
    main(TrainModelsParser().parse_args())
