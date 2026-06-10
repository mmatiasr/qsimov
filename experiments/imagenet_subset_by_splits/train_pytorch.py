"""Train VGG16 on ImageNet subset splits — PyTorch backend.

Mirrors train_keras.py: trains VGG16 (frozen conv base, trainable Dense top)
on the requested dataset splits, saving models and history pkl files.
"""

import pickle
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from experiments.path_utils import (
    get_imagenet_subset_by_splits_results_dir as get_results_dir,
)
from experiments.imagenet_subset_by_splits.preprocess_data import (
    load_dataset,
    make_split,
)
from experiments.imagenet_subset_by_splits.pytorch_model_factory import (
    load_model,
    get_optimizer,
)

BATCH_SIZE = 64


def split_to_name(split):
    return f"{split}_split" if split is not None else "full_dataset"


def _make_dataloader(x_uint8_nhwc, y_int, batch_size, shuffle):
    """Create DataLoader with on-the-fly uint8→float32 NCHW conversion."""
    x_nchw = np.transpose(x_uint8_nhwc.astype(np.float32), (0, 3, 1, 2))
    x_t = torch.from_numpy(x_nchw)
    y_t = torch.from_numpy(y_int.astype(np.int64))
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
        correct += (pred.argmax(1) == yb).sum().item()
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
            correct += (pred.argmax(1) == yb).sum().item()
            total += len(yb)
    return total_loss / total, correct / total


def train_model(model, train_x, train_y, test_x, test_y, epochs, device):
    loss_fn = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=1e-4)

    train_dl = _make_dataloader(train_x, train_y, BATCH_SIZE, shuffle=True)
    test_dl  = _make_dataloader(test_x,  test_y,  BATCH_SIZE, shuffle=False)

    model.to(device)
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "time(s)": []}

    import time
    for epoch in range(epochs):
        t0 = time.time()
        tl, ta = _train_one_epoch(model, train_dl, loss_fn, opt, device)
        vl, va = _val_one_epoch(model, test_dl, loss_fn, device)
        elapsed = time.time() - t0
        history["loss"].append(tl)
        history["accuracy"].append(ta)
        history["val_loss"].append(vl)
        history["val_accuracy"].append(va)
        history["time(s)"].append(elapsed)
        print(f"Epoch {epoch+1}/{epochs}  loss={tl:.4f} acc={ta:.4f}  "
              f"val_loss={vl:.4f} val_acc={va:.4f}")
    return history


def save_results(split_name, model_name, model_type, model, history, results_dir):
    with open(f"{results_dir}/{split_name}{model_type}_{model_name}_history.pkl", "wb") as f:
        pickle.dump(history, f)

    torch.save(model, f"{results_dir}/{split_name}{model_type}_{model_name}_model.pt")

    with open(f"{results_dir}/{model_name}{model_type}_model_summary.txt", "w") as f:
        f.write(str(model))


def execute_logic(results_dir, split, args, device):
    train_x, train_y, test_x, test_y = load_dataset()

    if split is not None:
        train_x, train_y = make_split(train_x, train_y, split)

    model_type = "_path_selector" if args.train_path_selector else ""
    model = load_model(args.model_name, path_selector=args.train_path_selector)

    history = train_model(model, train_x, train_y, test_x, test_y, args.epochs, device)
    save_results(split_to_name(split), args.model_name, model_type, model, history, results_dir)


def main(args):
    device = torch.device("cuda" if args.processor == "gpu" else "cpu")
    results_dir = get_results_dir("pytorch", args.processor)
    os.makedirs(results_dir, exist_ok=True)

    splits = args.splits + [None] if not args.train_path_selector else args.splits

    for split in splits:
        print(f"\n\nTraining model with split: {split_to_name(split)}\n")
        execute_logic(results_dir, split, args, device)


class TrainModelsParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_arguments()

    def add_arguments(self):
        self.add_processor_argument()
        self.add_epochs_argument()
        self.add_splits_argument()
        self.add_model_name_argument()
        self.add_train_path_selector()

    def add_processor_argument(self):
        self.add_argument("--processor", choices=["cpu", "gpu"], default="gpu")

    def add_model_name_argument(self):
        self.add_argument("--model-name", choices=["vgg16"], default="vgg16")

    def add_epochs_argument(self):
        self.add_argument("--epochs", type=int, default=10)

    def add_splits_argument(self):
        self.add_argument("--splits", nargs="+", type=int, required=True)

    def add_train_path_selector(self):
        self.add_argument("--train-path-selector", action="store_true")


if __name__ == "__main__":
    main(TrainModelsParser().parse_args())
