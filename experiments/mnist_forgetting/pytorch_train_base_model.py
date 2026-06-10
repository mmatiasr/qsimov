"""Train base MNIST model on phase 1 (classes 0-4) using PyTorch.

Uses MSE loss with linear final activation — required by QsimovLinearSystem.
Saves to results_dir/base_model.pt.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from experiments.path_utils import get_mnist_forgetting_results_dir
from experiments.mnist_forgetting.preprocess_data import load_data, get_data_dir
from experiments.mnist_speed_loss.pytorch.model_factory import build_mnist
from experiments.mnist_speed_loss.pytorch.utils import (
    init_torch,
    samples_to_channels_first,
    create_dataloader,
    fit,
)

BATCH_SIZE = 32
SEED = 42


def main(args):
    device = init_torch("cuda" if args.processor == "gpu" else "cpu", SEED)

    data_dir = get_data_dir()
    data = load_data(data_dir)

    results_dir = get_mnist_forgetting_results_dir("pytorch", args.processor)
    os.makedirs(results_dir, exist_ok=True)

    train_x = samples_to_channels_first(data["phase1_train_x"])
    train_y = data["phase1_train_y"]
    val_x   = samples_to_channels_first(data["test_phase1_x"])
    val_y   = data["test_phase1_y"]

    model = build_mnist(in_channels=1, loss="mse")

    train_dl = create_dataloader(train_x, train_y, BATCH_SIZE)
    val_dl   = create_dataloader(val_x, val_y, BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    fit(model, nn.MSELoss(), optimizer, train_dl, val_dl, device, args.epochs)

    acc_old = float(np.mean(
        np.argmax(_predict(model, data["phase1_train_x"], device), axis=1) ==
        np.argmax(data["test_phase1_y"], axis=1)
    ))
    acc_new = float(np.mean(
        np.argmax(_predict(model, data["test_phase2_x"], device), axis=1) ==
        np.argmax(data["test_phase2_y"], axis=1)
    ))
    print(f"\nBase model — acc_old={acc_old:.4f}  acc_new={acc_new:.4f}")
    print("(acc_new is near-random: model only saw classes 0-4)")

    model_path = os.path.join(results_dir, "base_model.pt")
    torch.save(model, model_path)
    print(f"Base model saved to {model_path}")


def _predict(model, x_nhwc, device, batch_size=256):
    x_nchw = samples_to_channels_first(x_nhwc)
    model.eval()
    model.to(device)
    chunks = []
    with torch.no_grad():
        for i in range(0, len(x_nchw), batch_size):
            batch = torch.from_numpy(x_nchw[i:i + batch_size]).to(device)
            chunks.append(model(batch).cpu().numpy())
    return np.concatenate(chunks)


class TrainBaseModelParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="cpu")
        self.add_argument("--epochs", type=int, default=10)


if __name__ == "__main__":
    main(TrainBaseModelParser().parse_args())
