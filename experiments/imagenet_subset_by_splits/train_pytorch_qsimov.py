"""Train Qsimov gradient model on ImageNet subset splits — PyTorch backend.

Mirrors train_keras_qsimov.py: loads a pre-trained path-selector model,
wraps it in PytorchPathSelector + PytorchQsimovGradient, and trains on
the requested dataset splits.
"""

import pickle
import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from experiments.path_utils import (
    get_imagenet_subset_by_splits_results_dir as get_results_dir,
)
from experiments.imagenet_subset_by_splits.preprocess_data import (
    load_dataset,
    INPUT_SHAPE_NCHW,
    NUM_LABELS,
)
from experiments.imagenet_subset_by_splits.train_pytorch import (
    TrainModelsParser,
    split_to_name,
    BATCH_SIZE,
)
from qsimov.pytorch_path_selector import PytorchPathSelector
from qsimov.pytorch_qsimov_gradient import PytorchQsimovGradient


def _to_nchw_float32(x_uint8_nhwc):
    return np.transpose(x_uint8_nhwc.astype(np.float32), (0, 3, 1, 2))


def save_results(name, model_name, model, history, results_dir):
    name += "_qsimov"
    with open(f"{results_dir}/{name}_{model_name}_history.pkl", "wb") as f:
        pickle.dump(history, f)

    n_paths = int(np.sum(model._path_selector.output_masks_, axis=1).max())
    with open(f"{results_dir}/number_of_paths_{model_name}.txt", "w") as f:
        f.write(str(n_paths))


def accuracy(y_true, y_pred):
    return float((y_true.argmax(1) == y_pred.argmax(1)).float().mean())


def make_qsimov_model(results_dir, split, args, device):
    model_file = f"{split_to_name(split)}_path_selector_{args.model_name}_model.pt"
    base_model = torch.load(os.path.join(results_dir, model_file), map_location=device, weights_only=False)

    path_selector = PytorchPathSelector(
        neural_network=base_model,
        input_shape=INPUT_SHAPE_NCHW,
        initial_layer=args.initial_layer,
        verbose=1,
        device=device,
    )
    qg = PytorchQsimovGradient(path_selector)
    return qg


def execute_logic(results_dir, split, args, device):
    train_x_raw, train_y, test_x_raw, test_y = load_dataset()

    train_x = _to_nchw_float32(train_x_raw)
    test_x  = _to_nchw_float32(test_x_raw)
    train_y_oh = np.eye(NUM_LABELS, dtype=np.float32)[train_y.astype(int)]
    test_y_oh  = np.eye(NUM_LABELS, dtype=np.float32)[test_y.astype(int)]

    model = make_qsimov_model(results_dir, split, args, device)

    history = model.fit(
        train_x,
        train_y_oh,
        X_val=test_x,
        Y_val=test_y_oh,
        batch_size=BATCH_SIZE,
        epochs=args.epochs,
        loss_function=nn.CrossEntropyLoss(),
        metrics=[accuracy],
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-5),
        device=device,
    )

    save_results(split_to_name(split), args.model_name, model, history, results_dir)


def main(args):
    device = torch.device("cuda" if args.processor == "gpu" else "cpu")
    results_dir = get_results_dir("pytorch", args.processor)
    os.makedirs(results_dir, exist_ok=True)

    for split in args.splits:
        print(f"\n\nTraining Qsimov model with split: {split_to_name(split)}\n")
        execute_logic(results_dir, split, args, device)


class TrainQsimovModelsParser(TrainModelsParser):
    def add_arguments(self):
        self.add_processor_argument()
        self.add_epochs_argument()
        self.add_splits_argument()
        self.add_model_name_argument()
        self.add_initial_layer_argument()

    def add_initial_layer_argument(self):
        self.add_argument("--initial-layer", type=int, default=-2)


if __name__ == "__main__":
    main(TrainQsimovModelsParser().parse_args())
