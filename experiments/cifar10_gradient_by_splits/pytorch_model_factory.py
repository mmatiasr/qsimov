import torch.nn as nn
from torchvision import models
import torch
import pickle
import os
import argparse
import numpy as np
from experiments.path_utils import (
    get_cifar10_gradient_by_splits_results_initial_weights_dir,
)


def pretained_build_vgg16():
    model = models.vgg16(weights="DEFAULT")

    # not present in keras
    del model.avgpool

    # Add flatten layer
    model.flatten = nn.Sequential(nn.Flatten())

    # Input dense layer
    num_filters_previous_layer = model.features[28].out_channels
    model.classifier[0] = nn.Linear(num_filters_previous_layer, 512)
    num_filters_previous_layer = model.classifier[0].out_features
    model.classifier[3] = nn.Linear(num_filters_previous_layer, 512)
    num_filters_previous_layer = model.classifier[3].out_features
    model.classifier[6] = nn.Linear(num_filters_previous_layer, 10)
    model = nn.Sequential(
        model.features, model.flatten, model.classifier, nn.Softmax(dim=1)
    )
    print(model)
    return model


def build_vgg16(include_top=True):
    top_layers = []
    if include_top:
        top_layers = [
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 10),
            nn.Softmax(dim=1),
        ]

    model = nn.Sequential(
        # Block 1
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Block 2
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Block 3
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Block 4
        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Block 5
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Top
        *top_layers,
    )
    print(model)
    return model


def build_path_selector_vgg16(include_top=True):
    top_layers = []
    if include_top:
        top_layers = [
            nn.Flatten(),
            nn.Linear(8, 10),
            nn.Softmax(dim=1),
        ]

    model = nn.Sequential(
        # Block 1
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Block 2
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Block 3
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Block 4
        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Block 5
        nn.Conv2d(512, 12, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(12, 12, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(12, 8, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Top
        *top_layers,
    )
    print(model)
    return model


def build_lenet():
    model = nn.Sequential(
        nn.Conv2d(3, 20, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(20, 50, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(3200, 500),  # As the output shape from Flatten layer is 3200
        nn.ReLU(),
        nn.Linear(500, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
        nn.Softmax(dim=1),
    )

    print(model)
    return model


def build_path_selector_lenet():
    """LeNet without final Softmax so PytorchPathSelector gets a linear output
    and initial_layer=-1 resolves to the last Linear (not the Softmax)."""
    model = nn.Sequential(
        nn.Conv2d(3, 20, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(20, 50, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(3200, 500),
        nn.ReLU(),
        nn.Linear(500, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )

    print(model)
    return model


def build_alexnet():
    model = nn.Sequential(
        nn.Conv2d(3, 96, kernel_size=1, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(4096, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 120),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(120, 10),
        nn.Softmax(dim=1),
    )
    print(model)

    return model


def get_optimizer(model_name, params):
    # Define optimizer for each pytorch model
    if model_name == "vgg16" or "alexnet":
        optimizer = torch.optim.Adam(params, lr=1e-4)
    elif model_name == "lenet":
        optimizer = torch.optim.SGD(
            params, momentum=0.9, lr=1e-2, nesterov=True
        )
    else:
        raise ValueError("Optimizer not supported")

    return optimizer


def load_weights_pytorch_format(
    pytorch_model, model_name, model_type, load_path
):
    # Load the weights and biases from the saved file
    with open(f"{load_path}/{model_name}{model_type}_weights.pkl", "rb") as f:
        weights = pickle.load(f)

    # Assign the loaded weights and biases to the PyTorch model
    for layer in pytorch_model.children():
        if type(layer) == nn.Conv2d:
            conv_weights, conv_biases = weights["conv"].pop(0)

            layer.weight.data = torch.from_numpy(conv_weights)
            layer.bias.data = torch.from_numpy(conv_biases)
        elif type(layer) == nn.Linear:
            dense_weights, dense_biases = weights["dense"].pop(0)

            layer.weight.data = torch.from_numpy(dense_weights)
            layer.bias.data = torch.from_numpy(dense_biases)
    return pytorch_model


def create_model(name, model_type):
    # build pytorch model
    if name == "alexnet":
        model = build_alexnet()
    elif name == "lenet":
        model = build_path_selector_lenet() if model_type else build_lenet()
    elif name == "vgg16":
        if model_type:
            model = build_path_selector_vgg16()
        else:
            model = build_vgg16()
    else:
        raise ValueError("Unknown model name: {}".format(name))
    # load weigths
    weights_dir = get_cifar10_gradient_by_splits_results_initial_weights_dir()
    model = load_weights_pytorch_format(model, name, model_type, weights_dir)
    return model


def save_initial_weights(model, model_name, model_type, save_path):
    weights = {"conv": [], "dense": []}
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            weights["conv"].append((
                layer.weight.data.cpu().numpy().copy(),
                layer.bias.data.cpu().numpy().copy(),
            ))
        elif isinstance(layer, nn.Linear):
            weights["dense"].append((
                layer.weight.data.cpu().numpy().copy(),
                layer.bias.data.cpu().numpy().copy(),
            ))
    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/{model_name}{model_type}_weights.pkl", "wb") as f:
        pickle.dump(weights, f, protocol=5)
    print(f"Saved weights: {save_path}/{model_name}{model_type}_weights.pkl")


def _build_model(model_name, model_type):
    if model_name == "lenet":
        return build_path_selector_lenet() if model_type else build_lenet()
    elif model_name == "alexnet":
        return build_alexnet()
    elif model_name == "vgg16":
        return build_path_selector_vgg16() if model_type else build_vgg16()
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    weights_dir = get_cifar10_gradient_by_splits_results_initial_weights_dir()

    for model_type in ("", "_path_selector"):
        model = _build_model(args.model_name, model_type)
        save_initial_weights(model, args.model_name, model_type, weights_dir)


if __name__ == "__main__":
    main()
