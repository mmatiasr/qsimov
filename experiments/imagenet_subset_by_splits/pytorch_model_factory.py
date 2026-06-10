"""PyTorch VGG16 factory for imagenet_subset_by_splits experiments.

Two variants:
  original_vgg16     — full-width top (4096→4096→NUM_LABELS), softmax
  path_selector_vgg16 — narrow top (2048→128→NUM_LABELS), softmax or linear

The model is a flat nn.Sequential so PytorchPathSelector can count layers
from the end with negative initial_layer indices.

Preprocessing (x/127.5 − 1.0) is built into the first module, matching
the Lambda layer in the Keras version, so path selection features are
computed on preprocessed data.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

from experiments.imagenet_subset_by_splits.preprocess_data import NUM_LABELS

ORIGINAL_SHAPE = (224, 224, 3)


class _Preprocess(nn.Module):
    """Scale uint8-range float32 tensors to [-1, 1]."""
    def forward(self, x):
        return x / 127.5 - 1.0


def _vgg16_features():
    """Return frozen VGG16 feature extractor layers."""
    vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
    for p in vgg.features.parameters():
        p.requires_grad = False
    return list(vgg.features.children())


def original_vgg16(activation="softmax"):
    """Full-width top: flatten(25088)→4096→4096→NUM_LABELS."""
    layers = [
        _Preprocess(),
        *_vgg16_features(),
        nn.AdaptiveAvgPool2d((7, 7)),
        nn.Flatten(),
        nn.Linear(25088, 4096), nn.ReLU(inplace=True),
        nn.Linear(4096, 4096),  nn.ReLU(inplace=True),
        nn.Linear(4096, NUM_LABELS),
    ]
    if activation == "softmax":
        layers.append(nn.Softmax(dim=1))
    return nn.Sequential(*layers)


def path_selector_vgg16(activation="softmax"):
    """Narrow top: flatten(25088)→2048→128→NUM_LABELS."""
    layers = [
        _Preprocess(),
        *_vgg16_features(),
        nn.AdaptiveAvgPool2d((7, 7)),
        nn.Flatten(),
        nn.Linear(25088, 2048), nn.ReLU(inplace=True),
        nn.Linear(2048, 128),   nn.ReLU(inplace=True),
        nn.Linear(128, NUM_LABELS),
    ]
    if activation == "softmax":
        layers.append(nn.Softmax(dim=1))
    return nn.Sequential(*layers)


def get_optimizer(model_name, is_qsimov=False):
    lr = 1e-5 if is_qsimov else 1e-4
    return lambda params: torch.optim.Adam(params, lr=lr)


def load_model(model_name, path_selector=False, activation="softmax"):
    if model_name != "vgg16":
        raise ValueError(f"Unknown model: {model_name}")
    if path_selector:
        return path_selector_vgg16(activation=activation)
    return original_vgg16(activation=activation)
