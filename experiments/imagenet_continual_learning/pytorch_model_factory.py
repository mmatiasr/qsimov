"""PyTorch VGG16 factory for imagenet_continual_learning experiments.

Two variants with narrow top (matching Keras CL architecture):
  path_selector_vgg16_softmax  — softmax output for QsimovGradient/finetune
  path_selector_vgg16_linear   — linear output for QsimovLinearSystem
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

from experiments.imagenet_continual_learning.preprocess_data import NUM_LABELS


class _Preprocess(nn.Module):
    def forward(self, x):
        return x / 127.5 - 1.0


def _vgg16_features():
    vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
    for p in vgg.features.parameters():
        p.requires_grad = False
    return list(vgg.features.children())


def _build_vgg16_top(last_activation):
    layers = [
        _Preprocess(),
        *_vgg16_features(),
        nn.AdaptiveAvgPool2d((7, 7)),
        nn.Flatten(),
        nn.Linear(25088, 2048), nn.ReLU(inplace=True),
        nn.Linear(2048, 128),   nn.ReLU(inplace=True),
        nn.Linear(128, NUM_LABELS),
    ]
    if last_activation == "softmax":
        layers.append(nn.Softmax(dim=1))
    else:
        # Identity keeps the layer count equal to the softmax model so that
        # initial_layer=-2 lands on Linear(128→NUM_LABELS) in both variants.
        layers.append(nn.Identity())
    return nn.Sequential(*layers)


def path_selector_vgg16_softmax():
    return _build_vgg16_top("softmax")


def path_selector_vgg16_linear():
    return _build_vgg16_top("linear")


def get_optimizer(model_name, is_qsimov=False):
    lr = 1e-5 if is_qsimov else 1e-4
    return lambda params: torch.optim.Adam(params, lr=lr)


def load_model(results_dir, tag):
    """Load saved .pt model. tag ∈ {'path_selector_softmax', 'path_selector_linear', 'standard'}."""
    return torch.load(f"{results_dir}/vgg16_{tag}_model.pt")
