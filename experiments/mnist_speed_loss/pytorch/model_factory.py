import torch.nn as nn


def build_mnist(in_channels=1, loss="categorical_crossentropy"):
    model = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(in_features=400, out_features=32),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=32, out_features=16),
        nn.ReLU(),
        nn.Linear(in_features=16, out_features=10),
        nn.Softmax(dim=1)
        if loss == "categorical_crossentropy"
        else nn.Identity(),
    )
    print(model)
    return model
