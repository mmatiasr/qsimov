import numpy as np
import os
import torch
import torch.nn as nn


def make_list_weights_3221(dropout=False):
    W1 = np.array([[2, -2], [-3, -4], [2, 1]], dtype=float)
    W2 = np.array([[2, -2], [0, -2]], dtype=float)
    W3 = np.array([[-3], [3]], dtype=float)

    if dropout:
        return [W1, None, W2, W3]
    return [W1, W2, W3]


def make_list_biases_3221(dropout=False):
    b1 = np.array([-1, -3], dtype=float)
    b2 = np.array([1, 1], dtype=float)
    b3 = np.array([1], dtype=float)

    if dropout:
        return [b1, None, b2, b3]
    return [b1, b2, b3]


def make_list_weights_222():
    W1 = np.array([[1, -1], [1, 1]], dtype=float)
    W2 = np.array([[2, 1], [-1, 2]], dtype=float)

    list_weights = [W1, W2]

    return list_weights


def make_list_biases_222():
    b1 = np.array([1, -1], dtype=float)
    b2 = np.array([0, 1], dtype=float)

    list_biases = [b1, b2]

    return list_biases


def make_data_dir():
    test_dir = os.path.dirname(__file__)
    test_data_dir = os.path.join(test_dir, "test_data")
    os.makedirs(test_data_dir, exist_ok=True)
    return test_data_dir


def make_data_3652(number_samples=1000):
    # data generation parameters
    number_outputs = 2
    number_features = 3

    X = np.random.uniform(
        low=-5, high=5, size=(number_samples, number_features)
    )
    Y = np.empty((number_samples, number_outputs))

    # problem 1: simple linear problem
    Y[:, 0] = (
        3 * X[:, 0]
        - 2.5 * X[:, 1]
        + 0.3 * X[:, 2]
        + np.random.standard_normal(size=number_samples) * 0.2
    )

    # problem 2: more complex expression
    Y[:, 1] = (
        (np.abs(X[:, 2] * X[:, 1]) + 1) / (X[:, 0] ** 2 + 1)
    ) - np.random.standard_normal(size=number_samples) * 0.1

    return X, Y


def make_data_3652_sigmoid(number_samples=1000):
    X, Y = make_data_3652(number_samples)
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    Y = 1 / (1 + np.exp(Y))
    return X, Y


def make_model_3652():
    X, Y = make_data_3652()
    data_dir = make_data_dir()
    if not os.path.exists(os.path.join(data_dir, "model_3652.pt")):
        model = nn.Sequential(
            nn.Linear(3, 6).double(),
            nn.ReLU(),
            nn.Linear(6, 5).double(),
            nn.ReLU(),
            nn.Linear(5, 2).double(),
        )
        torch.save(model, os.path.join(data_dir, "model_3652.pt"))
    return torch.load(os.path.join(data_dir, "model_3652.pt"))


def make_model_3652_sigmoid():
    data_dir = make_data_dir()
    if not os.path.exists(os.path.join(data_dir, "model_3652_sigmoid.pt")):
        model = nn.Sequential(
            nn.Linear(3, 6).double(),
            nn.ReLU(),
            nn.Linear(6, 5).double(),
            nn.ReLU(),
            nn.Linear(5, 2).double(),
            nn.Sigmoid(),
        )
        torch.save(model, os.path.join(data_dir, "model_3652_sigmoid.pt"))
    return torch.load(os.path.join(data_dir, "model_3652_sigmoid.pt"))


def make_model_222():
    # Define the model
    model = nn.Sequential(
        nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2), nn.ReLU()
    )
    # Set the weights and biases of the model
    list_weights = make_list_weights_222()
    list_biases = make_list_biases_222()

    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(list_weights[0].T))
        model[0].bias.copy_(torch.from_numpy(list_biases[0]))
        model[2].weight.copy_(torch.from_numpy(list_weights[1].T))
        model[2].bias.copy_(torch.from_numpy(list_biases[1]))

    return model


def make_model_3221(dropout=False):
    # Define the model
    model = nn.Sequential(
        nn.Linear(3, 2),
        nn.ReLU(),
        nn.Dropout(p=0.1) if dropout else None,
        nn.Linear(2, 2),
        nn.ReLU(),
        nn.Linear(2, 1),
    )

    list_weights = make_list_weights_3221(dropout=False)
    list_biases = make_list_biases_3221(dropout=False)

    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(list_weights[0].T))
        model[0].bias.copy_(torch.from_numpy(list_biases[0]))
        model[3].weight.copy_(torch.from_numpy(list_weights[1].T))
        model[3].bias.copy_(torch.from_numpy(list_biases[1]))
        model[5].weight.copy_(torch.from_numpy(list_weights[2].T))
        model[5].bias.copy_(torch.from_numpy(list_biases[2]))

    return model
