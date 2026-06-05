import numpy as np
import os
import tensorflow as tf
from tensorflow import keras as kr

krl = kr.layers


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


def make_model_3652_sigmoid():
    X, Y = make_data_3652_sigmoid()
    data_dir = make_data_dir()
    if not os.path.exists(os.path.join(data_dir, "model_3652_sigmoid.h5")):
        model = kr.Sequential()
        model.add(
            kr.layers.Dense(
                6,
                input_shape=(3,),
                activation="relu",
                dtype=tf.float64,
            )
        )
        model.add(kr.layers.Dense(5, activation="relu", dtype=tf.float64))
        model.add(kr.layers.Dense(2, activation="sigmoid", dtype=tf.float64))
        model.compile(optimizer="adam", loss="categorical_crossentropy")
        model.fit(X, Y, epochs=50, batch_size=16)
        model.save(os.path.join(data_dir, "model_3652_sigmoid.h5"))
    return kr.models.load_model(
        os.path.join(data_dir, "model_3652_sigmoid.h5")
    )


def make_model_3652():
    X, Y = make_data_3652()
    data_dir = make_data_dir()
    if not os.path.exists(os.path.join(data_dir, "model_3652.h5")):
        model = kr.Sequential()
        model.add(
            kr.layers.Dense(
                6,
                input_shape=(3,),
                activation="relu",
                dtype=tf.float64,
            )
        )
        model.add(kr.layers.Dense(5, activation="relu", dtype=tf.float64))
        model.add(kr.layers.Dense(2, dtype=tf.float64))
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, Y, epochs=50, batch_size=16)
        model.save(os.path.join(data_dir, "model_3652.h5"))
    return kr.models.load_model(os.path.join(data_dir, "model_3652.h5"))


def make_model_222():
    # create model
    model = kr.Sequential()
    model.add(krl.Dense(2, input_shape=(2,), activation="relu"))
    model.add(krl.Dense(2, activation="relu"))
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    # "train" model
    list_weights = make_list_weights_222()
    list_biases = make_list_biases_222()

    for layer_idx in range(len(list_biases)):
        model.layers[layer_idx].set_weights(
            [list_weights[layer_idx], list_biases[layer_idx]]
        )

    return model


def make_model_3221(dropout=False):
    # create model
    model = kr.Sequential()
    model.add(krl.Dense(2, input_shape=(3,), activation="relu"))
    model.add(krl.Dropout(0.1))
    model.add(krl.Dense(2, activation="relu"))
    model.add(krl.Dense(1, activation="linear"))
    model.compile(optimizer="adam", loss="mse")

    # "train" model
    list_weights = make_list_weights_3221(dropout=dropout)
    list_biases = make_list_biases_3221(dropout=dropout)

    for layer_idx in range(len(list_biases)):
        if list_weights[layer_idx] is not None:
            model.layers[layer_idx].set_weights(
                [list_weights[layer_idx], list_biases[layer_idx]]
            )

    return model
