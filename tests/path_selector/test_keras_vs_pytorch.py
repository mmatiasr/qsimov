import tensorflow as tf
import torch
import numpy as np
import torch.nn as nn
from tests.paths.utils_conv_tests import weights_to_channels_first
from qsimov.keras_path_selector import KerasPathSelector
from qsimov.pytorch_path_selector import PytorchPathSelector
from qsimov.paths.conv import _get_channels_last_to_channels_first_id_map
import os
from tensorflow import keras as kr


# GPU needs more precision to pass the tests
if torch.cuda.is_available():
    np_dtype = np.float64
    tf_dtype = tf.float64
    torch_dtype = torch.float64

    # with a bump in precision for GPU, we can also bump the testing threshold
    decimal = 5

else:
    np_dtype = np.float32
    tf_dtype = tf.float32
    torch_dtype = torch.float32
    decimal = 3


SEED = int(os.environ.get("QSIMOV_SEED", 42))


# define the weights and biases for a dense network in keras/pytorch
# and the corresponding models in each framework


# weights and biases for a dense network
def dense_weights(framework="keras"):
    # set the random seed for reproducibility
    np.random.seed(SEED)

    weights = [
        np.random.randn(28 * 28, 32).astype(np_dtype),
        np.random.randn(32, 16).astype(np_dtype),
        np.random.randn(16, 10).astype(np_dtype),
    ]
    biases = [
        np.random.randn(32).astype(np_dtype),
        np.random.randn(16).astype(np_dtype),
        np.random.randn(10).astype(np_dtype),
    ]
    if framework == "keras":
        return weights, biases
    elif framework == "pytorch":
        weights = [torch.from_numpy(w.T) for w in weights]
        biases = [torch.from_numpy(b) for b in biases]
        return weights, biases


# keras version of the dense network
def keras_dense_model():
    model = kr.Sequential(
        [
            kr.Input(shape=(28, 28, 1), dtype=tf_dtype),
            kr.layers.Flatten(),
            kr.layers.Dense(32, activation="relu", dtype=tf_dtype),
            kr.layers.Dense(16, activation="relu", dtype=tf_dtype),
            kr.layers.Dropout(0.2),
            kr.layers.Dense(10, activation="softmax", dtype=tf_dtype),
        ],
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=kr.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    # set weights and biases
    weights, biases = dense_weights(framework="keras")
    layers_to_set = [1, 2, 4]
    for idx, layer in enumerate(layers_to_set):
        model.layers[layer].set_weights([weights[idx], biases[idx]])
    return model


# path selector for the keras dense network
def keras_dense_path_selector():
    model = keras_dense_model()
    return KerasPathSelector(model, initial_layer=-3)


# pytorch version of the dense network
def pytorch_dense_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 32, dtype=torch_dtype),
        nn.ReLU(),
        nn.Linear(32, 16, dtype=torch_dtype),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(16, 10, dtype=torch_dtype),
        nn.Softmax(dim=1),
    )
    # set weights and biases
    weights, biases = dense_weights(framework="pytorch")
    layers_to_set = [1, 3, 6]
    for idx, layer_idx in enumerate(layers_to_set):
        layer = list(model.children())[layer_idx]
        layer.weight.data = weights[idx]
        layer.bias.data = biases[idx]
    return model


# path selector for the pytorch dense networkmaxpool_Weights
def pytorch_dense_path_selector():
    model = pytorch_dense_model()
    return PytorchPathSelector(
        model, initial_layer=-5, input_shape=(1, 28, 28)
    )


# weights and biases for a convolutional network
def conv_weights(framework="keras"):
    # set the random seed for reproducibility
    np.random.seed(SEED)

    weights = [
        np.random.randn(3, 3, 1, 32).astype(np_dtype),
        np.random.randn(3, 3, 32, 16).astype(np_dtype),
        np.random.randn(3, 3, 16, 8).astype(np_dtype),
    ]
    biases = [
        np.random.randn(32).astype(np_dtype),
        np.random.randn(16).astype(np_dtype),
        np.random.randn(8).astype(np_dtype),
    ]
    if framework == "keras":
        return weights, biases
    weights = [torch.from_numpy(weights_to_channels_first(w)) for w in weights]
    biases = [torch.from_numpy(b) for b in biases]
    return weights, biases


# keras version of the convolutional network
def keras_conv_model():
    model = kr.Sequential(
        [
            kr.Input(shape=(28, 28, 1), dtype=tf_dtype),
            kr.layers.Conv2D(
                32, kernel_size=(3, 3), activation="relu", dtype=tf_dtype
            ),
            kr.layers.MaxPooling2D(pool_size=(2, 2)),
            kr.layers.Conv2D(
                16, kernel_size=(3, 3), activation="relu", dtype=tf_dtype
            ),
            kr.layers.MaxPooling2D(pool_size=(2, 2)),
            kr.layers.Dropout(0.2),
            kr.layers.Conv2D(
                8, kernel_size=(3, 3), activation="relu", dtype=tf_dtype
            ),
        ],
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=kr.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    # set weights and biases
    weights, biases = conv_weights(framework="keras")
    layers_to_set = [0, 2, 5]
    for idx, layer in enumerate(layers_to_set):
        model.layers[layer].set_weights([weights[idx], biases[idx]])
    return model


# path selector for the keras convolutional network
def keras_conv_path_selector():
    model = keras_conv_model()
    return KerasPathSelector(model, initial_layer=-3)


# pytorch version of the convolutional network
def pytorch_conv_model():
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=(3, 3), dtype=torch_dtype),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Conv2d(32, 16, kernel_size=(3, 3), dtype=torch_dtype),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Dropout(0.2),
        nn.Conv2d(16, 8, kernel_size=(3, 3), dtype=torch_dtype),
        nn.ReLU(),
    )
    # set weights and biases
    weights, biases = conv_weights(framework="pytorch")
    layers_to_set = [0, 3, 7]
    for idx, layer_idx in enumerate(layers_to_set):
        layer = list(model.children())[layer_idx]
        layer.weight.data = weights[idx]
        layer.bias.data = biases[idx]
    return model


# path selector for the pytorch convolutional network
def pytorch_conv_path_selector():
    model = pytorch_conv_model()
    return PytorchPathSelector(
        model, initial_layer=-4, input_shape=(1, 28, 28)
    )


# convert channels last indices of an input array to channels first
def neuron_ids_to_channels_first(neuron_ids, channels_last_shape):
    # maps ids of neurons in channels last format to channels first format
    id_map = _get_channels_last_to_channels_first_id_map(channels_last_shape)
    return np.array([id_map[id] for id in neuron_ids])


# gets the indices to reorder paths in keras to paths in pytorch, needed
# beacuse keras uses channels last and pytorch uses channels first
def argsort_channels_last_to_channels_first_conv():
    conv_model = keras_conv_model()

    # get the keras path selector
    keras_path_selector = keras_conv_path_selector()

    # extract all paths
    paths = np.copy(keras_path_selector._all_paths)

    # map paths to channels first
    paths[:, 0] = neuron_ids_to_channels_first(
        paths[:, 0], conv_model.layers[-3].input_shape[1:]
    )
    paths[:, 1] = neuron_ids_to_channels_first(
        paths[:, 1], conv_model.layers[-1].input_shape[1:]
    )
    # no need to map the last layer, since it is all ones by design

    # argsort the paths
    def get_path_key(path):
        return path[0] * 1000 + path[1]

    return np.array(
        [
            idx
            for idx, _ in sorted(
                enumerate(paths),
                key=lambda idx_and_path: get_path_key(idx_and_path[1]),
            )
        ]
    )


# get the indices to reorder paths in the last layer of the convolutional
# network, needed because keras uses channels last and pytorch uses channels
# first
def argsort_channels_last_to_channels_first_last_layer_conv():
    conv_model = keras_conv_model()

    # get the keras path selector
    keras_path_selector = keras_conv_path_selector()

    # get the output_neuron_ids
    output_neuron_ids = np.arange(1, keras_path_selector._number_outputs + 1)

    # map paths to channels first
    output_neuron_ids = neuron_ids_to_channels_first(
        output_neuron_ids, conv_model.layers[-1].output_shape[1:]
    )

    return np.array(
        [
            idx
            for idx, _ in sorted(
                enumerate(output_neuron_ids),
                key=lambda idx_and_neuron_id: idx_and_neuron_id[1],
            )
        ]
    )


# weights and biases for a maxpool network
def maxpool_weights(framework="keras"):
    # set the random seed for reproducibility
    np.random.seed(SEED)

    weights = [
        np.random.randn(3, 3, 1, 32).astype(np_dtype),
        np.random.randn(3, 3, 32, 16).astype(np_dtype),
        np.random.randn(3, 3, 16, 8).astype(np_dtype),
    ]
    biases = [
        np.random.randn(32).astype(np_dtype),
        np.random.randn(16).astype(np_dtype),
        np.random.randn(8).astype(np_dtype),
    ]

    # To see errors in the maxpool network, uncomment the following lines
    # weights[0][1, 1, 0, 0] = np.array(10**5, dtype=np_dtype)
    # weights[1][1, 1, 16, 8] = np.array(10**5, dtype=np_dtype)
    # weights[2][1, 1, 8, 4] = np.array(10**5, dtype=np_dtype)

    if framework == "keras":
        return weights, biases
    weights = [torch.from_numpy(weights_to_channels_first(w)) for w in weights]
    biases = [torch.from_numpy(b) for b in biases]
    return weights, biases


# keras version of the maxpool network
def keras_maxpool_model():
    model = kr.Sequential(
        [
            kr.Input(shape=(28, 28, 1), dtype=tf_dtype),
            kr.layers.Conv2D(
                32, kernel_size=(3, 3), activation="relu", dtype=tf_dtype
            ),
            kr.layers.MaxPooling2D(pool_size=(2, 2)),
            kr.layers.Conv2D(
                16, kernel_size=(3, 3), activation="relu", dtype=tf_dtype
            ),
            kr.layers.MaxPooling2D(pool_size=(2, 2)),
            kr.layers.Conv2D(
                8, kernel_size=(3, 3), activation="relu", dtype=tf_dtype
            ),
            kr.layers.Dropout(0.2),
            kr.layers.MaxPooling2D(pool_size=(2, 2)),
        ],
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=kr.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    # set weights and biases
    weights, biases = maxpool_weights(framework="keras")
    layers_to_set = [0, 2, 4]
    for idx, layer in enumerate(layers_to_set):
        model.layers[layer].set_weights([weights[idx], biases[idx]])
    return model


# path selector for the keras maxpool network
def keras_maxpool_path_selector():
    model = keras_maxpool_model()
    return KerasPathSelector(model, initial_layer=-3)


# gets the indices to reorder paths in keras to paths in pytorch, needed
# beacuse keras uses channels last and pytorch uses channels first
def argsort_channels_last_to_channels_first_maxpool():
    maxpool_model = keras_maxpool_model()

    # get the keras path selector
    keras_path_selector = keras_maxpool_path_selector()

    # extract all paths
    paths = np.copy(keras_path_selector._all_paths)

    # map paths to channels first
    paths[:, 0] = neuron_ids_to_channels_first(
        paths[:, 0], maxpool_model.layers[-3].input_shape[1:]
    )
    paths[:, 1] = neuron_ids_to_channels_first(
        paths[:, 1], maxpool_model.layers[-1].input_shape[1:]
    )
    # no need to map the last layer, since it is all ones by design

    # argsort the paths
    def get_path_key(path):
        return path[0] * 1000 + path[1]

    return np.array(
        [
            idx
            for idx, _ in sorted(
                enumerate(paths),
                key=lambda idx_and_path: get_path_key(idx_and_path[1]),
            )
        ]
    )


# get the indices to reorder paths in the last layer of the maxpool
# network, needed because keras uses channels last and pytorch uses channels
# first
def argsort_channels_last_to_channels_first_last_layer_maxpool():
    maxpool_model = keras_maxpool_model()

    # get the keras path selector
    keras_path_selector = keras_maxpool_path_selector()

    # get the output_neuron_ids
    output_neuron_ids = np.arange(1, keras_path_selector._number_outputs + 1)

    # map paths to channels first
    output_neuron_ids = neuron_ids_to_channels_first(
        output_neuron_ids, maxpool_model.layers[-1].output_shape[1:]
    )

    return np.array(
        [
            idx
            for idx, _ in sorted(
                enumerate(output_neuron_ids),
                key=lambda idx_and_neuron_id: idx_and_neuron_id[1],
            )
        ]
    )


# pytorch version of the maxpool network
def pytorch_maxpool_model():
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=(3, 3), dtype=torch_dtype),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Conv2d(32, 16, kernel_size=(3, 3), dtype=torch_dtype),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Conv2d(16, 8, kernel_size=(3, 3), dtype=torch_dtype),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.MaxPool2d(kernel_size=(2, 2)),
    )
    # set weights and biases
    weights, biases = maxpool_weights(framework="pytorch")
    layers_to_set = [0, 3, 6]
    for idx, layer_idx in enumerate(layers_to_set):
        layer = list(model.children())[layer_idx]
        layer.weight.data = weights[idx]
        layer.bias.data = biases[idx]
    return model


# path selector for the pytorch maxpool network
def pytorch_maxpool_path_selector():
    model = pytorch_maxpool_model()
    return PytorchPathSelector(
        model, initial_layer=-4, input_shape=(1, 28, 28)
    )


# sample data for the mnist dataset
def sample_data(framework="keras"):
    number_samples = 1000
    np.random.seed(SEED)
    if framework == "keras":
        return np.random.randn(number_samples, 28, 28, 1).astype(np_dtype)
    return torch.from_numpy(
        np.random.randn(number_samples, 1, 28, 28).astype(np_dtype)
    )


# test that for the dense model, coefficients are the same
def test_dense_coefficients():
    # in keras
    data = sample_data("keras")
    keras_path_selector = keras_dense_path_selector()
    keras_coefficients = keras_path_selector.samples_to_coefficients(data)

    # in pytorch
    data = sample_data("pytorch")
    pytorch_path_selector = pytorch_dense_path_selector()
    pytorch_coefficients = pytorch_path_selector.samples_to_coefficients(data)

    # check that the weights are the same in the left model
    np.testing.assert_array_equal(
        keras_path_selector.left_model_.layers[1].get_weights()[0],
        pytorch_path_selector.left_model_[1].weight.data.numpy().T,
    )
    # check that the weights are the same in the right model
    np.testing.assert_array_equal(
        keras_path_selector.right_model_.layers[0].get_weights()[0],
        pytorch_path_selector.right_model_[0].weight.data.numpy().T,
    )

    # check that the coefficients are the same
    # zeroes should be equal with no regards to computation precision
    np.testing.assert_array_equal(
        keras_coefficients == 0, pytorch_coefficients == 0
    )
    np.testing.assert_array_almost_equal(
        keras_coefficients, pytorch_coefficients, decimal=decimal + 1
    )

    # compare also the output masks
    np.testing.assert_array_equal(
        keras_path_selector.output_masks_, pytorch_path_selector.output_masks_
    )


# test that for the conv model, coefficients are the same (after reordering)
def test_conv_coefficients():
    # in keras
    data = sample_data("keras")
    keras_path_selector = keras_conv_path_selector()
    keras_coefficients = keras_path_selector.samples_to_coefficients(data)

    # we need to convert the keras coefficients to channels first
    reordering_indices = argsort_channels_last_to_channels_first_conv()
    keras_coefficients = keras_coefficients[:, reordering_indices]

    # in pytorch
    data = sample_data("pytorch")
    pytorch_path_selector = pytorch_conv_path_selector()
    pytorch_coefficients = pytorch_path_selector.samples_to_coefficients(data)

    # check that the coefficients are the same
    # zeroes should be equal with no regards to computation precision
    np.testing.assert_array_equal(
        keras_coefficients == 0, pytorch_coefficients == 0
    )
    np.testing.assert_array_almost_equal(
        keras_coefficients, pytorch_coefficients, decimal=decimal
    )

    # compare also the output masks
    # reoerdering indices for the output layer
    out_reordering_indices = (
        argsort_channels_last_to_channels_first_last_layer_conv()
    )
    # reorder on the first axis
    keras_output_masks = keras_path_selector.output_masks_[
        out_reordering_indices, :
    ]
    np.testing.assert_array_equal(
        keras_output_masks[:, reordering_indices],  # reorder on second axis
        pytorch_path_selector.output_masks_,
    )


# test that for the maxpool model, coefficients are the same (after reordering)
def test_maxpool_coefficients():
    # in keras
    data = sample_data("keras")
    keras_path_selector = keras_maxpool_path_selector()
    keras_coefficients = keras_path_selector.samples_to_coefficients(data)

    # we need to convert the keras coefficients to channels first
    reordering_indices = argsort_channels_last_to_channels_first_maxpool()
    keras_coefficients = keras_coefficients[:, reordering_indices]

    # in pytorch
    data = sample_data("pytorch")
    pytorch_path_selector = pytorch_maxpool_path_selector()
    pytorch_coefficients = pytorch_path_selector.samples_to_coefficients(data)

    # check that the coefficients are the same
    # zeroes should be equal with no regards to computation precision
    np.testing.assert_array_equal(
        keras_coefficients == 0, pytorch_coefficients == 0
    )
    np.testing.assert_array_almost_equal(
        keras_coefficients, pytorch_coefficients, decimal=decimal
    )

    # compare also the output masks
    # reoerdering indices for the output layer
    out_reordering_indices = (
        argsort_channels_last_to_channels_first_last_layer_maxpool()
    )
    # reorder on the first axis
    keras_output_masks = keras_path_selector.output_masks_[
        out_reordering_indices, :
    ]
    np.testing.assert_array_equal(
        keras_output_masks[:, reordering_indices],
        pytorch_path_selector.output_masks_,
    )
