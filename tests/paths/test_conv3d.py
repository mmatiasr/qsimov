import pytest
import numpy as np
import tensorflow as tf
import torch.nn.functional as F
import torch
from qsimov.paths.conv import (
    _compute_conv_output_shape,
    get_all_paths_conv_layer,
    _map_channels_last_paths_to_channels_first,
    _side_shape_to_channels_first,
)
from qsimov.paths.paths import sort_paths
from tests.paths.utils_conv_tests import (
    out_to_in_adjacency_list_to_edge_list,
    weights_to_channels_first,
)


# ------- HELPERS


@pytest.fixture(name="weights_22212")
def make_weights_22212():
    # weights for kernel size (2, 2, 2) 1 input channel and 2 output channels
    weights = np.random.uniform(1, 2, (2, 2, 2, 1, 2))
    return weights


@pytest.fixture(name="biases_2")
def make_biases_2():
    return np.array([0, 1])


# ------- OUTPUT SHAPE COMPUTATION


@pytest.mark.parametrize(
    "input_shape, filters, kernel_size, strides, padding",
    [
        ((1, 1, 1, 3), 1, (1, 1, 1), 1, "valid"),
        ((8, 8, 8, 4), 4, (2, 2, 2), (2, 2, 2), "same"),
        ((8, 16, 24, 2), 4, (3, 4, 2), (1, 2, 2), "same"),
    ],
)
def test_compute_conv3d_output_shape_keras(
    input_shape, filters, kernel_size, strides, padding
):
    x = tf.random.normal((1, *input_shape))
    layer = tf.keras.layers.Conv3D(
        filters=filters,
        kernel_size=kernel_size,
        activation="relu",
        padding=padding,
        strides=strides,
        input_shape=input_shape,
    )
    solution = layer(x).shape[1:]  # remove batch dimension

    output_shape = _compute_conv_output_shape(
        input_shape[:-1],
        filters,
        kernel_size,
        strides,
        padding,
        "channels_last",
    )

    np.testing.assert_array_equal(output_shape, solution)


# ------- PATH COMPUTATION


@pytest.mark.parametrize("data_format", ["channels_last", "channels_first"])
def test_get_all_paths_conv3d_layer(weights_22212, biases_2, data_format):
    input_shape = np.array((2, 2, 3, 1))

    # Correct solution adjacencies (output neuron to input neurons)
    paths_out_to_in = {
        # output channel 1
        1: [1, 2, 4, 5, 7, 8, 10, 11],
        3: [2, 3, 5, 6, 8, 9, 11, 12],
        # output channel 2
        2: [1, 2, 4, 5, 7, 8, 10, 11, 0],
        4: [2, 3, 5, 6, 8, 9, 11, 12, 0],
    }
    paths_true = out_to_in_adjacency_list_to_edge_list(paths_out_to_in)

    # adjust for channels first testcase
    if data_format == "channels_first":
        output_shape = _compute_conv_output_shape(
            input_shape[:-1],
            weights_22212.shape[-1],
            weights_22212.shape[:-2],
            1,
            "valid",
            "channels_last",
        )
        paths_true = _map_channels_last_paths_to_channels_first(
            input_shape, output_shape, paths_true
        )
        input_shape = _side_shape_to_channels_first(input_shape)
        weights_22212 = weights_to_channels_first(weights_22212)

    # sort paths
    paths_true = sort_paths(paths_true)

    # Computed solution
    paths_computed = get_all_paths_conv_layer(
        input_shape, weights_22212, biases_2, 1, "valid", 1, data_format
    )
    paths_computed = sort_paths(paths_computed)

    np.testing.assert_array_equal(paths_computed, paths_true)


@pytest.mark.parametrize("data_format", ["channels_last", "channels_first"])
@pytest.mark.parametrize(
    "input_shape, weight_shape, strides, padding",
    [
        ((1, 1, 1, 3), (1, 1, 1, 3, 1), 1, "valid"),
        ((8, 8, 8, 4), (2, 2, 2, 4, 4), (2, 2, 2), "valid"),
        ((8, 16, 4, 2), (3, 4, 2, 2, 4), (1, 1, 1), "same"),
    ],
)
def test_get_all_paths_conv3d_by_map_to_dense(
    input_shape, weight_shape, strides, padding, data_format
):
    # number of output channels
    filters = weight_shape[-1]
    kernel_shape = weight_shape[:-2]

    # unit weights according to weight shape
    weights = np.ones(weight_shape).ravel()
    # set one random weight to zero
    weights[np.random.randint(0, weights.size)] = 0
    weights = weights.reshape(weight_shape)
    biases = np.ones(filters)
    # set one random bias to zero
    biases[np.random.randint(0, biases.size)] = 0

    # adjust for channels first testcase
    if data_format == "channels_first":
        input_shape = _side_shape_to_channels_first(input_shape)
        weights = weights_to_channels_first(weights)

        # random input for layer float 32
        input_to_layer = torch.rand((1, *input_shape)).to(torch.float32)

        # compute output using torch (it uses channels first)
        output = F.conv3d(
            input_to_layer,
            weight=torch.tensor(weights).to(torch.float32),
            bias=torch.tensor(biases).to(torch.float32),
            stride=strides,
            padding=padding,
        )
        output = output.numpy().ravel()
    else:
        # random input for layer
        input_to_layer = tf.random.normal((1, *input_shape))

        # create conv3d layer using keras (it uses channels last)
        layer = tf.keras.layers.Conv3D(
            filters=filters,
            kernel_size=kernel_shape,
            activation="linear",
            padding=padding,
            strides=strides,
            input_shape=input_shape,
            kernel_initializer=tf.keras.initializers.Constant(weights),
            bias_initializer=tf.keras.initializers.Constant(biases),
        )
        output = layer(input_to_layer).numpy().ravel()

    # compute paths
    paths_computed = get_all_paths_conv_layer(
        input_shape, weights, biases, strides, padding, 1, data_format
    )
    paths_computed = sort_paths(paths_computed)

    # use paths to compute dense equivalent
    weights_by_paths = np.zeros((input_to_layer.numpy().size, output.size))
    biases_by_paths = np.zeros(output.size)
    for path in paths_computed:
        if path[0] == 0:
            biases_by_paths[path[1] - 1] += 1
        else:
            weights_by_paths[path[0] - 1, path[1] - 1] += 1
    output_computed = (
        input_to_layer.numpy().ravel() @ weights_by_paths + biases_by_paths
    )

    np.testing.assert_array_almost_equal(output, output_computed, decimal=5)


@pytest.mark.parametrize("data_format", ["channels_last", "channels_first"])
@pytest.mark.parametrize(
    "input_shape, weight_shape, strides, padding, groups",
    [
        ((7, 8, 8, 4), (3, 2, 2, 2, 8), (2, 2, 2), "valid", 2),
        ((5, 16, 4, 12), (2, 4, 2, 4, 9), (1, 1, 1), "same", 3),
    ],
)
def test_get_all_paths_conv3d_groups_neq_1_by_map_to_dense(
    input_shape, weight_shape, strides, padding, groups, data_format
):
    # number of output channels
    filters = weight_shape[-1]
    kernel_shape = weight_shape[:-2]

    # unit weights according to weight shape
    weights = np.ones(weight_shape).ravel()
    # set one random weight to zero
    weights[np.random.randint(0, weights.size)] = 0
    weights = weights.reshape(weight_shape)
    biases = np.ones(filters)
    # set one random bias to zero
    biases[np.random.randint(0, biases.size)] = 0

    # adjust for channels first testcase
    if data_format == "channels_first":
        input_shape = _side_shape_to_channels_first(input_shape)
        weights = weights_to_channels_first(weights)

        # random input for layer float 32
        input_to_layer = torch.rand((1, *input_shape)).to(torch.float32)

        # compute output using torch (it uses channels first)
        output = F.conv3d(
            input_to_layer,
            weight=torch.tensor(weights).to(torch.float32),
            bias=torch.tensor(biases).to(torch.float32),
            stride=strides,
            padding=padding,
            groups=groups,
        )
        output = output.numpy().ravel()
    else:
        # random input for layer
        input_to_layer = tf.random.normal((1, *input_shape))

        # create conv3d layer using keras (it uses channels last)
        layer = tf.keras.layers.Conv3D(
            filters=filters,
            kernel_size=kernel_shape,
            activation="linear",
            padding=padding,
            strides=strides,
            input_shape=input_shape,
            kernel_initializer=tf.keras.initializers.Constant(weights),
            bias_initializer=tf.keras.initializers.Constant(biases),
            groups=groups,
        )
        output = layer(input_to_layer).numpy().ravel()

    # compute paths
    paths_computed = get_all_paths_conv_layer(
        input_shape, weights, biases, strides, padding, groups, data_format
    )
    paths_computed = sort_paths(paths_computed)

    # use paths to compute dense equivalent
    weights_by_paths = np.zeros((input_to_layer.numpy().size, output.size))
    biases_by_paths = np.zeros(output.size)
    for path in paths_computed:
        if path[0] == 0:
            biases_by_paths[path[1] - 1] += 1
        else:
            weights_by_paths[path[0] - 1, path[1] - 1] += 1

    # compute dense equivalent matrix multiplication
    output_computed = (
        input_to_layer.numpy().ravel() @ weights_by_paths + biases_by_paths
    )

    np.testing.assert_array_almost_equal(output, output_computed, decimal=5)
