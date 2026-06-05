import pytest
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
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


@pytest.fixture(name="weights_323")
def make_weights_323():
    # weights for kernel size 3, 2 input channels, 3 output channels
    weights = np.array(
        [
            # col 1
            [  # input channel 1
                [
                    1,  # output channel 1
                    2,  # output channel 2
                    3,  # output channel 3
                ],
                # input channel 2
                [4, 5, 6],
            ],
            # col 2
            [[-3, -2, -1], [0, 1, 2]],
            # col 3
            [[-1, 1, 1], [-2, -3, -4]],
        ]
    )
    return weights


@pytest.fixture(name="biases_3")
def make_biases_3():
    return np.array([2, 1, 0])


# ------- TEST UTIL FUNCTIONS


@pytest.mark.parametrize(
    "input_shape, output_shape, paths_channel_last, solution",
    [
        (
            (5, 2),
            (4, 3),
            np.array(
                [[0, 1], [1, 1], [1, 2], [4, 12], [10, 7]],
            ),
            np.array(
                [
                    [0, 1],
                    [1, 1],
                    [1, 5],
                    [7, 12],
                    [10, 3],
                ]
            ),
        ),
    ],
)
def test__map_channels_last_paths_to_channels_first(
    input_shape, output_shape, paths_channel_last, solution
):
    paths_channel_first = _map_channels_last_paths_to_channels_first(
        np.array(input_shape), np.array(output_shape), paths_channel_last
    )
    np.testing.assert_array_equal(paths_channel_first, solution)


# ------- OUTPUT SHAPE COMPUTATION


@pytest.mark.parametrize(
    "input_shape, filters, kernel_size, strides, padding",
    [
        ((5, 1), 3, (3,), 1, "same"),
        ((6, 1), 2, (2,), 2, "valid"),
    ],
)
def test_compute_conv1d_output_shape_keras(
    input_shape, filters, kernel_size, strides, padding
):
    x = tf.random.normal((1, *input_shape))
    layer = tf.keras.layers.Conv1D(
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
def test_get_all_paths_conv1d_layer(weights_323, biases_3, data_format):
    input_shape = np.array((5, 2))

    # Correct solution adjacencies (output neuron to input neurons)
    paths_out_to_in = {
        # [bias] + [input channel 1 neurons] + [input channel 2 neurons]
        # removed connections (zero weight) marked with None values
        # output channel 1
        1: [0] + [1, 3, 5] + [2, None, 6],
        4: [0] + [3, 5, 7] + [4, None, 8],
        7: [0] + [5, 7, 9] + [6, None, 10],
        # output channel 2
        2: [0] + [1, 3, 5] + [2, 4, 6],
        5: [0] + [3, 5, 7] + [4, 6, 8],
        8: [0] + [5, 7, 9] + [6, 8, 10],
        # output channel 3
        3: [None] + [1, 3, 5] + [2, 4, 6],
        6: [None] + [3, 5, 7] + [4, 6, 8],
        9: [None] + [5, 7, 9] + [6, 8, 10],
    }
    paths_true = out_to_in_adjacency_list_to_edge_list(paths_out_to_in)

    # adjust for channels first testcase
    if data_format == "channels_first":
        output_shape = _compute_conv_output_shape(
            input_shape[:-1],
            weights_323.shape[-1],
            weights_323.shape[:-2],
            1,
            "valid",
            "channels_last",
        )
        paths_true = _map_channels_last_paths_to_channels_first(
            input_shape, output_shape, paths_true
        )
        input_shape = _side_shape_to_channels_first(input_shape)
        weights_323 = weights_to_channels_first(weights_323)

    # sort paths
    paths_true = sort_paths(paths_true)

    # Computed solution
    paths_computed = get_all_paths_conv_layer(
        input_shape, weights_323, biases_3, 1, "valid", 1, data_format
    )
    paths_computed = sort_paths(paths_computed)

    np.testing.assert_array_equal(paths_computed, paths_true)


@pytest.mark.parametrize("data_format", ["channels_last", "channels_first"])
def test_get_all_paths_conv1d_layer_padding_same(
    weights_323, biases_3, data_format
):
    input_shape = np.array((5, 2))

    paths_out_to_in = {
        # [bias] + [input channel 1 neurons] + [input channel 2 neurons]
        # removed connections (zero weight or padding) marked with None values
        # output channel 1
        1: [0] + [None, 1, 3] + [None, None, 4],
        4: [0] + [1, 3, 5] + [2, None, 6],
        7: [0] + [3, 5, 7] + [4, None, 8],
        10: [0] + [5, 7, 9] + [6, None, 10],
        13: [0] + [7, 9, None] + [8, None, None],
        # output channel 2
        2: [0] + [None, 1, 3] + [None, 2, 4],
        5: [0] + [1, 3, 5] + [2, 4, 6],
        8: [0] + [3, 5, 7] + [4, 6, 8],
        11: [0] + [5, 7, 9] + [6, 8, 10],
        14: [0] + [7, 9, None] + [8, 10, None],
        # output channel 3
        3: [None] + [None, 1, 3] + [None, 2, 4],
        6: [None] + [1, 3, 5] + [2, 4, 6],
        9: [None] + [3, 5, 7] + [4, 6, 8],
        12: [None] + [5, 7, 9] + [6, 8, 10],
        15: [None] + [7, 9, None] + [8, 10, None],
    }

    paths_true = out_to_in_adjacency_list_to_edge_list(paths_out_to_in)

    # adjust for channels first testcase
    if data_format == "channels_first":
        output_shape = _compute_conv_output_shape(
            input_shape[:-1],
            weights_323.shape[-1],
            weights_323.shape[:-2],
            1,
            "same",
            "channels_last",
        )
        paths_true = _map_channels_last_paths_to_channels_first(
            input_shape, output_shape, paths_true
        )
        input_shape = _side_shape_to_channels_first(input_shape)
        weights_323 = weights_to_channels_first(weights_323)

    # sort paths
    paths_true = sort_paths(paths_true)

    # Computed solution
    paths_computed = get_all_paths_conv_layer(
        input_shape, weights_323, biases_3, 1, "same", 1, data_format
    )
    paths_computed = sort_paths(paths_computed)

    np.testing.assert_array_equal(paths_computed, paths_true)


@pytest.mark.parametrize("data_format", ["channels_last", "channels_first"])
def test_get_all_paths_conv1d_layer_stride_neq_1(
    weights_323, biases_3, data_format
):
    input_shape = np.array((5, 2))

    paths_out_to_in = {
        # [bias] + [input channel 1 neurons] + [input channel 2 neurons]
        # removed connections (zero weight or padding) marked with None values
        # output channel 1
        1: [0] + [None, 1, 3] + [None, None, 4],
        4: [0] + [3, 5, 7] + [4, None, 8],
        7: [0] + [7, 9, None] + [8, None, None],
        # output channel 2
        2: [0] + [None, 1, 3] + [None, 2, 4],
        5: [0] + [3, 5, 7] + [4, 6, 8],
        8: [0] + [7, 9, None] + [8, 10, None],
        # output channel 3
        3: [None] + [None, 1, 3] + [None, 2, 4],
        6: [None] + [3, 5, 7] + [4, 6, 8],
        9: [None] + [7, 9, None] + [8, 10, None],
    }

    paths_true = out_to_in_adjacency_list_to_edge_list(paths_out_to_in)

    # adjust for channels first testcase
    if data_format == "channels_first":
        output_shape = _compute_conv_output_shape(
            input_shape[:-1],
            weights_323.shape[-1],
            weights_323.shape[:-2],
            2,  # stride 2
            "same",
            "channels_last",
        )
        paths_true = _map_channels_last_paths_to_channels_first(
            input_shape, output_shape, paths_true
        )
        input_shape = _side_shape_to_channels_first(input_shape)
        weights_323 = weights_to_channels_first(weights_323)

    # sort paths
    paths_true = sort_paths(paths_true)

    # Computed solution
    paths_computed = get_all_paths_conv_layer(
        input_shape, weights_323, biases_3, 2, "same", 1, data_format
    )
    paths_computed = sort_paths(paths_computed)

    np.testing.assert_array_equal(paths_computed, paths_true)


@pytest.mark.parametrize("data_format", ["channels_last", "channels_first"])
@pytest.mark.parametrize(
    "input_shape, weight_shape, strides, padding, groups",
    [
        ((8, 4), (2, 2, 8), (2,), "valid", 2),
        ((4, 12), (2, 4, 9), (1,), "same", 3),
    ],
)
def test_get_all_paths_conv1d_groups_neq_1_by_map_to_dense(
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
        output = F.conv1d(
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
        layer = tf.keras.layers.Conv1D(
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
