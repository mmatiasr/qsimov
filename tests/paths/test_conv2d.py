import pytest
import numpy as np
import tensorflow as tf
from qsimov.paths.conv import (
    _compute_conv_output_shape,
    get_all_paths_conv_layer,
    _map_channels_last_paths_to_channels_first,
    _side_shape_to_channels_first,
)
from qsimov.paths.paths import sort_paths, non_zero_input_select_paths
from tests.paths.utils_conv_tests import (
    out_to_in_adjacency_list_to_edge_list,
    weights_to_channels_first,
)
import torch
import torch.nn.functional as F


# ------- HELPERS


@pytest.fixture(name="weights_3312")
def make_weights_3312():
    # weights for 3x3 kernel with 1 channel input, obtaining 2 filter outputs
    weights = np.array(
        [
            [
                [0, 1, 2],
                [2, 2, 0],
                [0, 1, 2],
            ],
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
        ]
    ).reshape((2, 3, 3, 1))

    # use channels last
    weights = np.moveaxis(weights, 0, 3)

    return weights


@pytest.fixture(name="biases_2")
def make_biases_2():
    return np.array([0, 1])


# ------- TEST UTIL FUNCTIONS


@pytest.mark.parametrize(
    "input_shape, output_shape, paths_channel_last, solution",
    [
        (
            (3, 3, 2),
            (2, 2, 3),
            np.array(
                [
                    [0, 1],
                    [0, 6],
                    [1, 2],
                    [1, 7],
                    [1, 9],
                    [5, 12],
                    [6, 1],
                    [18, 4],
                ]
            ),
            np.array(
                [
                    [0, 1],
                    [0, 10],
                    [1, 5],
                    [1, 3],
                    [1, 11],
                    [3, 12],
                    [12, 1],
                    [18, 2],
                ]
            ),
        )
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
    "image_shape, filters, kernel_size, strides, padding, solution",
    [
        ((5, 5, 1), 5, 3, (1, 1), "valid", (3, 3, 5)),
        ((7, 6, 4), 2, (1, 2), (2, 3), "valid", (4, 2, 2)),
        ((7, 6, 4), 6, (2, 3), (3, 3), "valid", (2, 2, 6)),
    ],
)
def test_compute_conv2d_output_shape(
    image_shape, filters, kernel_size, strides, padding, solution
):
    # check against channels last
    output_shape = _compute_conv_output_shape(
        image_shape[:-1],
        filters,
        kernel_size,
        strides,
        padding,
        data_format="channels_last",
    )
    np.testing.assert_array_equal(output_shape, solution)

    # check against channels first
    output_shape = _compute_conv_output_shape(
        image_shape[:-1],
        filters,
        kernel_size,
        strides,
        padding,
        data_format="channels_first",
    )
    solution = (solution[2], solution[0], solution[1])
    np.testing.assert_array_equal(output_shape, solution)


@pytest.mark.parametrize(
    "image_shape, filters, kernel_size, strides, padding",
    [
        ((5, 5, 1), 5, 3, (1, 1), "same"),
        ((7, 6, 4), 2, (1, 2), (2, 3), "same"),
        ((7, 6, 4), 6, (2, 3), (3, 3), "same"),
    ],
)
def test_compute_conv2d_output_shape_keras(
    image_shape, filters, kernel_size, strides, padding
):
    x = tf.random.normal((1, *image_shape))
    layer = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        activation="relu",
        padding=padding,
        strides=strides,
        input_shape=image_shape,
    )
    solution = layer(x).shape[1:]  # remove batch dimension

    output_shape = _compute_conv_output_shape(
        image_shape[:-1],
        filters,
        kernel_size,
        strides,
        padding,
        "channels_last",
    )

    np.testing.assert_array_equal(output_shape, solution)


# ------- PATH COMPUTATION


@pytest.mark.parametrize("data_format", ["channels_last", "channels_first"])
def test_get_all_paths_conv2d_layer(weights_3312, biases_2, data_format):
    input_shape = np.array([5, 5, 1], dtype=int)

    # second filter connections to each output
    connections_out_to_in = {
        2: np.array([0, 1, 2, 3, 6, 7, 8, 11, 12, 13]),
        4: np.array([0, 2, 3, 4, 7, 8, 9, 12, 13, 14]),
        6: np.array([0, 3, 4, 5, 8, 9, 10, 13, 14, 15]),
        8: np.array([0, 6, 7, 8, 11, 12, 13, 16, 17, 18]),
        10: np.array([0, 7, 8, 9, 12, 13, 14, 17, 18, 19]),
        12: np.array([0, 8, 9, 10, 13, 14, 15, 18, 19, 20]),
        14: np.array([0, 11, 12, 13, 16, 17, 18, 21, 22, 23]),
        16: np.array([0, 12, 13, 14, 17, 18, 19, 22, 23, 24]),
        18: np.array([0, 13, 14, 15, 18, 19, 20, 23, 24, 25]),
    }

    # the mask extracts input neurons with non zero weight or bias on filter 1
    filter_1_mask = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1], dtype=bool)

    # add connections to first filter
    for output_filter2, input_filter2 in list(connections_out_to_in.items()):
        connections_out_to_in[output_filter2 - 1] = np.array(input_filter2)[
            filter_1_mask
        ]

    # transform to edge list
    paths_true = out_to_in_adjacency_list_to_edge_list(connections_out_to_in)

    # adjust for channels first testcase
    if data_format == "channels_first":
        output_shape = _compute_conv_output_shape(
            input_shape[:-1],
            weights_3312.shape[-1],
            weights_3312.shape[:-2],
            (1, 1),
            "valid",
            "channels_last",
        )
        paths_true = _map_channels_last_paths_to_channels_first(
            input_shape, output_shape, paths_true
        )
        input_shape = _side_shape_to_channels_first(input_shape)
        weights_3312 = weights_to_channels_first(weights_3312)

    # compare to computation using function
    paths_computed = sort_paths(
        get_all_paths_conv_layer(
            input_shape,
            weights_3312,
            biases_2,
            strides=(1, 1),
            data_format=data_format,
        )
    )

    np.testing.assert_array_equal(
        paths_computed, sort_paths(np.array(paths_true))
    )


@pytest.mark.parametrize("data_format", ["channels_last", "channels_first"])
def test_get_all_paths_conv2d_layer_padding_same(
    weights_3312, biases_2, data_format
):
    input_shape = np.array([5, 5, 1], dtype=int)

    # second filter connections to each output (None is connection to padding)
    connections_out_to_in = {
        2: [None, None, None, None, 1, 2, None, 6, 7],
        4: [None, None, None, 1, 2, 3, 6, 7, 8],
        6: [None, None, None, 2, 3, 4, 7, 8, 9],
        8: [None, None, None, 3, 4, 5, 8, 9, 10],
        10: [None, None, None, 4, 5, None, 9, 10, None],
        #
        12: [None, 1, 2, None, 6, 7, None, 11, 12],
        14: [1, 2, 3, 6, 7, 8, 11, 12, 13],
        16: [2, 3, 4, 7, 8, 9, 12, 13, 14],
        18: [3, 4, 5, 8, 9, 10, 13, 14, 15],
        20: [4, 5, None, 9, 10, None, 14, 15, None],
        #
        22: [None, 6, 7, None, 11, 12, None, 16, 17],
        24: [6, 7, 8, 11, 12, 13, 16, 17, 18],
        26: [7, 8, 9, 12, 13, 14, 17, 18, 19],
        28: [8, 9, 10, 13, 14, 15, 18, 19, 20],
        30: [9, 10, None, 14, 15, None, 19, 20, None],
        #
        32: [None, 11, 12, None, 16, 17, None, 21, 22],
        34: [11, 12, 13, 16, 17, 18, 21, 22, 23],
        36: [12, 13, 14, 17, 18, 19, 22, 23, 24],
        38: [13, 14, 15, 18, 19, 20, 23, 24, 25],
        40: [14, 15, None, 19, 20, None, 24, 25, None],
        #
        42: [None, 16, 17, None, 21, 22, None, None, None],
        44: [16, 17, 18, 21, 22, 23, None, None, None],
        46: [17, 18, 19, 22, 23, 24, None, None, None],
        48: [18, 19, 20, 23, 24, 25, None, None, None],
        50: [19, 20, None, 24, 25, None, None, None, None],
    }

    # the mask extracts input neurons with non zero weight or bias on filter 1
    filter_1_mask = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1], dtype=bool)

    # add connections to first filter
    for output_filter2, input_filter2 in list(connections_out_to_in.items()):
        # add bias, swap None with -1
        input_filter2 = np.array(
            [0] + [neuron or -1 for neuron in input_filter2]
        )
        input_filter1 = input_filter2[filter_1_mask]

        # remove padding and add connections
        connections_out_to_in[output_filter2] = input_filter2[
            input_filter2 >= 0
        ]
        connections_out_to_in[output_filter2 - 1] = input_filter1[
            input_filter1 >= 0
        ]

    # transform to edge list
    paths_true = out_to_in_adjacency_list_to_edge_list(connections_out_to_in)

    # adjust for channels first testcase
    if data_format == "channels_first":
        output_shape = _compute_conv_output_shape(
            input_shape[:-1],
            weights_3312.shape[-1],
            weights_3312.shape[:-2],
            (1, 1),
            "same",
            "channels_last",
        )
        paths_true = _map_channels_last_paths_to_channels_first(
            input_shape, output_shape, paths_true
        )
        input_shape = _side_shape_to_channels_first(input_shape)
        weights_3312 = weights_to_channels_first(weights_3312)
    paths_true = sort_paths(np.array(paths_true))

    # compare to computation using function
    paths_computed = sort_paths(
        get_all_paths_conv_layer(
            input_shape,
            weights_3312,
            biases_2,
            strides=(1, 1),
            padding="same",
            data_format=data_format,
        )
    )

    np.testing.assert_array_equal(paths_computed, paths_true)


@pytest.mark.parametrize("data_format", ["channels_last", "channels_first"])
def test_get_all_paths_conv2d_layer_stride_neq_1(
    weights_3312, biases_2, data_format
):
    input_shape = np.array([5, 5, 1], dtype=int)

    # second filter connections to each output
    connections_out_to_in = {
        2: np.array([0, 1, 2, 3, 6, 7, 8, 11, 12, 13]),
        4: np.array([0, 3, 4, 5, 8, 9, 10, 13, 14, 15]),
        6: np.array([0, 11, 12, 13, 16, 17, 18, 21, 22, 23]),
        8: np.array([0, 13, 14, 15, 18, 19, 20, 23, 24, 25]),
    }

    # the mask extracts input neurons with non zero weight or bias on filter 1
    filter_1_mask = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1], dtype=bool)

    # add connections to first filter
    for output_filter2, input_filter2 in list(connections_out_to_in.items()):
        connections_out_to_in[output_filter2 - 1] = np.array(input_filter2)[
            filter_1_mask
        ]

    # transform to edge list
    paths_true = out_to_in_adjacency_list_to_edge_list(connections_out_to_in)

    # adjust for channels first testcase
    if data_format == "channels_first":
        output_shape = _compute_conv_output_shape(
            input_shape[:-1],
            weights_3312.shape[-1],
            weights_3312.shape[:-2],
            (2, 2),
            "valid",
            "channels_last",
        )
        paths_true = _map_channels_last_paths_to_channels_first(
            input_shape, output_shape, paths_true
        )
        input_shape = _side_shape_to_channels_first(input_shape)
        weights_3312 = weights_to_channels_first(weights_3312)
    paths_true = sort_paths(np.array(paths_true))

    # compare to computation using function
    paths_computed = sort_paths(
        get_all_paths_conv_layer(
            input_shape,
            weights_3312,
            biases_2,
            strides=(2, 2),
            data_format=data_format,
        )
    )

    np.testing.assert_array_equal(paths_computed, paths_true)


@pytest.mark.parametrize("data_format", ["channels_last", "channels_first"])
@pytest.mark.parametrize(
    "input_shape, weight_shape, strides, padding, groups",
    [
        ((8, 8, 4), (2, 2, 2, 8), (2, 2), "valid", 2),
        ((16, 4, 12), (4, 2, 4, 9), (1, 1), "same", 3),
    ],
)
def test_get_all_paths_conv2d_groups_neq_1_by_map_to_dense(
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
        output = F.conv2d(
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
        layer = tf.keras.layers.Conv2D(
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


@pytest.mark.parametrize(
    "input, input_shape, strides, padding, paths_layer_true",
    [
        (
            np.hstack((np.arange(1, 20, dtype="int"), np.zeros(6))).reshape(
                (1, 5, 5, 1)
            ),
            np.array([5, 5, 1], dtype=int),
            (2, 2),
            "valid",
            np.array(
                [
                    [input_neuron, output_neuron]
                    for input_neuron, output_neurons in {
                        0: [2, 4, 6, 8],
                        1: [2],
                        2: [1, 2],
                        3: [1, 2, 4],
                        4: [3, 4],
                        5: [3, 4],
                        6: [1, 2],
                        7: [1, 2],
                        8: [3, 2, 4],
                        9: [3, 4],
                        10: [4],
                        11: [2, 6],
                        12: [1, 5, 2, 6],
                        13: [1, 5, 2, 4, 6, 8],
                        14: [3, 7, 4, 8],
                        15: [3, 7, 4, 8],
                        16: [5, 6],
                        17: [5, 6],
                        18: [7, 6, 8],
                        19: [7, 8],
                    }.items()
                    for output_neuron in output_neurons
                ]
            ),
        ),
    ],
)
def test_select_paths_conv2d_layer(
    input,
    input_shape,
    weights_3312,
    biases_2,
    strides,
    padding,
    paths_layer_true,
):
    # connection pattern of the layer
    all_paths = get_all_paths_conv_layer(
        input_shape, weights_3312, biases_2, strides, padding
    )

    # flat inputs per each sample (only 1 sample) with 1 inserted for bias
    flat_inputs_with_bias = np.asfarray(
        [np.concatenate(([1], np.ravel(input)))]
    )

    # make path selection mask and use it to index full path list
    paths_layer = sort_paths(
        all_paths[
            non_zero_input_select_paths(flat_inputs_with_bias, all_paths)[0]
        ]
    )

    # select paths and get output for given input
    np.testing.assert_array_equal(paths_layer, sort_paths(paths_layer_true))
