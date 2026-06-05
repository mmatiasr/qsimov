import numpy as np
import pytest
import qsimov.paths.paths as paths_
import qsimov.paths.dense as dense
import tests.nn_mocks_keras as nn_mocks


@pytest.fixture(name="list_weights")
def make_list_weights():
    return nn_mocks.make_list_weights_3221()


@pytest.fixture(name="list_biases")
def make_list_biases():
    return nn_mocks.make_list_biases_3221()


@pytest.mark.parametrize(
    "paths_layer_true, layer_idx",
    [
        (
            paths_.sort_paths(
                np.array(
                    [
                        [0, 1],
                        [1, 1],
                        [2, 1],
                        [3, 1],
                        [0, 2],
                        [1, 2],
                        [2, 2],
                        [3, 2],
                    ]
                )
            ),
            0,
        ),
        (
            paths_.sort_paths(
                np.array([[0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])
            ),
            1,
        ),
        (paths_.sort_paths(np.array([[0, 1], [1, 1], [2, 1]])), 2),
    ],
)
def test_get_all_paths_dense_layer(
    paths_layer_true, layer_idx, list_weights, list_biases
):
    # get paths between layers
    paths_layer = paths_.sort_paths(
        dense.get_all_paths_dense_layer(
            list_weights[layer_idx], list_biases[layer_idx]
        )
    )
    np.testing.assert_equal(paths_layer, paths_layer_true)


def test_get_all_paths_dense_layer_no_paths():
    # get paths between layers
    paths_layer = paths_.sort_paths(
        dense.get_all_paths_dense_layer(np.zeros((3, 3)), np.zeros((3,)))
    )
    np.testing.assert_equal(paths_layer, np.empty((0, 2)))


@pytest.mark.parametrize(
    "input, paths_layer_true, layer_idx",
    [
        # sample 1,2,3
        (
            np.array([1, 2, 3]),
            paths_.sort_paths(
                np.array(
                    [
                        [0, 1],
                        [0, 2],
                        [1, 1],
                        [1, 2],
                        [2, 1],
                        [2, 2],
                        [3, 1],
                        [3, 2],
                    ]
                )
            ),
            0,
        ),
        (
            np.array([1, 0]),
            paths_.sort_paths(np.array([[0, 1], [0, 2], [1, 1], [1, 2]])),
            1,
        ),
        (
            np.array([3, 0]),
            paths_.sort_paths(np.array([[0, 1], [1, 1]])),
            2,
        ),
        # sample 0,3,4
        (
            np.array([0, 3, 4]),
            paths_.sort_paths(
                np.array([[0, 1], [0, 2], [2, 1], [2, 2], [3, 1], [3, 2]])
            ),
            0,
        ),
        (
            np.array([0, 0]),
            paths_.sort_paths(np.array([[0, 1], [0, 2]])),
            1,
        ),
        (
            np.array([1, 1]),
            paths_.sort_paths(np.array([[0, 1], [1, 1], [2, 1]])),
            2,
        ),
        # sample -2,-2,-2
        (
            np.array([-2, -2, -2]),
            paths_.sort_paths(
                np.array(
                    [
                        [0, 1],
                        [0, 2],
                        [1, 1],
                        [1, 2],
                        [2, 1],
                        [2, 2],
                        [3, 1],
                        [3, 2],
                    ]
                )
            ),
            0,
        ),
        (
            np.array([0, 7]),
            paths_.sort_paths(np.array([[0, 1], [0, 2], [2, 2]])),
            1,
        ),
        (
            np.array([1, 0]),
            paths_.sort_paths(np.array([[0, 1], [1, 1]])),
            2,
        ),
    ],
)
def test_select_paths_dense_layer(
    list_weights,
    list_biases,
    input,
    paths_layer_true,
    layer_idx,
):
    # get connection pattern for the layer
    all_paths_layer = dense.get_all_paths_dense_layer(
        list_weights[layer_idx], list_biases[layer_idx]
    )

    # add bias neuron to the layer input
    input_with_bias = np.array([np.concatenate(([1], input))])

    # make path selection mask and use it to index the connection pattern
    paths_layer = all_paths_layer[
        paths_.non_zero_input_select_paths(input_with_bias, all_paths_layer)[0]
    ]

    # sort paths
    paths_layer = paths_.sort_paths(paths_layer)

    np.testing.assert_array_equal(paths_layer, paths_layer_true)


def test_select_paths_dense_layer_no_paths(list_weights):
    # if inputs are 0 and biases are 0 we have no paths
    all_paths_layer = dense.get_all_paths_dense_layer(
        list_weights[0], np.zeros((2,))
    )

    input_with_bias = np.asfarray([[1, 0, 0, 0]])

    paths_layer = all_paths_layer[
        paths_.non_zero_input_select_paths(input_with_bias, all_paths_layer)[0]
    ]
    np.testing.assert_equal(paths_layer, np.empty((0, 2)))
