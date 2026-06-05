import numpy as np
import pytest
from qsimov.paths.maxpooling import (
    get_all_paths_maxpool_layer,
    select_paths_maxpool_layer,
)
from qsimov.paths.conv import (
    _map_channels_last_paths_to_channels_first,
    _side_shape_to_channels_first,
    _compute_conv_output_shape,
)
import qsimov.paths.paths as paths_


@pytest.mark.parametrize("data_format", ["channels_last", "channels_first"])
@pytest.mark.parametrize(
    "input_shape, pool_size, strides, padding, paths_layer_true",
    [
        (
            (4, 4, 1),
            (2, 2),
            (2, 2),
            "valid",
            np.array(
                [
                    [1, 1],
                    [2, 1],
                    [3, 2],
                    [4, 2],
                    [5, 1],
                    [6, 1],
                    [7, 2],
                    [8, 2],
                    [9, 3],
                    [10, 3],
                    [11, 4],
                    [12, 4],
                    [13, 3],
                    [14, 3],
                    [15, 4],
                    [16, 4],
                ]
            ),
        ),
        (
            (2, 2, 2),
            (2, 2),
            (2, 2),
            "valid",
            np.array(
                [
                    [1, 1],
                    [2, 2],
                    [3, 1],
                    [4, 2],
                    [5, 1],
                    [6, 2],
                    [7, 1],
                    [8, 2],
                ]
            ),
        ),
        (
            (4, 4, 1),
            (2, 2),
            (1, 1),
            "valid",
            np.array(
                [
                    [1, 1],
                    [2, 1],
                    [2, 2],
                    [3, 2],
                    [3, 3],
                    [4, 3],
                    [5, 1],
                    [5, 4],
                    [6, 1],
                    [6, 2],
                    [6, 4],
                    [6, 5],
                    [7, 2],
                    [7, 3],
                    [7, 5],
                    [7, 6],
                    [8, 3],
                    [8, 6],
                    [9, 4],
                    [9, 7],
                    [10, 4],
                    [10, 5],
                    [10, 7],
                    [10, 8],
                    [11, 5],
                    [11, 6],
                    [11, 8],
                    [11, 9],
                    [12, 6],
                    [12, 9],
                    [13, 7],
                    [14, 7],
                    [14, 8],
                    [15, 8],
                    [15, 9],
                    [16, 9],
                ]
            ),
        ),
        (
            (3, 4, 1),
            (2, 2),
            (1, 1),
            "valid",
            np.array(
                [
                    [1, 1],
                    [2, 1],
                    [2, 2],
                    [3, 2],
                    [3, 3],
                    [4, 3],
                    [5, 1],
                    [5, 4],
                    [6, 1],
                    [6, 2],
                    [6, 4],
                    [6, 5],
                    [7, 2],
                    [7, 3],
                    [7, 5],
                    [7, 6],
                    [8, 3],
                    [8, 6],
                    [9, 4],
                    [10, 4],
                    [10, 5],
                    [11, 5],
                    [11, 6],
                    [12, 6],
                ]
            ),
        ),
        (
            (4, 4, 1),
            (3, 3),
            None,
            "same",
            np.array(
                [
                    [1, 1],
                    [2, 1],
                    [3, 2],
                    [4, 2],
                    [5, 1],
                    [6, 1],
                    [7, 2],
                    [8, 2],
                    [9, 3],
                    [10, 3],
                    [11, 4],
                    [12, 4],
                    [13, 3],
                    [14, 3],
                    [15, 4],
                    [16, 4],
                ]
            ),
        ),
    ],
)
def test_get_all_paths_maxpool(
    paths_layer_true, input_shape, pool_size, strides, padding, data_format
):
    input_shape = np.array(input_shape)

    # adapt to channels first testcase
    if data_format == "channels_first":
        output_shape = _compute_conv_output_shape(
            input_shape[:-1],
            input_shape[-1],
            pool_size,
            strides or pool_size,
            padding,
            "channels_last",
        )
        paths_layer_true = _map_channels_last_paths_to_channels_first(
            input_shape, output_shape, paths_layer_true
        )
        input_shape = _side_shape_to_channels_first(input_shape)

    # get paths between layers
    paths_layer = get_all_paths_maxpool_layer(
        input_shape, pool_size, strides, padding, data_format
    )

    np.testing.assert_equal(
        paths_.sort_paths(paths_layer), paths_.sort_paths(paths_layer_true)
    )


@pytest.mark.parametrize(
    "input, pool_size, strides, padding, active_paths_true",
    [
        (
            np.array(range(8)).reshape(2, 2, 2),
            (2, 2),
            (2, 2),
            "valid",
            np.array([[7, 1], [8, 2]]),
        ),
        (
            np.array(range(8)).reshape(2, 2, 2),
            (2, 2),
            None,
            "valid",
            np.array([[7, 1], [8, 2]]),
        ),
        (
            np.array(range(8)).reshape(2, 2, 2),
            2,
            None,
            "valid",
            np.array([[7, 1], [8, 2]]),
        ),
        (
            # input shape is (2, 4, 1)
            np.array([[[-1], [5], [8], [9]], [[0], [3], [1], [-1]]]),
            (2, 2),
            (1, 1),
            "valid",
            np.array([[2, 1], [3, 2], [4, 3]], dtype="int32"),
        ),
        (
            np.array([[1, 2], [2, 1]]).reshape(1, 2, 2),
            (1, 2),
            None,
            "valid",
            np.array([[3, 1], [2, 2]], dtype="int32"),
        ),
        (
            np.array([[1, 2], [2, 1]]).reshape(1, 2, 2),
            (1, 1),
            None,
            "valid",
            np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
        ),
        (
            np.array(range(16)).reshape(4, 4, 1),
            (3, 3),
            2,
            "valid",
            np.array([[11, 1]]),
        ),
        (
            np.ones((3, 3, 2)),
            (2, 2),
            2,
            "valid",
            np.array([[1, 1], [2, 2]]),
        ),
        (
            np.array(
                [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]
            ).reshape(4, 4, 1),
            (2, 2),
            2,
            "valid",
            np.array([[5, 1], [7, 2], [13, 3], [15, 4]]),
        ),
    ],
)
def test_select_paths_maxpool(
    active_paths_true, input, pool_size, strides, padding
):
    # connection pattern of the layer
    all_paths_layer = get_all_paths_maxpool_layer(
        input.shape, pool_size, strides, padding
    )
    # flat input for each sample (only 1 sample in this case), adding 1
    # to simulate a bias. Even if maxpooling doesnt work with a bias, this is
    # done to provide a generalized API to path selection
    flat_inputs_with_bias = [np.concatenate(([1], np.ravel(input)))]

    # make path selection mask and retrieve associated paths
    paths_layer = all_paths_layer[
        select_paths_maxpool_layer(
            np.asfarray(flat_inputs_with_bias), all_paths_layer
        )[0]
    ]

    # compare expected paths
    np.testing.assert_equal(
        paths_.sort_paths(paths_layer), paths_.sort_paths(active_paths_true)
    )
