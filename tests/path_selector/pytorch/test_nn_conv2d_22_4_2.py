import numpy as np
import pytest
from qsimov.pytorch_path_selector import PytorchPathSelector
from qsimov.path_selector import _flatten_and_insert_one
import qsimov.paths.paths as paths_
import qsimov.paths.combine as combine
from qsimov.paths.conv import _map_channels_last_paths_to_channels_first

import torch.nn as nn
import torch

from tests.paths.utils_conv_tests import (
    weights_to_channels_first,
    samples_to_channels_first,
)


@pytest.fixture(name="path_selector")
def make_path_selector():
    # create model
    conv_layer = nn.Conv2d(
        in_channels=2,
        out_channels=1,
        kernel_size=2,
        bias=False,
        padding="same",
    )
    relu_layer = nn.ReLU()
    dropout_layer = nn.Dropout(0.1)
    flatten_layer = nn.Flatten()
    dense_layer = nn.Linear(4, 2, bias=True)
    model = nn.Sequential(
        conv_layer, relu_layer, dropout_layer, flatten_layer, dense_layer
    )

    # "train" model (set weights and biases)
    weights1 = np.array([[[[1], [1]], [[1], [1]]], [[[0], [0]], [[-1], [-1]]]])
    weights4 = np.array([[1, -1], [-1, 1], [1, -1], [-1, 1]])
    biases4 = np.array([3, 3])

    # transform to channels first in the conv layer
    weights1 = weights_to_channels_first(weights1)

    # set weights and biases
    conv_layer.weight.data = torch.from_numpy(np.asfarray(weights1))
    dense_layer.weight.data = torch.from_numpy(np.asfarray(weights4.T))
    dense_layer.bias.data = torch.from_numpy(np.asfarray(biases4))

    # use for path selection
    return PytorchPathSelector(
        neural_network=model, input_shape=(2, 2, 2), initial_layer=0
    )


@pytest.fixture(name="X")
def make_samples():
    sample1 = np.array([[[1, 0], [2, 3]], [[0, -1], [1, -2]]])
    return samples_to_channels_first(np.array([sample1]).astype(float))


def test__compute_all_paths(path_selector):
    # check full path list
    all_paths = path_selector._all_paths

    # all paths between layer 0 and 1
    all_paths_layer1_true = np.vstack(
        [
            # insert(input neurons, _, output_neuron,_)
            np.insert([[1], [2], [3], [4], [7], [8]], 1, 1, axis=1),
            np.insert([[3], [4]], 1, 2, axis=1),
            np.insert([[5], [6], [7], [8]], 1, 3, axis=1),
            np.insert([[7], [8]], 1, 4, axis=1),
        ]
    ).astype(np.int32)

    # convert paths to channels first
    all_paths_layer1_true = _map_channels_last_paths_to_channels_first(
        input_shape=(2, 2, 2),
        output_shape=(2, 2, 1),
        paths=all_paths_layer1_true,
    )

    # all paths between layer 1 and 2
    all_paths_layer4_true = np.array(
        [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]]
    ).astype(np.int32)

    # convert paths to channels first (only in the input side, as the output
    # is dense and does not have channels)
    all_paths_layer4_true = _map_channels_last_paths_to_channels_first(
        input_shape=(2, 2, 1),
        output_shape=None,
        paths=all_paths_layer4_true,
    )

    # compute all paths
    all_paths_true = combine.combine_paths(
        [all_paths_layer1_true, all_paths_layer4_true]
    )

    np.testing.assert_array_equal(all_paths, paths_.sort_paths(all_paths_true))

    # no zeroes on either output
    np.testing.assert_equal(
        sum(path_selector.output_masks_[0]), len(all_paths)
    )
    np.testing.assert_equal(
        sum(path_selector.output_masks_[1]), len(all_paths)
    )


def test_samples_to_coefficients(X, path_selector):
    coefficients = path_selector.samples_to_coefficients(X)

    # selected paths between layer 0 and 1
    selected_paths_layer1_true = np.vstack(
        [
            # insert(input neurons, _, output_neuron,_)
            np.insert([[1], [3], [4], [7], [8]], 1, 1, axis=1),
            np.insert([[3], [4]], 1, 2, axis=1),
            np.insert([[6], [7], [8]], 1, 3, axis=1),
            np.insert([[7], [8]], 1, 4, axis=1),
        ]
    ).astype(np.int32)

    # convert paths to channels first
    selected_paths_layer1_true = _map_channels_last_paths_to_channels_first(
        input_shape=(2, 2, 2),
        output_shape=(2, 2, 1),
        paths=selected_paths_layer1_true,
    )

    # all paths between layer 1 and 2
    selected_paths_layer4_true = np.array([[0, 1], [1, 1], [2, 1]]).astype(
        np.int32
    )

    # convert paths to channels first (only in the input side, as the output
    # is dense and does not have channels)
    selected_paths_layer4_true = _map_channels_last_paths_to_channels_first(
        input_shape=(2, 2, 1),
        output_shape=None,
        paths=selected_paths_layer4_true,
    )

    # filter by output on last layer and combine selected paths
    selected_paths_true = combine.combine_paths(
        [selected_paths_layer1_true, selected_paths_layer4_true]
    )

    # transform to bool mask
    select_mask = paths_.paths_subset_of(
        path_selector._all_paths, selected_paths_true
    )

    # build coefficients
    coefficients_true = paths_.retrieve_coefficients(
        [select_mask],
        path_selector._all_paths[:, 0],
        _flatten_and_insert_one(X),
    )

    np.testing.assert_array_equal(coefficients, coefficients_true)
