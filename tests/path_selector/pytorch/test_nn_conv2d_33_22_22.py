import numpy as np
import pytest
from qsimov.path_selector import _flatten_and_insert_one
from qsimov.pytorch_path_selector import PytorchPathSelector
import qsimov.paths.paths as paths_
import qsimov.paths.combine as combine
from qsimov.paths.conv import (
    _map_channels_last_paths_to_channels_first,
    get_all_paths_conv_layer,
)

import torch.nn as nn
import torch

from tests.paths.utils_conv_tests import (
    weights_to_channels_first,
    samples_to_channels_first,
)


def get_first_layer_weights():
    weights1 = np.array([[[[0, -2]], [[3, 1]]], [[[2, 1]], [[-1, 1]]]])
    biases1 = np.array([0, 1])

    # transform to channels first ordering
    weights1 = weights_to_channels_first(weights1)
    return torch.from_numpy(np.asfarray(weights1)), torch.from_numpy(
        np.asfarray(biases1)
    )


def get_second_layer_weights():
    weights2 = np.array(
        [[[[1], [-1]], [[1], [-1]]], [[[1], [-1]], [[1], [-1]]]]
    )

    # transform to channels first ordering
    weights2 = weights_to_channels_first(weights2)
    return torch.from_numpy(np.asfarray(weights2))


@pytest.fixture(name="path_selector")
def make_path_selector():
    # define layers
    conv_layer = nn.Conv2d(
        in_channels=1,
        out_channels=2,
        kernel_size=2,
        bias=True,
        padding="valid",
    )
    relu_layer = nn.ReLU()
    conv_layer2 = nn.Conv2d(
        in_channels=2,
        out_channels=1,
        kernel_size=2,
        bias=False,
        padding="same",
    )
    # create model
    model = nn.Sequential(conv_layer, relu_layer, conv_layer2)

    # get weights
    weights1, biases1 = get_first_layer_weights()
    weights2 = get_second_layer_weights()

    # set weights and biases
    conv_layer.weight.data = torch.from_numpy(np.asfarray(weights1))
    conv_layer.bias.data = torch.from_numpy(np.asfarray(biases1))
    conv_layer2.weight.data = torch.from_numpy(np.asfarray(weights2))

    # use for path selection
    return PytorchPathSelector(
        neural_network=model, input_shape=(1, 3, 3), initial_layer=0
    )


@pytest.fixture(name="X")
def make_samples():
    sample1 = np.array([[1, 2, 0], [3, -1, 4], [0, 2, 2]]).reshape(3, 3, 1)
    return samples_to_channels_first(np.array([sample1]).astype(float))


def all_paths_between_layer_0_and_1_channels_last():
    # all paths between layer 0 and 1
    all_paths_layer1_true = np.vstack(
        [
            # insert(input neurons, _, output_neuron,_)
            np.insert([[2], [4], [5]], 1, 1, axis=1),
            np.insert([[3], [5], [6]], 1, 3, axis=1),
            np.insert([[5], [7], [8]], 1, 5, axis=1),
            np.insert([[6], [8], [9]], 1, 7, axis=1),
            np.insert([[0], [1], [2], [4], [5]], 1, 2, axis=1),
            np.insert([[0], [2], [3], [5], [6]], 1, 4, axis=1),
            np.insert([[0], [4], [5], [7], [8]], 1, 6, axis=1),
            np.insert([[0], [5], [6], [8], [9]], 1, 8, axis=1),
        ]
    ).astype(np.int32)
    return all_paths_layer1_true


def all_paths_between_layer_1_and_2_channels_last():
    # all paths between layer 1 and 2
    all_paths_layer2_true = np.vstack(
        [
            np.insert([[1], [2], [3], [4], [5], [6], [7], [8]], 1, 1, axis=1),
            np.insert([[3], [4], [7], [8]], 1, 2, axis=1),
            np.insert([[5], [6], [7], [8]], 1, 3, axis=1),
            np.insert([[7], [8]], 1, 4, axis=1),
        ]
    ).astype(np.int32)
    return all_paths_layer2_true


def all_paths_between_layer_0_and_1_channels_first():
    return _map_channels_last_paths_to_channels_first(
        input_shape=(3, 3, 1),
        output_shape=(2, 2, 2),
        paths=all_paths_between_layer_0_and_1_channels_last(),
    )


def all_paths_between_layer_1_and_2_channels_first():
    return _map_channels_last_paths_to_channels_first(
        input_shape=(2, 2, 2),
        output_shape=(2, 2, 1),
        paths=all_paths_between_layer_1_and_2_channels_last(),
    )


def test__compute_all_paths(path_selector):
    # check full path list
    all_paths = path_selector._all_paths

    # all paths between layer 0 and 1
    all_paths_layer1_true = all_paths_between_layer_0_and_1_channels_first()

    # all paths between layer 1 and 2
    all_paths_layer2_true = all_paths_between_layer_1_and_2_channels_first()

    # compute all paths and split by output neuron
    all_paths_true = combine.combine_paths(
        [all_paths_layer1_true, all_paths_layer2_true]
    )
    all_paths_true = [
        all_paths_true[all_paths_true[:, 2] == output_neuron]
        for output_neuron in range(1, 5)
    ]

    # compare paths to each output against masked paths to any output
    out_masks = path_selector.output_masks_
    for output_idx in range(4):
        # get correct paths only to this output
        paths_true = np.array(paths_.sort_paths(all_paths_true[output_idx]))

        # set output neuron to 1 to match internal representation of path
        # selector
        paths_true[:, -1] = 1

        # compare to paths returned by path selector
        np.testing.assert_array_equal(
            all_paths[out_masks[output_idx]], paths_true
        )


def test_samples_to_coefficients(X, path_selector):
    coefficients = path_selector.samples_to_coefficients(X)

    # get all paths between layer 0 and 1
    all_paths_layer1_true = all_paths_between_layer_0_and_1_channels_first()

    # get all paths between layer 1 and 2, but not applying output mask
    # convention (as if last layer was dense)
    all_paths_layer2_true = get_all_paths_conv_layer(
        input_shape=(2, 2, 2),
        weights=get_second_layer_weights(),
        biases=np.zeros(1),
        padding="same",
        strides=(1, 1),
        data_format="channels_first",
    )

    # combine paths: this contains all possible paths
    all_paths_true = combine.combine_paths(
        [all_paths_layer1_true, all_paths_layer2_true]
    )

    # map each possible path to corresponding coefficient
    coefficients_true = _flatten_and_insert_one(X)[:, all_paths_true[:, 0]]

    # make path selection mask: output neurons 2, 3, 6, 7 of the first layer
    # are inactive, so we remove all paths to these neurons
    selection_mask = np.ones(all_paths_true.shape[0], dtype=bool)
    for output_neuron in [2, 3, 6, 7]:
        selection_mask &= all_paths_true[:, 1] != output_neuron

    # apply mask to coefficients
    coefficients_true[:, ~selection_mask] = 0

    # make masks to split by output neuron
    output_masks = [
        all_paths_true[:, -1] == output_neuron for output_neuron in range(1, 5)
    ]

    # split coefficients by output neuron
    coefficients_by_output_true = [
        coefficients_true[:, output_mask] for output_mask in output_masks
    ]

    # compare coefficients for each output neuron
    for output_idx in range(4):
        np.testing.assert_array_equal(
            coefficients[:, path_selector.output_masks_[output_idx]],
            coefficients_by_output_true[output_idx],
        )
