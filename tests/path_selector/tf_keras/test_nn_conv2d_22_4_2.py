import numpy as np
import pytest
from qsimov.keras_path_selector import KerasPathSelector
from qsimov.path_selector import _flatten_and_insert_one
import qsimov.paths.paths as paths_
import qsimov.paths.combine as combine
from tensorflow import keras as kr

krl = kr.layers


@pytest.fixture(name="path_selector")
def make_path_selector():
    # create model
    model = kr.Sequential()
    model.add(
        krl.Conv2D(
            1,
            kernel_size=2,
            input_shape=(2, 2, 2),
            activation="relu",
            use_bias=False,
            padding="same",
        )
    )
    model.add(krl.Dropout(0.1))
    model.add(krl.Flatten())
    model.add(krl.Dense(2))
    model.compile(optimizer="adam", loss="mse")

    # "train" model
    weights1 = np.array([[[[1], [1]], [[1], [1]]], [[[0], [0]], [[-1], [-1]]]])
    weights4 = np.array([[1, -1], [-1, 1], [1, -1], [-1, 1]])
    biases4 = np.array([3, 3])
    model.layers[0].set_weights([weights1])
    model.layers[3].set_weights([weights4, biases4])

    # use for path selection
    return KerasPathSelector(neural_network=model, initial_layer=0)


@pytest.fixture(name="X")
def make_samples():
    sample1 = np.array([[[1, 0], [2, 3]], [[0, -1], [1, -2]]])
    return np.array([sample1]).astype(float)


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

    # all paths between layer 1 and 2
    all_paths_layer4_true = np.array(
        [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]]
    ).astype(np.int32)

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

    # all paths between layer 1 and 2
    selected_paths_layer4_true = np.array([[0, 1], [1, 1], [2, 1]]).astype(
        np.int32
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
