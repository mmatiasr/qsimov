import numpy as np
import pytest
from qsimov.keras_path_selector import KerasPathSelector
from qsimov.path_selector import _flatten_and_insert_one
import qsimov.paths.paths as paths_
from tensorflow import keras as kr

krl = kr.layers


@pytest.fixture(name="path_selector")
def make_path_selector():
    # create model
    model = kr.Sequential()
    # output shape (2,2,2)
    model.add(
        krl.MaxPool2D(
            pool_size=2, input_shape=(3, 3, 2), strides=(1, 1), padding="valid"
        )
    )
    model.add(krl.Dropout(0.1))
    model.add(krl.Flatten())
    model.add(krl.Dense(1))
    model.compile(optimizer="adam", loss="mse")

    # "train" model
    weights4 = np.ones((8, 1))  # 8 = 2*2*2
    biases4 = np.array([0])
    model.layers[3].set_weights([weights4, biases4])

    # use for path selection
    return KerasPathSelector(neural_network=model, initial_layer=0)


@pytest.fixture(name="X")
def make_samples():
    sample1 = np.array(
        [
            [[1, 0], [2, 0], [3, 1]],
            [[6, -1], [5, 0], [4, -1]],
            [[8, 2], [9, 3], [7, 4]],
        ]
    )
    return np.array([sample1]).astype(float)


def test__compute_all_paths(path_selector):
    # check full path list
    all_paths = path_selector._all_paths

    # expected output
    all_paths_true = np.vstack(
        [
            # insert(input neurons, _, output_neuron,_)
            np.insert([[1], [3], [7], [9]], 1, 1, axis=1),
            np.insert([[3], [5], [9], [11]], 1, 3, axis=1),
            np.insert([[7], [9], [13], [15]], 1, 5, axis=1),
            np.insert([[9], [11], [15], [17]], 1, 7, axis=1),
            np.insert([[2], [4], [8], [10]], 1, 2, axis=1),
            np.insert([[4], [6], [10], [12]], 1, 4, axis=1),
            np.insert([[8], [10], [14], [16]], 1, 6, axis=1),
            np.insert([[10], [12], [16], [18]], 1, 8, axis=1),
        ]
    )
    # add path to final neuron
    all_paths_true = np.insert(all_paths_true, 2, 1, axis=1).astype(np.int32)

    np.testing.assert_array_equal(
        all_paths[path_selector.output_masks_[0]],
        paths_.sort_paths(all_paths_true),
    )


def test_samples_to_coefficients(X, path_selector):
    coefficients = path_selector.samples_to_coefficients(X)

    # selected paths for input
    selected_paths_true = np.array(
        [[6, 4], [7, 1], [9, 3], [15, 5], [15, 7], [16, 6], [18, 8]]
    ).astype(np.int32)
    # add path to final neuron
    selected_paths_true = np.insert(selected_paths_true, 2, 1, axis=1)
    selected_paths_true = paths_.sort_paths(selected_paths_true)

    # transform selected paths to bool mask
    select_mask = paths_.paths_subset_of(
        path_selector._all_paths, selected_paths_true
    )

    # build coefficients
    coefficients_true = paths_.retrieve_coefficients(
        [select_mask],
        path_selector._all_paths[:, 0],
        _flatten_and_insert_one(X),
    )

    np.testing.assert_array_equal(
        coefficients[:, path_selector.output_masks_[0]],
        coefficients_true[:, path_selector.output_masks_[0]],
    )
