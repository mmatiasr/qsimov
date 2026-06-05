import numpy as np
import pytest
import tests.nn_mocks_keras as nn_mocks
from qsimov.keras_path_selector import KerasPathSelector
import qsimov.paths.paths as paths_
from tensorflow import keras as kr

krl = kr.layers


@pytest.fixture(name="path_selector")
def make_path_selector():
    model = nn_mocks.make_model_222()

    # use for path selection
    return KerasPathSelector(neural_network=model, initial_layer=0)


@pytest.fixture(name="X")
def make_samples():
    return np.array([[1, 2], [0, 2]])


def test__layer_subset_valid():
    # invalid initial layer
    with pytest.raises(ValueError):
        KerasPathSelector(
            neural_network=nn_mocks.make_model_222(), initial_layer=-5
        )

    with pytest.raises(ValueError):
        KerasPathSelector(
            neural_network=nn_mocks.make_model_222(), initial_layer=4
        )


def test__compute_all_paths(path_selector):
    # check full path list
    all_paths = path_selector._all_paths
    all_paths_true = np.array(
        [
            [1, 1, 1],
            [1, 2, 1],
            [2, 1, 1],
            [2, 2, 1],
            [0, 1, 1],
            [0, 2, 1],
            [0, 0, 1],
        ]
    )
    np.testing.assert_array_equal(all_paths, paths_.sort_paths(all_paths_true))

    # first path (bias) should not be present for first output as it is zero
    np.testing.assert_array_equal(
        path_selector.output_masks_[0],
        [False, True, True, True, True, True, True],
    )
    # there are no zeros in last layer for second output
    np.testing.assert_equal(
        sum(path_selector.output_masks_[1]), len(all_paths)
    )


def test_samples_to_coefficients(X, path_selector):
    coefficients = path_selector.samples_to_coefficients(X)

    # known expected equations
    coefficients_true = np.array(
        [
            [1, 1, 0, 1, 0, 2, 0],
            [1, 1, 1, 0, 0, 2, 2],
        ]
    )
    np.testing.assert_array_equal(coefficients, coefficients_true)
