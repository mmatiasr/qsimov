import numpy as np
import pytest
import tests.nn_mocks_keras as nn_mocks
from qsimov.keras_path_selector import KerasPathSelector
import qsimov.paths.paths as paths_
import tensorflow as tf
from tensorflow import keras as kr

krl = kr.layers
Dataset = tf.data.Dataset


def make_model():
    return nn_mocks.make_model_3221(dropout=True)


@pytest.fixture(name="path_selector")
def make_path_selector():
    # use for path selection
    return KerasPathSelector(neural_network=make_model(), initial_layer=0)


@pytest.fixture(name="X")
def make_samples():
    return np.array([[1, 2, 3], [0, 3, 4], [-2, -2, -2]])


def test__all_layer_types_valid():
    # invalid layer type
    with pytest.raises(ValueError):
        model = kr.Sequential()
        model.add(krl.Dense(2, input_shape=(3,), activation="relu"))
        model.add(krl.BatchNormalization())
        model.add(krl.Dense(2, input_shape=(3,), activation="relu"))
        model.compile(optimizer="adam", loss="mse")
        KerasPathSelector(neural_network=model, initial_layer=0)


def test__layer_subset_valid():
    # invalid initial layer
    with pytest.raises(ValueError):
        KerasPathSelector(neural_network=make_model(), initial_layer=-5)

    with pytest.raises(ValueError):
        KerasPathSelector(neural_network=make_model(), initial_layer=4)


def test__compute_all_paths(path_selector):
    # check full path list
    all_paths = path_selector._all_paths
    all_paths_true = np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 2, 1],
            [0, 1, 1, 1],
            [0, 1, 2, 1],
            [0, 2, 2, 1],
            [1, 1, 1, 1],
            [1, 1, 2, 1],
            [1, 2, 2, 1],
            [2, 1, 1, 1],
            [2, 1, 2, 1],
            [2, 2, 2, 1],
            [3, 1, 1, 1],
            [3, 1, 2, 1],
            [3, 2, 2, 1],
        ]
    )
    all_paths_true = paths_.sort_paths(all_paths_true)
    np.testing.assert_array_equal(all_paths, all_paths_true)

    # check output masks as well, should all be true
    np.testing.assert_equal(
        sum(path_selector.output_masks_[0]), len(all_paths)
    )


def test__make_transformation_indices(path_selector):
    # check zero to zero counts
    np.testing.assert_array_equal(
        path_selector._zero_to_zero_counts, [3, None, 1, None, 0]
    )

    # check partial to full transformation indexes
    partial_to_full_idxs_true = [
        [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7],
        None,
        [0, 1, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4],
        None,
        [0, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2],
    ]
    for layer_idx in range(len(partial_to_full_idxs_true)):
        np.testing.assert_array_equal(
            path_selector._partial_to_full_idxs[layer_idx],
            partial_to_full_idxs_true[layer_idx],
        )


def test__propagate_left(path_selector, X):
    # initial version doesnt propagate, as initial_layer is 0
    np.testing.assert_array_equal(path_selector._propagate_left(X), X)

    # start path selector on second layer
    path_selector = KerasPathSelector(
        neural_network=make_model(), initial_layer=1
    )
    propagation = path_selector._as_numpy(path_selector._propagate_left(X))

    # check propagation
    propagation_true = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 7.0]])
    np.testing.assert_array_almost_equal(propagation, propagation_true)


def test_samples_to_coefficients(X, path_selector):
    coefficients = path_selector.samples_to_coefficients(X)

    # known expected equations
    coefficients_true = np.asfarray(
        [
            [1, 1, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    np.testing.assert_array_equal(coefficients, coefficients_true)


BATCH_SIZE = 2
NUMBER_OUTPUTS = 2


def test_as_tensorflow_dataset(path_selector, X):
    Y_dummy = np.random.uniform(0, 1, (len(X), NUMBER_OUTPUTS))

    for Y in [None, Y_dummy]:
        dataset = path_selector.as_tensorflow_dataset(
            X, Y, batch_size=BATCH_SIZE
        )
        assert isinstance(dataset, Dataset)

        # retrieve a batch from the mapped dataset
        if Y is None:
            X_batch = next(iter(dataset))
        else:
            X_batch, Y_batch = next(iter(dataset))

        # as many elements as batch size
        np.testing.assert_equal(X_batch.shape[0], BATCH_SIZE)

        if Y is not None:
            np.testing.assert_equal(Y_batch.shape[0], BATCH_SIZE)

        # input size is as many paths there are in path selector
        np.testing.assert_equal(
            X_batch.shape[1], len(path_selector._all_paths)
        )

        # output shape still makes sense
        if Y is not None:
            np.testing.assert_equal(Y_batch.shape[1], NUMBER_OUTPUTS)
