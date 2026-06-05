import numpy as np
import pytest
import tests.nn_mocks_pytorch as nn_mocks
from qsimov.pytorch_path_selector import PytorchPathSelector
import qsimov.paths.paths as paths_
import torch.nn as nn
from torch.utils.data import DataLoader


def make_model():
    return nn_mocks.make_model_3221(dropout=True)


@pytest.fixture(name="path_selector")
def make_path_selector():
    # use for path selection
    return PytorchPathSelector(
        neural_network=make_model(), initial_layer=0, input_shape=(3)
    )


@pytest.fixture(name="X")
def make_samples():
    return np.array(
        [[1.0, 2.0, 3.0], [0.0, 3.0, 4.0], [-2.0, -2.0, -2.0]],
        dtype=np.float32,
    )


def test__all_layer_types_valid():
    # invalid layer type
    with pytest.raises(ValueError):
        model = nn.Sequential(
            nn.Linear(3, 2),
            nn.ReLU(),
            nn.BatchNorm1d(2),
            nn.Linear(2, 2),
            nn.ReLU(),
        )
        PytorchPathSelector(
            neural_network=model, initial_layer=0, input_shape=(3)
        )


def test__layer_subset_valid():
    # invalid initial layer
    with pytest.raises(ValueError):
        PytorchPathSelector(
            neural_network=make_model(), initial_layer=-5, input_shape=(3)
        )

    with pytest.raises(ValueError):
        PytorchPathSelector(
            neural_network=make_model(), initial_layer=4, input_shape=(3)
        )


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
    path_selector = PytorchPathSelector(
        neural_network=make_model(), initial_layer=2, input_shape=(3)
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


def test_as_pytorch_dataloader(path_selector, X):
    Y_dummy = np.random.uniform(0, 1, (len(X), NUMBER_OUTPUTS))

    for Y in [None, Y_dummy]:
        dataset = path_selector.as_pytorch_dataloader(
            X, Y, batch_size=BATCH_SIZE, shuffle=False
        )
        assert isinstance(dataset, DataLoader)

        batches = list(dataset)

        # retrieve a batch from the mapped dataset
        if Y is None:
            X_batch = batches[0]
            # compare to samples_to_coefficients
            coefficients = path_selector.samples_to_coefficients(X)
            np.testing.assert_array_equal(
                np.concatenate(batches), coefficients
            )
        else:
            X_batch, Y_batch = batches[0]

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
