import numpy as np
import pytest
import qsimov.paths.paths as paths_


def test_sort_paths():
    paths_nn = np.array(
        [
            [1, 1, 1],
            [1, 2, 1],
            [2, 1, 1],
            [2, 2, 1],
            [0, 1, 1],
            [0, 2, 1],
            [1, 1, 2],
            [1, 2, 2],
            [2, 1, 2],
            [2, 2, 2],
            [0, 1, 2],
            [0, 2, 2],
            [0, 0, 2],
        ]
    )
    paths_sorted_true = np.array(
        [
            [0, 0, 2],
            [0, 1, 1],
            [0, 1, 2],
            [0, 2, 1],
            [0, 2, 2],
            [1, 1, 1],
            [1, 1, 2],
            [1, 2, 1],
            [1, 2, 2],
            [2, 1, 1],
            [2, 1, 2],
            [2, 2, 1],
            [2, 2, 2],
        ]
    )
    paths_sorted = paths_.sort_paths(paths_nn)
    np.testing.assert_array_equal(paths_sorted, paths_sorted_true)


@pytest.mark.parametrize(
    "paths, paths_set, solution",
    [
        (
            np.array([[0, 1], [1, 1]]),
            np.array([[1, 1]]),
            np.array([False, True]),
        ),
        (
            np.array([[0, 0, 1], [1, 1, 1], [1, 2, 2], [3, 1, 1]]),
            np.array([[0, 0, 1], [1, 1, 1], [3, 1, 1]]),
            np.array([True, True, False, True]),
        ),
        (np.empty((0, 4)), np.empty((0, 4)), np.empty((0,), dtype=np.bool8)),
        (
            np.array([[0, 0, 0, 1], [0, 0, 1, 2]]),
            np.empty((0, 4)),
            np.array([False, False]),
        ),
        (
            np.array([[0, 1, 5]]),
            np.empty((0, 3)),
            np.array([False]),
        ),
        (
            np.array([[0, 1], [1, 1], [1, 2]]),
            np.array([[0, 1], [1, 2]]),
            np.array([True, False, True]),
        ),
    ],
)
def test_paths_subset_of(paths, paths_set, solution):
    paths = paths_.sort_paths(paths)
    paths_set = paths_.sort_paths(paths_set)

    is_subset = paths_.paths_subset_of(paths, paths_set)

    # check correct type
    assert is_subset.dtype == np.bool8

    # check correct shape
    assert is_subset.shape == (paths.shape[0],)

    # compare arrays
    np.testing.assert_array_equal(is_subset, solution)


@pytest.mark.parametrize(
    "paths_input_neurons, flat_inputs_with_bias, select_masks, solution",
    [
        (np.empty((0,)), [[1, -3, 4.5]], np.empty((1, 0)), np.empty((1, 0))),
        (
            np.array([0, 0, 1, 1, 1, 2, 2]),
            np.array([[1, -3, 4.5], [1, 2, 7]]),
            np.array(
                [
                    [True, False, True, False, False, True, True],
                    [False, False, False, True, False, False, True],
                ]
            ),
            np.array([[1, 0, -3, 0, 0, 4.5, 4.5], [0, 0, 0, 2, 0, 0, 7.0]]),
        ),
    ],
)
def test_retrieve_coefficients(
    paths_input_neurons, flat_inputs_with_bias, select_masks, solution
):
    coefficients = paths_.retrieve_coefficients(
        select_masks, paths_input_neurons, flat_inputs_with_bias
    )

    assert coefficients.shape == (
        select_masks.shape[0],
        paths_input_neurons.shape[0],
    )

    # compare arrays
    np.testing.assert_array_equal(coefficients, solution)
