import numpy as np
from qsimov.paths.c_combine import (
    c_combine_paths_left_right_sort_join,
    c_combine_paths_left_right_hash_join,
)
from qsimov.paths.combine import (
    compute_combine_paths_output_size,
    combine_paths,
)
from qsimov.paths.paths import sort_paths
import pytest


# different implementations
@pytest.mark.parametrize(
    "combine_paths_left_right_f",
    [
        c_combine_paths_left_right_sort_join,
        c_combine_paths_left_right_hash_join,
    ],
)
@pytest.mark.parametrize(
    "paths_left, paths_right, solution",
    [
        (np.empty((0, 4)), np.empty((0, 2)), np.empty((0, 5))),
        (np.empty((0, 3)), np.empty((0, 4)), np.empty((0, 6))),
        (
            np.array([[0, 0, 1, 2], [0, 0, 0, 1]]),
            np.empty((0, 4)),
            np.empty((0, 7)),
        ),
        (
            np.empty((0, 3)),
            np.array([[0, 1, 1], [0, 0, 1]]),
            np.array([[0, 0, 0, 1, 1], [0, 0, 0, 0, 1]]),
        ),
        (
            np.empty((0, 3)),
            np.array([[0, 1]]),
            np.array([[0, 0, 0, 1]]),
        ),
        (
            np.array([[0, 1], [1, 1]]),
            np.array([[1, 2], [1, 1]]),
            np.array([[0, 1, 2], [0, 1, 1], [1, 1, 2], [1, 1, 1]]),
        ),
        (
            np.array([[1, 2, 3]]),
            np.array([[2, 2, 2]]),
            np.empty((0, 5)),
        ),
        (
            np.array([[1, 2], [1, 1], [0, 1]]),
            np.array([[3, 4], [1, 2], [0, 1]]),
            np.array([[1, 1, 2], [0, 1, 2], [0, 0, 1]]),
        ),
        (
            np.array([[0, 0, 1], [1, 1, 1], [0, 1, 2]]),
            np.array([[0, 0, 1], [1, 1, 1], [1, 2, 2], [3, 1, 1]]),
            np.array(
                [
                    [0, 0, 0, 0, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 2, 2],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 2, 2],
                ]
            ),
        ),
    ],
)
def test_combine_paths_left_right(
    combine_paths_left_right_f, paths_left, paths_right, solution
):
    paths_left = paths_left.astype(np.int32)
    paths_right = paths_right.astype(np.int32)

    paths_combined = combine_paths_left_right_f(paths_left, paths_right)

    # check correct type
    assert paths_combined.dtype == "int32"

    # check correct shape (necessary for empty cases)
    assert (
        paths_combined.shape[1]
        == paths_left.shape[1] + paths_right.shape[1] - 1
    )

    # compare arrays
    np.testing.assert_array_equal(
        sort_paths(paths_combined), sort_paths(solution)
    )


@pytest.mark.parametrize(
    "list_paths, solution",
    [
        (
            [
                np.array([[0, 1], [1, 1]]),
                np.array([[1, 2], [1, 1]]),
                np.array([[0, 1], [1, 1], [1, 2], [2, 2]]),
            ],
            np.array(
                [
                    [0, 0, 0, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 2],
                    [0, 1, 2, 2],
                    [1, 1, 1, 1],
                    [1, 1, 1, 2],
                    [1, 1, 2, 2],
                ]
            ),
        ),
        (
            [np.array([[0, 1], [1, 1], [1, 2], [2, 2]])],
            np.array([[0, 1], [1, 1], [1, 2], [2, 2]]),
        ),
        ([], np.empty((0, 2))),
        (
            [
                np.array([[0, 1], [1, 1]]),
                np.empty((0, 2)),
                np.array([[0, 1], [1, 1], [1, 2], [2, 2]]),
            ],
            np.array([[0, 0, 0, 1]]),
        ),
    ],
)
def test_combine_paths(list_paths, solution):
    paths_combined = combine_paths(list_paths)

    # check correct type
    assert paths_combined.dtype == "int32"

    # check correct shape (necessary for empty cases)
    np.testing.assert_array_equal(paths_combined.shape, solution.shape)

    # compare arrays
    np.testing.assert_array_equal(paths_combined, solution)

    assert paths_combined.shape[0] == compute_combine_paths_output_size(
        list_paths
    )
