import pytest
import qsimov.linalg as qs_alg
import numpy as np


@pytest.fixture(name="determinate_system")
def make_determinate_system():
    A = np.array([[2, 1, 1], [2, -1, 2], [1, -2, 1]], dtype=float)
    B = np.array([7, 6, 0], dtype=float)
    return np.hstack((A, B[:, None]))


@pytest.fixture(name="overdetermined_incompatible_system")
def make_overdetermined_incompatible_system():
    A = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 17],
        ],
        dtype=float,
    )
    B = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    return np.hstack((A, B[:, None]))


@pytest.fixture(name="underdetermined_incompatible_system")
def make_underdetermined_incompatible_system():
    A = np.array([[1, 1, 1], [1, 1, 1]], dtype=float)
    B = np.array([1, 0], dtype=float)
    return np.hstack((A, B[:, None]))


def test_r_transform(
    determinate_system,
    overdetermined_incompatible_system,
    underdetermined_incompatible_system,
):
    include_last_rows = [True, False, False]
    for AB, include_last_row in zip(
        (
            determinate_system,
            overdetermined_incompatible_system,
            underdetermined_incompatible_system,
        ),
        include_last_rows,
    ):
        # split in half
        AB_old, AB_new = AB[: len(AB) // 2], AB[len(AB) // 2 :]

        # r update empty equation matrix with first half
        AB_old = qs_alg.r_transform(AB_old)

        # r update first half of equation matrix with second half
        AB_final = qs_alg.r_transform(np.vstack((AB_old, AB_new)))

        # assert solution equal when previous transform of full system
        AB_transform = qs_alg._qr_transform_linear_system(AB)

        np.testing.assert_array_almost_equal(
            # last equation not included when using r updating
            qs_alg.solve(
                AB_final, include_last_row=include_last_row, rcond=None
            ),
            qs_alg.solve(AB_transform, include_last_row=True, rcond=None),
        )

        # also if we use r transforms
        AB_transform = qs_alg.r_transform(AB)
        np.testing.assert_array_almost_equal(
            qs_alg.solve(
                AB_final, include_last_row=include_last_row, rcond=None
            ),
            qs_alg.solve(
                AB_transform, include_last_row=include_last_row, rcond=None
            ),
        )


@pytest.mark.parametrize(
    "A, b, A_square, b_square",
    [
        (
            np.array([[1, 2, 3, 4], [0, 0, 0, 1], [0, 0, 0, 2]]),
            np.array([1, 2, 3]),
            np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 2.0],
                ]
            ),
            np.array([1.0, 0.0, 0.0, 3.0]),
        ),
        (
            np.array([[1, 2, 3, 4], [0, 0, 5, 6]]),
            np.array([1, 2]),
            np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 5.0, 6.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            np.array([1.0, 0.0, 2.0, 0.0]),
        ),
    ],
)
def test__make_square_system(A, b, A_square, b_square):
    A_new, b_new = qs_alg._make_square_system(A, b)
    np.testing.assert_array_equal(A_new, A_square)
    np.testing.assert_array_equal(b_new, b_square)


def test_solve(
    determinate_system,
    overdetermined_incompatible_system,
    underdetermined_incompatible_system,
):
    # back substitution must be able to handle determinate system
    solution = qs_alg.solve(
        qs_alg.r_transform(determinate_system),
        solver="back_substitution",
        include_last_row=True,
    )
    np.testing.assert_array_almost_equal(
        solution,
        np.linalg.solve(determinate_system[:, :-1], determinate_system[:, -1]),
    )

    # also overdetermined system after if we use r transform
    solution = qs_alg.solve(
        qs_alg.r_transform(overdetermined_incompatible_system),
        solver="back_substitution",
        include_last_row=False,
    )
    np.testing.assert_array_almost_equal(
        solution,
        np.linalg.lstsq(
            overdetermined_incompatible_system[:, :-1],
            overdetermined_incompatible_system[:, -1],
            rcond=None,
        )[0],
    )

    # should not be able to process underdetermined system
    with pytest.warns():
        solution = qs_alg.solve(
            qs_alg.r_transform(underdetermined_incompatible_system),
            solver="back_substitution",
            include_last_row=False,
        )

    np.testing.assert_array_almost_equal(solution, [0.5, 0.0, 0.0])


@pytest.mark.parametrize(
    "A, b, absolute_cutoff, relative_cutoff, solution",
    [
        (
            np.array([[1, 1, 1], [0, 2, 1], [0, 0, 1]], dtype=float),
            np.array([0, 1, 0], dtype=float),
            1e-8,
            1e6,
            np.array([-1 / 2, 1 / 2, 0]),
        ),
        (
            np.array([[3, 7, 1], [0, 4, 2], [0, 0, 1]], dtype=float),
            np.array([1, 2, 4], dtype=float),
            1e-9,
            1e6,
            np.array([5 / 2, -3 / 2, 4]),
        ),
        (
            np.array(
                [[4, 7, 2, 5], [0, 3, 2, 3], [0, 0, 4, 8], [0, 0, 0, 9]],
                dtype=float,
            ),
            np.array([7, 2, 1, 0], dtype=float),
            1e-5,
            1e5,
            np.array([3 / 4, 1 / 2, 1 / 4, 0]),
        ),
    ],
)
def test_back_substitution(A, b, absolute_cutoff, relative_cutoff, solution):
    solution_test = qs_alg.back_substitution(
        A, b, absolute_cutoff, relative_cutoff
    )
    np.testing.assert_array_almost_equal(solution_test, solution)


@pytest.mark.parametrize(
    "A, b, absolute_cutoff, relative_cutoff, solution",
    [
        (
            np.array([[4, 1, 2, 3], [0, 0, 1, 1]], dtype=float),
            np.array([1, 1], dtype=float),
            1e-8,
            1e6,
            np.array([-1 / 4, 0, 1, 0]),
        ),
        (
            np.array([[0, 5, 2, 1], [0, 0, 0, 3]], dtype=float),
            np.array([1, 3], dtype=float),
            1e-8,
            1e6,
            np.array([0, 0, 0, 1]),
        ),
    ],
)
def test_back_substitution_warnings_underdetermined(
    A, b, absolute_cutoff, relative_cutoff, solution
):
    with pytest.warns():
        solution_test = qs_alg.back_substitution(
            A, b, absolute_cutoff, relative_cutoff
        )
    np.testing.assert_array_almost_equal(solution_test, solution)


@pytest.mark.parametrize(
    "A, b, absolute_cutoff, relative_cutoff, solution",
    [
        (
            np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=float),
            np.array([1, 1], dtype=float),
            1e-8,
            1e6,
            np.array([0, 0, 0, 0]),
        ),
    ],
)
def test_back_substitution_warnings_zeros(
    A, b, absolute_cutoff, relative_cutoff, solution
):
    with pytest.warns():
        solution_test = qs_alg.back_substitution(
            A, b, absolute_cutoff, relative_cutoff
        )
    np.testing.assert_array_almost_equal(solution_test, solution)
