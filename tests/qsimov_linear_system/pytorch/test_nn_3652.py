import numpy as np
import pytest
from qsimov.pytorch_qsimov_linear_system import PytorchQsimovLinearSystem
from qsimov.pytorch_path_selector import PytorchPathSelector
from tests.nn_mocks_pytorch import make_data_3652, make_model_3652
from tests.utils_load_save_tests import (
    assert_pytorch_qsimov_linear_system_equals,
)
import os
import tempfile


np.random.seed(42)
NUMBER_SAMPLES = 1000


@pytest.fixture(name="qsimov_linear_full")
def make_qsimov_linear_full():
    model = make_model_3652()
    # use for path selection
    return PytorchQsimovLinearSystem(
        PytorchPathSelector(
            neural_network=model,
            input_shape=3,
            initial_layer=0,
        ),
        absolute_cutoff=1e-8,
    )


@pytest.fixture(name="qsimov_linear_partial")
def make_qsimov_linear_partial():
    model = make_model_3652()
    # use for path selection
    return PytorchQsimovLinearSystem(
        PytorchPathSelector(
            neural_network=model,
            input_shape=3,
            initial_layer=-3,
        ),
        absolute_cutoff=1e-8,
    )


@pytest.fixture(name="data")
def make_data():
    return make_data_3652()


def mse(y, y_true):
    return np.mean(np.square(y - y_true))


def mean_norm(y):
    return np.linalg.norm(y) / len(y)


@pytest.mark.parametrize("solver", ["back_substitution", "lstsq"])
def test_fit(qsimov_linear_full, qsimov_linear_partial, data, solver):
    print(f"\n\nRunning with solver {solver}")
    X, Y = data

    def eval_solution(msg, solution, A0, B0, A1, B1):
        print(msg)
        print(f"MSE: {mse(A0 @ solution[0], B0)}, {mse(A1 @ solution[1], B1)}")
        print(f"Norm: {mean_norm(solution[0])}, {mean_norm(solution[1])}")

    for qsimov_linear in (qsimov_linear_full, qsimov_linear_partial):
        qsimov_linear._solver = solver
        path_selector = qsimov_linear._path_selector

        print(f"\nInitial layer: {path_selector._initial_layer}", end=". ")
        print(f"Paths: {len(path_selector._all_paths)}")

        # fit with whole data with no batch
        qsimov_linear.fit(X, Y)

        # extract equations
        equations = qsimov_linear.equations_
        A0, B0 = equations[0][:-1, :-1], equations[0][:-1, -1]
        A1, B1 = equations[1][:-1, :-1], equations[1][:-1, -1]

        # get solutions for whole values
        solutions_full_data = qsimov_linear.solutions_
        np.testing.assert_equal(
            len(solutions_full_data[0]), sum(path_selector.output_masks_[0])
        )
        np.testing.assert_equal(
            len(solutions_full_data[1]), sum(path_selector.output_masks_[1])
        )

        eval_solution("No batch", solutions_full_data, A0, B0, A1, B1)

        # fit in batches, with different batch sizes
        for batch_size in ((NUMBER_SAMPLES - 1) // 6, 32):
            print("Batch size", batch_size)

            # clear previous equations and fit
            qsimov_linear.reset_equations()
            qsimov_linear.fit(X, Y, batch_size=batch_size)

            # extract equations
            equations = qsimov_linear.equations_
            A0, B0 = equations[0][:-1, :-1], equations[0][:-1, -1]
            A1, B1 = equations[1][:-1, :-1], equations[1][:-1, -1]

            # get solutions
            solutions_batch = qsimov_linear.solutions_
            np.testing.assert_equal(
                len(solutions_batch[0]), sum(path_selector.output_masks_[0])
            )
            np.testing.assert_equal(
                len(solutions_batch[1]), sum(path_selector.output_masks_[1])
            )

            eval_solution("", solutions_batch, A0, B0, A1, B1)

            # solutions should be equal if lstsq
            if solver == "lstsq":
                np.testing.assert_array_almost_equal(
                    solutions_full_data[0], solutions_batch[0], decimal=5
                )
                np.testing.assert_array_almost_equal(
                    solutions_full_data[1], solutions_batch[1], decimal=5
                )


def test_save_load(qsimov_linear_full, data):
    for fit in (False, True):
        if fit:
            qsimov_linear_full.fit(*data)
        path = tempfile.mkdtemp()
        qsimov_linear_full.save(path)

        # save internally adds .qsi extension
        path += ".qsi"

        # check files were created
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(path, "numpy_variables.npz"))
        assert os.path.exists(os.path.join(path, "py_objects.pkl"))
        assert os.path.exists(os.path.join(path, "path_selector.qsi"))

        # load model
        qsimov_linear_full_loaded = PytorchQsimovLinearSystem.load(path)

        # assert correctly stored information
        assert_pytorch_qsimov_linear_system_equals(
            qsimov_linear_full_loaded, qsimov_linear_full
        )
