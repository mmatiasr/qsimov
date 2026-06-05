import numpy as np
import pytest
from qsimov.pytorch_qsimov_gradient import PytorchQsimovGradient
from qsimov.pytorch_path_selector import PytorchPathSelector
from tests.nn_mocks_pytorch import (
    make_data_3652_sigmoid,
    make_model_3652_sigmoid,
)
from tests.utils_load_save_tests import assert_pytorch_qsimov_gradient_equals
import tempfile
import os
import torch.nn as nn
from torch.optim import Adam


np.random.seed(42)
NUMBER_FEATURES = 3
NUMBER_OUTPUTS = 2
NUMBER_SAMPLES = 300
BATCH_SIZE = 128


@pytest.fixture(name="qsimov_gradient_full")
def make_qsimov_gradient_full():
    model = make_model_3652_sigmoid()

    # use for path selection
    return PytorchQsimovGradient(
        PytorchPathSelector(
            neural_network=model,
            input_shape=(NUMBER_FEATURES,),
            initial_layer=0,
        )
    )


def fit_args():
    def accuracy(y_true, y_pred):
        return (y_true == y_pred).float().mean()

    return dict(
        batch_size=BATCH_SIZE,
        epochs=2,
        loss_function=nn.BCELoss(),
        metrics=[accuracy],
        optimizer=lambda params: Adam(params, lr=0.01),
        verbose=1,
    )


@pytest.fixture(name="data")
def make_data():
    return make_data_3652_sigmoid(number_samples=NUMBER_SAMPLES)


def test__make_model(qsimov_gradient_full):
    model = qsimov_gradient_full._make_model()

    # inputs are all paths
    input_layer = next(model.children())
    np.testing.assert_equal(
        input_layer.in_features,
        qsimov_gradient_full._path_selector._all_paths.shape[0],
    )

    # matches the number output neurons
    np.testing.assert_equal(input_layer.out_features, NUMBER_OUTPUTS)

    # activation function is sigmoid
    activation_layer = list(model.children())[1]
    np.testing.assert_equal(activation_layer.__class__.__name__, "Sigmoid")

    # connection mask is the same as path selector outputs masks
    np.testing.assert_array_equal(
        input_layer.connection_mask,
        qsimov_gradient_full._path_selector.output_masks_,
    )


def test_save_load(qsimov_gradient_full, data):
    for fit in (False, True):
        if fit:
            qsimov_gradient_full.fit(*data, **fit_args())
        path = tempfile.mkdtemp()
        qsimov_gradient_full.save(path)

        # save internally adds .qsi extension
        path += ".qsi"

        # check files were created
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(path, "numpy_variables.npz"))
        assert os.path.exists(os.path.join(path, "py_objects.pkl"))
        assert os.path.exists(os.path.join(path, "path_selector.qsi"))

        if fit:
            assert os.path.exists(os.path.join(path, "model_weights.pt"))

        # load model
        qsimov_gradient_full_loaded = PytorchQsimovGradient.load(path)

        # assert correctly stored information
        assert_pytorch_qsimov_gradient_equals(
            qsimov_gradient_full_loaded, qsimov_gradient_full
        )


def test_fit(qsimov_gradient_full, data):
    # force model creation
    qsimov_gradient_full.model_ = qsimov_gradient_full._make_model()

    # store original weights
    input_layer = next(qsimov_gradient_full.model_.children())
    original_weights = input_layer.weight.detach().cpu().numpy().copy()

    # ensure that fit does not raise any error
    history = qsimov_gradient_full.fit(*data, **fit_args())

    # ensure that fit updates the weights (at least one weight is different)
    np.testing.assert_equal(
        np.all(
            original_weights
            == input_layer.weight.detach().cpu().numpy().copy()
        ),
        False,
    )

    # ensure that history is returned
    assert isinstance(history, dict)
    assert isinstance(history["loss"], list)
    assert isinstance(history["accuracy"], list)
    assert len(history["loss"]) == 2
    assert len(history["accuracy"]) == 2
    assert history["loss"][0] > history["loss"][1]

    # fit again, with validation data
    history = qsimov_gradient_full.fit(*data, *data, **fit_args())

    # ensure that history is returned
    for prefix in ("", "val_"):
        assert isinstance(history, dict)
        assert isinstance(history[prefix + "loss"], list)
        assert isinstance(history[prefix + "accuracy"], list)
        assert len(history[prefix + "loss"]) == 2
        assert len(history[prefix + "accuracy"]) == 2
