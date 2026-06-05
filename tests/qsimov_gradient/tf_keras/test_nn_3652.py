import numpy as np
import pytest
from qsimov.keras_qsimov_gradient import KerasQsimovGradient
from qsimov.keras_path_selector import KerasPathSelector
from tests.nn_mocks_keras import (
    make_data_3652_sigmoid,
    make_model_3652_sigmoid,
)
from tensorflow import keras as kr
from tests.utils_load_save_tests import assert_keras_qsimov_gradient_equals
import tempfile
import os

sigmoid = kr.activations.sigmoid
Adam = kr.optimizers.Adam


kr.utils.set_random_seed(42)
NUMBER_OUTPUTS = 2
NUMBER_SAMPLES = 300
BATCH_SIZE = 128


@pytest.fixture(name="qsimov_gradient_full")
def make_qsimov_gradient_full():
    model = make_model_3652_sigmoid()

    # remove one connection to first output
    weights_last_layer, biases_last_layer = model.layers[-1].get_weights()
    weights_last_layer[0, 0] = 0
    model.layers[-1].set_weights([weights_last_layer, biases_last_layer])

    # use for path selection
    qsimov_gradient = KerasQsimovGradient(
        KerasPathSelector(model, initial_layer=0, verbose=1)
    )
    qsimov_gradient.compile(
        optimizer=Adam(learning_rate=0.1),
        metrics=["accuracy"],
        loss="categorical_crossentropy",
    )
    return qsimov_gradient


@pytest.fixture(name="qsimov_gradient_full_no_compile")
def make_qsimov_gradient_full_no_compile():
    model = make_model_3652_sigmoid()

    # remove one connection to first output
    weights_last_layer, biases_last_layer = model.layers[-1].get_weights()
    weights_last_layer[0, 0] = 0
    model.layers[-1].set_weights([weights_last_layer, biases_last_layer])

    # use for path selection
    qsimov_gradient = KerasQsimovGradient(
        KerasPathSelector(model, initial_layer=0, verbose=1)
    )
    return qsimov_gradient


def fit_args():
    return dict(batch_size=BATCH_SIZE, epochs=2, verbose=1)


@pytest.fixture(name="data")
def make_data():
    return make_data_3652_sigmoid(number_samples=NUMBER_SAMPLES)


def test__make_model(qsimov_gradient_full_no_compile):
    model = qsimov_gradient_full_no_compile._make_model(
        optimizer=Adam(learning_rate=0.1),
        metrics=["accuracy"],
        loss="categorical_crossentropy",
    )

    # inputs are all paths
    np.testing.assert_equal(
        model.layers[0].input_shape[1],
        qsimov_gradient_full_no_compile._path_selector._all_paths.shape[0],
    )

    # matches the number output neurons
    np.testing.assert_equal(model.layers[0].output_shape[1], NUMBER_OUTPUTS)

    # activation function is sigmoid
    np.testing.assert_equal(model.layers[-1].activation, sigmoid)

    # optimizer is kept
    np.testing.assert_equal(model.optimizer.__class__.__name__, "Adam")
    np.testing.assert_almost_equal(float(model.optimizer.learning_rate), 0.1)

    # training loss is used
    np.testing.assert_equal(model.loss, "categorical_crossentropy")

    # connection mask is the same as path selector outputs masks transposed
    np.testing.assert_array_equal(
        model.layers[0].connection_mask,
        qsimov_gradient_full_no_compile._path_selector.output_masks_.T,
    )


@pytest.mark.parametrize("path_selector_device", ["/cpu:0", None])
@pytest.mark.parametrize("device", ["/cpu:0", None])
def test_save_load(qsimov_gradient_full, device, path_selector_device):
    path = tempfile.mkdtemp()
    qsimov_gradient_full.save(path)

    # save internally adds .qsi extension
    path += ".qsi"

    # check files were created
    assert os.path.exists(path)
    assert os.path.exists(os.path.join(path, "numpy_variables.npz"))
    assert os.path.exists(os.path.join(path, "py_objects.pkl"))
    assert os.path.exists(os.path.join(path, "path_selector.qsi"))
    assert os.path.exists(os.path.join(path, "model.h5"))

    # load model
    qsimov_gradient_full_loaded = KerasQsimovGradient.load(
        path, device=device, path_selector_device=path_selector_device
    )

    # assert correctly stored information
    assert_keras_qsimov_gradient_equals(
        qsimov_gradient_full_loaded, qsimov_gradient_full
    )


def test_fit(qsimov_gradient_full, data):
    # force model creation
    qsimov_gradient_full.model_ = qsimov_gradient_full._make_model()
    original_weights = qsimov_gradient_full.model_.layers[0].kernel.numpy()

    # ensure that fit does not raise any error
    qsimov_gradient_full.fit(*data, **fit_args())

    # ensure that fit updates the weights
    np.testing.assert_equal(
        np.all(
            original_weights
            == qsimov_gradient_full.model_.layers[0].kernel.numpy()
        ),
        False,
    )


def test_fit_no_compile_model(qsimov_gradient_full_no_compile, data):
    # fit raise compile before fitting error
    with pytest.raises(RuntimeError):
        qsimov_gradient_full_no_compile.fit(*data, **fit_args())
