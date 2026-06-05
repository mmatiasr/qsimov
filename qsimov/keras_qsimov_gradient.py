"""Contains functionality related to the qsimov neural network algorithm
applying gradient descent.
"""

import numpy as np
from qsimov.keras_path_selector import KerasPathSelector
from qsimov.mixins import LogMixin, NumpyPersistanceMixin
import tensorflow as tf
import os
from tensorflow import keras as kr


krl = kr.layers
backend = kr.backend
InputSpec = krl.InputSpec
custom_object_scope = kr.utils.custom_object_scope


class KerasQsimovGradient(LogMixin, NumpyPersistanceMixin):
    """Applies Qsimov algorithm with gradient descent to a keras neural network
    to replace a subset of the last layers with a flat layer.

    Attributes
    ----------
    model_ : tensorflow.keras.Sequential
        Model that replaces layers where the algorithm is applied.
    """

    # Variables to be persisted using np.savez_compressed
    _NUMPY_VARIABLES = ["_Y_shape"]

    def __init__(
        self,
        path_selector,
        verbose=1,
    ):
        """Create a KerasQsimovGradient instance.

        Parameters
        ----------
        path_selector : KerasPathSelector
            Instance of KerasPathSelector that will be used to select the
            paths that will be replaced by a flat layer.
        verbose : int, optional
            Verbosity level to be used by logging messages, and default
            verbosity for the fit method, by default 1.
        """
        LogMixin.__init__(self, verbose)
        # persistance settings
        NumpyPersistanceMixin.__init__(self, self._NUMPY_VARIABLES)

        # path selector
        self._path_selector = path_selector

        # model that represents the algorithm.
        self.model_ = None

        # flag to indicate if the model has been compiled
        self._compiled = False

        # output shape
        self._Y_shape = None

    def _make_model(self, device=None, **kwargs):
        """Create model that solves path selection dataset using keras API.

        Parameters
        ----------
        device : str, optional
            Device to be used by the model, by default None, which means
            that the default device will be used.

        **kwargs: dict
            Keyword arguments that will be passed on the compile to the
            internal one layer model that will replace the original model's
            layers where the algorithm will be applied.

            The following are supported, and the details may be consulted on
            the keras official docummentation:

            https://keras.io/api/models/model_training_apis/

            - Keras compile:\n
                - optimizer\n
                - loss\n
                - metrics\n
                - loss_weights\n
                - weighted_metrics\n
                - steps_per_execution\n

        Returns
        -------
        tensorflow.keras.Sequential
            One layer model.
        """
        if self.model_ is not None:
            self._compile_model(self.model_, **kwargs)
            return self.model_

        device = device or KerasPathSelector._check_device_or_get_default(
            device
        )

        with tf.device(device):
            # activation layers after last layer that affects path selection
            activation_layers = list(
                map(
                    lambda idx: self._path_selector._layers[idx],
                    self._path_selector._get_last_activation_layer_idxs(),
                )
            )
            # use copies
            activation_layers = map(
                lambda layer: layer.__class__(**layer.get_config()),
                activation_layers,
            )

            # create model for gradient based solving
            model = kr.Sequential(
                [
                    krl.Input(shape=len(self._path_selector._all_paths)),
                    # mask conenctions to each output
                    CustomConnectedDense(
                        units=self._path_selector._number_outputs,
                        connection_mask=self._path_selector.output_masks_.T,
                        use_bias=False,
                    ),
                    *activation_layers,
                ],
            )

            # compile model
            self._compile_model(model, **kwargs)

            return model

    def _compile_model(self, model, **kwargs):
        """Compile the one layer model with the arguments passed.

        Parameters
        ----------
        model : tensorflow.keras.Sequential
            Model to be compiled.
        **kwargs: dict
            Keyword arguments that will be passed on the compile to the
            internal one layer model that will replace the original model's
            layers where the algorithm will be applied.
        """
        model.compile(
            optimizer=kwargs.get("optimizer", "adam"),
            loss=kwargs.get("loss", "mse"),
            metrics=kwargs.get("metrics"),
            loss_weights=kwargs.get("loss_weights"),
            weighted_metrics=kwargs.get("weighted_metrics"),
            run_eagerly=kwargs.get("run_eagerly", False),
            steps_per_execution=kwargs.get("steps_per_execution", 1),
            jit_compile=False,
        )
        return model

    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        device=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=False,
        steps_per_execution=None,
    ):
        """Create and compile the model that solves path selection dataset
        using keras API. Specifying the device is possible, but currently
        only if the model has not been created yet, i.e. a call to compile()
        has not been made yet.

        Parameters
        ----------
        optimizer : str or tf.keras.optimizers
            The optimizer for the model.
        loss : str or tf.keras.losses
            The loss function.
        device : str, optional
            Device to be used by the model, by default None, which means
            that the default device will be used. Currently only supported
            if the model has not been created yet.
        metrics : list of str, functions or tf.keras.metrics
            List of metrics to be evaluated by
            the model during training and testing.
        loss_weights : list or dictionary
            Optional list or dictionary specifying scalar coefficients
            to weight the loss contributions of different model outputs.
        weighted_metrics : list
            List of metrics to be evaluated and weighted by
            sample_weight or class_weight during training and testing.
        run_eagerly : bool
            Control the execution mode of the model during training.
        steps_per_execution : int
            The number of batches to run.
        """
        self.model_ = self._make_model(
            device=device,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
        )
        self._compiled = True

    def _get_validation_dataset(self, **kwargs):
        """Util function to extract possibly specified validation dataset
        for the model fitting.

        Parameters
        ----------
        **kwargs: dict
            Keyword arguments passed to the fit method.

        Returns
        -------
        None or tensorflow.data.Dataset
            None if no dataset was specified, else the validation dataset
            mapped to coefficients generated by path selector.

        Raises
        ------
        ValueError
            If specified validation dataset is of incorrect type.
        """
        validation_data = kwargs.get("validation_data", None)
        if validation_data is None:
            return validation_data
        if type(validation_data) is tuple and len(validation_data) == 2:
            X, Y = validation_data
            valid_types = (np.ndarray,)
            if type(X) in valid_types and type(Y) in valid_types:
                return self._path_selector.as_tensorflow_dataset(
                    X, Y, batch_size=kwargs.get("batch_size", None)
                )
        raise ValueError("Validation data must be a tuple of two numpy arrays")

    def fit(self, X, Y, **kwargs):
        """Applies Qsimov algorithm to transform specified layers of a neural
        network, adapted to new data X, Y.

        Parameters
        ----------
        X : arrayNd
            Inputs.
        Y : arrayNd
            Outputs.
        **kwargs: dict
            Keyword arguments that will be passed on the fit method to the
            internal model that will replace the original model's layers where
            the algorithm will be applied.
            Supported arguments are:\n
                - epochs\n
                - batch_size\n
                - callbacks\n
                - validation_data: But currently only a tuple of numpy arrays
                    (X, y) is accepted.\n
                - class_weight\n
                - sample_weight\n
                - initial_epoch\n
                - steps_per_epoch\n
                - shuffle\n
                - verbose\n

        Returns
        -------
        tensorflow.keras.callbacks.History
            Result of calling fit on internal keras neural network that
            transforms the original model's layers.

        Raises
        ------
        ValueError
            If passed validation data is not a tuple of two numpy arrays.
        RuntimeError
            If model has not been compiled before fitting.
        """
        if not self._compiled:
            raise RuntimeError(
                "Model must be compiled before fitting, call compile() first."
            )

        # save for prediction
        self._Y_shape = Y.shape

        # reshape if necessary
        if len(Y.shape) != 2:
            Y = Y.reshape(len(Y), -1)

        # map inputs to coefficients
        path_selection_dataset = self._path_selector.as_tensorflow_dataset(
            X,
            Y,
            batch_size=kwargs.get("batch_size", None),
            shuffle=kwargs.get("shuffle", True),
        )

        return self.model_.fit(
            x=path_selection_dataset,
            epochs=kwargs.get("epochs", 10),
            verbose=kwargs.get("verbose", self._verbose),
            callbacks=kwargs.get("callbacks", None),
            validation_data=self._get_validation_dataset(**kwargs),
            class_weight=kwargs.get("class_weight", None),
            sample_weight=kwargs.get("sample_weight", None),
            initial_epoch=kwargs.get("initial_epoch", 0),
            steps_per_epoch=kwargs.get("steps_per_epoch", None),
            workers=1,
            use_multiprocessing=False,
            shuffle=False,
        )

    def predict(self, X, batch_size=None, verbose=1):
        """Uses fitted model to predict outputs for specified inputs.

        Parameters
        ----------
        X : arrayNd
            Inputs.
        batch_size : int, optional
            Batch size to use for prediction, by default None
        verbose : int, optional
            Verbosity mode, by default 1

        Returns
        -------
        arrayNd
            Predicted outputs, with same shape as seen during train.

        Raises
        ------
        ValueError
            If model has not been fitted.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        return self.model_.predict(
            self._path_selector.as_tensorflow_dataset(
                X, batch_size=batch_size, shuffle=False
            ),
            verbose=verbose,
        ).reshape(len(X), *self._Y_shape[1:])

    @classmethod
    def load(cls, directory_path, device=None, path_selector_device=None):
        """Load KerasQsimovGradient instance from directory. Note that a call
        to compile() might be necessary after loading if you want to change
        the optimizer or loss function.

        Parameters
        ----------
        directory_path : str
            Path to directory.
        device : str, optional
            Device to use for the model, by default None, which means the
            GPU will be used if available.
        path_selector_device : str, optional
            Device to use for the path selector, by default None, which means
            the GPU will be used if available.

        Returns
        -------
        KerasQsimovGradient
            Qsimov gradient instance.

        Raises
        ------
        ValueError
            If the specified device is invalid.
        """
        instance = NumpyPersistanceMixin.load(directory_path)
        device = KerasPathSelector._check_device_or_get_default(device)

        # replace path to model with actual model
        if instance.model_ is not None:
            with custom_object_scope(
                {"CustomConnectedDense": CustomConnectedDense}
            ):
                with tf.device(device):
                    instance.model_ = kr.models.load_model(instance.model_)

        # replace path selector with actual path selector
        instance._path_selector = KerasPathSelector.load(
            instance._path_selector, device=path_selector_device
        )
        return instance

    def __getstate__(self):
        """Defines which variables are to be pickled.

        Note
        ----
        The directory where numpy and keras variables are stored is taken
        from the internal attribute _save_dir, which will be initialized
        if not previously set (normally when calling save method). This
        method also saves the path selector to a directory inside the
        _save_dir directory and return in the state the path to that
        directory.

        Returns
        -------
        dict
            Variables to be pickled with python pickle module.
        """
        # capture what is normally pickled
        state = NumpyPersistanceMixin.__getstate__(self)

        # persist path selector as a path to a directory
        path_selector_dir = os.path.join(self._save_dir, "path_selector.qsi")
        os.makedirs(path_selector_dir, exist_ok=True)
        KerasPathSelector.save(
            self._path_selector, path_selector_dir, verbose=0
        )
        state["_path_selector"] = path_selector_dir

        # if there is no neural network, we are finished
        if self.model_ is None:
            return state

        # save neural network in a directory and dont try to persist it
        # with pickle
        model_dir = os.path.join(self._save_dir, "model.h5")
        self.model_.save(model_dir)
        state["model_"] = model_dir

        return state


class CustomConnectedDense(krl.Dense):
    """Implements a keras dense layer with a specific connection pattern."""

    def __init__(self, units, connection_mask=None, **kwargs):
        """Create a dense layer with a custom connection pattern.

        Parameters
        ----------
        units : int
            Number of neurons in the layer.
        connection_mask : array2d or None
            A boolean array indicating which weights are kept, by default None.
        **kwargs : dict
            Keyword arguments to be passed to keras Dense class.
        """
        super().__init__(units, **kwargs)

        assert connection_mask is None or connection_mask.ndim == 2
        self.connection_mask = connection_mask

    def build(self, input_shape):
        """Builds the layer

        Parameters
        ----------
        input_shape : (int, )
            Shape of input to the network.

        Raises
        ------
        TypeError
            Attempt to build layer with a non floating point dtype.
        ValueError
            Incorrect input shape definition.
        """
        # copy start from source code in keras dense
        dtype = tf.as_dtype(self.dtype)
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "A Dense layer can only be built with a floating-point "
                f"dtype. Received: dtype={dtype}"
            )

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            "kernel",
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )

        # compute kernel shape
        kernel_shape = (last_dim, self.units)

        # set default connection mask
        if self.connection_mask is None:
            self.connection_mask = np.ones(kernel_shape)

        # define tf variable
        self.connection_mask = tf.Variable(
            initial_value=self.connection_mask,
            trainable=False,
            dtype=self.dtype,
            name="connection_mask",
        )

        # save built state
        self.built = True

    def call(self, inputs):
        """Use the layer to map an input.

        Parameters
        ----------
        inputs : array2d
            Sets of inputs.

        Returns
        -------
        array2d
            Outputs.
        """
        outputs = tf.linalg.matmul(
            inputs, tf.math.multiply(self.kernel, self.connection_mask)
        )

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
