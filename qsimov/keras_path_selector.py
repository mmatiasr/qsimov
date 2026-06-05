"""Contains functionality related to path selection tasks on keras neural
networks.
"""
import contextlib
import numpy as np
from qsimov.mixins import NumpyPersistanceMixin
from qsimov.path_selector import (
    PathSelector,
    BaseLayerTypes,
    DEFAULT_BATCH_SIZE,
)
import tensorflow as tf
import os
import qsimov.paths.conv as qs_conv
import qsimov.paths.maxpooling as qs_maxpool
import qsimov.paths.dense as qs_dense
from tensorflow import keras as kr

Dataset = tf.data.Dataset


class KerasPathSelector(PathSelector):
    """Implements path selection for a subset of layers of a trained keras
    model. Further details can be found in the documentation of the parent
    class.

    Attributes
    ----------
    output_masks_ : array2d
        As documented in the parent class.
    left_model_ : torch.nn.Sequential
        As documented in the parent class.
    right_model_ : torch.nn.Sequential
        As documented in the parent class.
    """

    ############################
    # Class variables
    ############################

    @property
    def LayerTypes(self):
        """Returns the LayerTypes class for the path selector.

        Returns
        -------
        KerasLayerTypes
            BaseLayerTypes subclass.
        """
        return KerasLayerTypes

    @staticmethod
    def _check_device_or_get_default(device=None):
        """Checks if a device is valid or returns the default device.

        Parameters
        ----------
        device : str, optional
            Device to check, by default None.

        Returns
        -------
        str
            Device to use.

        Raises
        ------
        ValueError
            If the device is not valid.
        """
        if device is None:
            # Return first GPU if available, otherwise CPU.
            for device in tf.config.list_logical_devices("GPU"):
                return device.name
            return "/cpu:0"

        device_str = str(device)
        try:
            with tf.device(device_str):
                _ = tf.random.uniform((1,))
        except Exception:
            raise ValueError(
                "Invalid device: {}. Valid devices are: {}".format(
                    device_str, tf.config.list_logical_devices()
                )
            )
        return device_str

    ############################
    # Class initialization
    ############################

    def __init__(
        self, neural_network, initial_layer=0, verbose=1, device=None
    ):
        """Implements path selection using last layers of a given Keras model.
        Further details can be found in the documentation of the parent class.

        Parameters
        ----------
        neural_network : tensorflow.keras.Sequential
            Further details can be found in the documentation of the parent
            class.
        initial_layer : int, optional
            Further details can be found in the documentation of the parent
            class, by default 0.
        verbose : int, optional
            Further details can be found in the documentation of the parent
            class, by default 0.
        device : str, optional
            Device to use for the left model forwarding in path selection,
            by default None, meaning that it will use GPU if available.

        Raises
        ------
        ValueError
            - If the input model is not built.\n
            - If the input model cannot be flattened.\n
            - If the input model is not Sequential.\n
            - If there is a convolutional layer with dilation > 1
                in the right model.\n
            - If a ReLU in the path selector is leaky.\n
            - There is an unsupported layer type in the path selector. The list
                of supported layer types can be found in the notes section of
                this method.\n
            - Other errors documented in the parent class.\n
            - If the device is not valid.\n

        Notes
        -----
        Supported layer types in the path selector are: Dropout,
        SpatialDropout1D, SpatialDropout2D, SpatialDropout3D, GaussianDropout,
        GaussianNoise, AlphaDropout, ActivityRegularization, Dense, Conv1D,
        Conv2D, Conv3D, MaxPooling1D, MaxPooling2D, MaxPooling3D, Flatten,
        ReLU, PReLU, ELU, Softmax, LeakyReLU, ThresholdedReLU, Activation.

        However, note that for path selection, only ReLUs may be used as
        activation function before the last layer that affects path selection.
        """
        self._device = self._check_device_or_get_default(device)
        PathSelector.__init__(self, neural_network, initial_layer, verbose)

    def _init_layer_type_sets(self):
        """Initializes the sets of layer types."""

        # Call super method to initialize the layer type sets.
        PathSelector._init_layer_type_sets(self)

        # Add keras specific layer types.

        self.TRAIN_ONLY_LAYERS = self.TRAIN_ONLY_LAYERS | (
            self.LayerTypes.ACTIVITY_REGULARIZATION
            | self.LayerTypes.SPATIAL_DROPOUT_1D
            | self.LayerTypes.SPATIAL_DROPOUT_2D
            | self.LayerTypes.SPATIAL_DROPOUT_3D
            | self.LayerTypes.GAUSSIAN_DROPOUT
            | self.LayerTypes.GAUSSIAN_NOISE
            | self.LayerTypes.ALPHA_DROPOUT
        )
        """Layers only active when training."""

        self.SUPPORTED_ACTIVATIONS |= self.LayerTypes.ACTIVATION
        """Supported activation layers."""

        self.ALL_ACTIVATIONS |= (
            self.SUPPORTED_ACTIVATIONS
            | self.LayerTypes.PRELU
            | self.LayerTypes.ELU
            | self.LayerTypes.LEAKY_RELU
            | self.LayerTypes.THRESHOLDED_RELU
        )
        """All activation layers."""

        self.SUPPORTED_LAYERS |= (
            self.TRAIN_ONLY_LAYERS | self.PATH_LAYERS | self.ALL_ACTIVATIONS
        )
        """All layers that may be present in the path selector."""

    def _interpret_negative_layers(self, initial_layer, neural_network):
        """Interprets negative layer indices.

        Parameters
        ----------
        initial_layer : int
            Index of the initial layer.
        neural_network : tensorflow.keras.Sequential
            Neural network.

        """
        if initial_layer < 0:
            initial_layer = len(neural_network.layers) + initial_layer
        self._initial_layer = initial_layer

    def _compute_input_shapes(self):
        """Computes the input shapes of layers in the right model."""
        self._input_shapes = [
            layer.input_shape[1:] for layer in self.right_model_.layers
        ]

    def _get_number_of_outputs(self):
        """Returns the number of outputs of the right neural network.

        Returns
        -------
        int
            Number of outputs.
        """
        return np.prod(self.right_model_.output_shape[1:])

    @property
    def _output_layer_dtype(self):
        """Returns the data type of the output layer

        Returns
        -------
        str
            Data type of outputs
        """
        return self._layers[-1].dtype

    @property
    def _layers(self):
        """Returns the layers of the right neural network.

        Returns
        -------
        list
            List of layers.
        """
        if self.right_model_ is None:
            return None
        return self.right_model_.layers

    def _layer_subset_valid(self, initial_layer, neural_network):
        """Checks that path selector uses layers in bounds of neural network.

        Parameters
        ----------
        initial_layer : int
            Layer of the model where path selection starts.
        neural_network : tensorflow.keras.Sequential
            Keras sequential model.

        Returns
        -------
        bool
            True if path selector in bounds of neural network.
        """
        return 0 <= initial_layer < len(neural_network.layers)

    ##############################
    # Path selector compilation
    ##############################

    def _get_connections_conv_layer(self, layer_idx):
        """Computes all connections in a convolutional layer.

        Parameters
        ----------
        layer_idx : int
            Index of the layer where connections are computed.

        Returns
        -------
        array2d
            Array of paths.
        """
        layer = self._layers[layer_idx]
        weights = self._list_weights[layer_idx]
        biases = self._list_biases[layer_idx]

        return qs_conv.get_all_paths_conv_layer(
            input_shape=self._input_shapes[layer_idx],
            weights=weights,
            biases=biases,
            strides=layer.strides,
            padding=layer.padding,
            data_format=layer.data_format,
            groups=layer.groups,
        )

    def _get_connections_maxpool_layer(self, layer_idx):
        """Computes all connections in a maxpool layer.

        Parameters
        ----------
        layer_idx : int
            Index of the layer where connections are computed.

        Returns
        -------
        array2d
            Array of paths.
        """
        layer = self._layers[layer_idx]

        return qs_maxpool.get_all_paths_maxpool_layer(
            input_shape=self._input_shapes[layer_idx],
            pool_size=layer.pool_size,
            strides=layer.strides,
            padding=layer.padding,
            data_format=layer.data_format,
        )

    def _get_connections_dense_layer(self, layer_idx):
        """Computes all connections in a dense layer.

        Parameters
        ----------
        layer_idx : int

        Returns
        -------
        array2d
            Array of paths.
        """
        weights = self._list_weights[layer_idx]
        biases = self._list_biases[layer_idx]

        return qs_dense.get_all_paths_dense_layer(weights, biases)

    def _make_left_right_models(self, initial_layer, neural_network):
        """Creates left and right models by splitting neural network at
        initial layer.

        Parameters
        ----------
        initial_layer : int
            Layer of the model where path selection starts.
        neural_network : tensorflow.keras.Sequential
            Keras sequential model.
        """

        # if initial layer is the first layer, then the left model is None
        if initial_layer == 0:
            self.left_model_ = None
        else:
            # maybe use GPU for the left model
            self._log("Creating left model...", log_level=1)
            with tf.device(self._device):
                # Create left model with the original weights
                input_shape = neural_network.layers[0].input_shape[1:]
                self.left_model_ = kr.Sequential([kr.Input(shape=input_shape)])
                weights = []
                for layer in neural_network.layers[:initial_layer]:
                    layer_copy = type(layer).from_config(layer.get_config())
                    self.left_model_.add(layer_copy)
                    weights.append(layer.get_weights())

                # build the model
                self.left_model_.build()
                for layer, layer_weights in zip(
                    self.left_model_.layers, weights
                ):
                    layer.set_weights(layer_weights)

        # Second section of the model (right model)
        layers = []
        weights = []

        # Use CPU for the right model
        self._log("Creating right model...", log_level=1)
        with tf.device("/cpu:0"):
            # Transform parameter layers and filter out training only layers
            for layer in neural_network.layers[initial_layer:]:
                # Get layer type
                layer_name = layer.__class__.__name__
                layer_type = self.LayerTypes.name_to_type(layer_name)

                # Unkown layer type
                if layer_type is None:
                    raise ValueError(
                        f"Unsupported layer type {layer_name} in path "
                        f"selector. Supported: {self.SUPPORTED_LAYERS}"
                    )

                # Training only layers
                if layer_type in self.TRAIN_ONLY_LAYERS:
                    continue

                # Transform parameter layers to use Layer API on activations
                if layer_type in self.PARAMETER_LAYERS:
                    layers += self._transform_parameter_layer(
                        layer, layer_type
                    )
                    weights.append(layer.get_weights())
                # Other layers (maxpooling or flatten, activation layers...)
                else:
                    layers.append(layer)

                # Match weights to layers
                while len(weights) != len(layers):
                    weights.append(None)

            # Add input layer
            right_input_shape = neural_network.layers[
                initial_layer
            ].input_shape
            layers.insert(0, kr.Input(right_input_shape[1:]))

            self.right_model_ = kr.models.Sequential(layers)
            # build the model
            self.right_model_.build()

            # set weights
            for layer_idx, layer in enumerate(self.right_model_.layers):
                if weights[layer_idx] is not None:
                    layer.set_weights(weights[layer_idx])

    def _get_name_if_not_linear_activation(self, layer_idx):
        """Returns the name of the layer if it is an activation layer that is
        not relu or linear.

        Parameters
        ----------
        layer_idx : int
            Index of the layer in the right neural network.

        Returns
        -------
        str
            Name of the layer if it is an activation layer that is not relu or
            linear, None otherwise.
        """

        layer_type = self._layer_types[layer_idx]

        # skip layers that are not activation layers
        if layer_type not in self.ALL_ACTIVATIONS:
            return None

        # check if activation layer is allowed (relu or linear)
        non_linear_layer = None

        # "Activation" layer is allowed, but only if it is relu or linear
        if layer_type == self.LayerTypes.ACTIVATION:
            activation_name = self._layers[layer_idx].activation.__name__
            if activation_name not in ["relu", "linear"]:
                non_linear_layer = f"Activation({activation_name})"

        # A layer that is not supported
        elif layer_type not in self.SUPPORTED_ACTIVATIONS:
            non_linear_layer = str(layer_type)

        # A layer that is supported but with incorrect parameters
        elif layer_type == self.LayerTypes.RELU:
            layer = self._layers[layer_idx]

            # negative slope is not 0 (leaky relu)
            if layer.negative_slope != 0:
                non_linear_layer = str(layer_type)

        return non_linear_layer

    def _layer_weights(self, layer_idx):
        """Extract neural network layer weights.

        Parameters
        ----------
        layer_idx : int
            Layer number, not including layers out of path selector.

        Returns
        -------
        (arrayNd) or (None, None)
            Layer weights and biases or None, None if there are none.
        """
        # no weights or bias
        if not self._layer_types[layer_idx] in self.PARAMETER_LAYERS:
            return None, None

        layer = self._layers[layer_idx]
        parameters = layer.get_weights()

        # weights and biases defined
        if len(parameters) == 2:
            return parameters

        # no bias defined, replace with zeroes
        return parameters[0], np.zeros(parameters[0].shape[-1])

    def _transform_parameter_layer(self, layer, layer_type):
        """Transforms a parameter layer so that it doesnt have an activation
        function, if it has one inside the layer. Instead, the activation
        function is applied in a separate layer.

        Parameters
        ----------
        layer : tensorflow.keras.layers.Layer
            Layer to be transformed.
        layer_type : KerasLayerTypes
            Type of the layer.

        Returns
        -------
        tensorflow.keras.layers.Layer
            Transformed layer.
        """
        # if layer is not a parameter layer or has no activation function
        if (
            layer_type not in self.PARAMETER_LAYERS
            or layer.activation.__name__ == "linear"
        ):
            return [layer]

        # if layer is a convolutional layer check its parameters
        if layer_type in self.CONVOLUTIONAL_LAYERS:
            if (
                layer.dilation_rate is not None
                and np.ravel(layer.dilation_rate).prod() != 1
            ):
                raise ValueError(
                    "Dilated convolutions are not supported. ",
                    layer.dilation_rate,
                )

        # make new layer without activation function
        layer_config = layer.get_config()
        layer_config["activation"] = None
        new_layer = type(layer).from_config(layer_config)

        return [new_layer, kr.layers.Activation(layer.activation)]

    def _flatten_model(self, neural_network):
        """Flattens a keras sequential model.

        Parameters
        ----------
        neural_network : tensorflow.keras.Sequential
            Keras sequential model.

        Raises
        ------
        ValueError
            If the model is not sequential or not built.

        Returns
        -------
        tensorflow.keras.Sequential
            Flattened keras sequential model.
        """

        # Ensure that the model is sequential, which disallows concatenation
        # of layers and other operations not supported by the path selector
        if not isinstance(neural_network, kr.Sequential):
            raise ValueError(
                "Path selector only supports sequential models. "
                "Please ensure that your model is sequential."
            )

        # Ensure that the model is built
        if neural_network.built is False:
            raise ValueError(
                "Path selector only supports built models. "
                "Please build or compile your model before using it."
            )

        # Create an empty list to store the flattened layers
        layers = []

        def flatten_model_recursive(model):
            # Iterate over the layers of the model
            for layer in model.layers:
                # If the layer is a Sequential model, flatten it recursively
                if isinstance(layer, kr.Sequential):
                    flatten_model_recursive(layer)
                else:
                    # Otherwise, add the layer to the list of flattened layers
                    layers.append(layer)

        # Update the list of layers with the flattened layers
        flatten_model_recursive(neural_network)

        # Build the model with the same input shape as the original model
        model = kr.Sequential(layers)
        model.build(input_shape=neural_network.input_shape)

        # Return a new flattened model
        return model

    ##############################
    # Path selection
    ##############################

    @staticmethod
    def _as_numpy(X):
        """Converts X to numpy array if it is not already.

        Parameters
        ----------
        X : np.ndarray or tf.Tensor
            Input to the neural network.

        Returns
        -------
        np.ndarray
            Converted input to numpy array.
        """
        if isinstance(X, np.ndarray):
            return X
        return X.numpy()

    @staticmethod
    @tf.function(reduce_retracing=True)
    def _forward_inference(model, X):
        """Inference of model on X, in evaluation mode, so that layers like
        dropout are not applied.

        Parameters
        ----------
        model : tensorflow.keras.Sequential or tensorflow.keras.layers.Layer
            Model to be inferred.
        X : arrayNd
            Input to the neural network.

        Returns
        -------
        arrayNd
            Output of the neural network.
        """
        return model(X, training=False)

    def as_tensorflow_dataset(self, X, Y=None, batch_size=None, shuffle=False):
        """Maps samples_to_coefficients across X using the specified batch
        size. Larger batch size is faster, but may cause memory problems
        depending on the number of coefficients.

        Parameters
        ----------
        X : arrayNd
            Input to the neural network
        Y : arrayNd, optional
            Desired outputs of the neural network. If None, the tensorflow
            dataset only yields the path selection of X. By default None.
        batch_size : int or None, optional
            Size of each batch of the path selection, by default None, meaning
            use a batch size of 32.
        shuffle : bool, optional
            Whether to shuffle the dataset, by default False.
        """

        out_dtype = self._output_layer_dtype
        in_dtype = self.right_model_.inputs[0].dtype

        with tf.device("/cpu:0"):  # no copy conversion to tensor
            X_tensor = tf.convert_to_tensor(X, in_dtype)

        # auxiliar function
        def path_select(X):
            return self.samples_to_coefficients(
                tf.convert_to_tensor(X)
            ).astype(out_dtype, copy=False)

        # use default batch size if not specified
        batch_size = batch_size or DEFAULT_BATCH_SIZE

        # output or no output variations
        if Y is not None:
            with tf.device("/cpu:0"):
                Y_tensor = tf.convert_to_tensor(Y)

            def tf_path_select(indices):
                X_batch = tf.gather(X_tensor, indices)
                Y_batch = tf.gather(Y_tensor, indices)

                return (
                    tf.numpy_function(
                        path_select, [X_batch], out_dtype, False
                    ),
                    Y_batch,
                )

        else:

            def tf_path_select(indices):
                X_batch = tf.gather(X_tensor, indices)
                return tf.numpy_function(
                    path_select, [X_batch], out_dtype, False
                )

        path_selection_dataset = tf.data.Dataset.range(len(X))

        if shuffle:
            path_selection_dataset = path_selection_dataset.shuffle(
                buffer_size=len(X), reshuffle_each_iteration=True
            )

        # create dataset (path selection is lazy)
        return path_selection_dataset.batch(batch_size).map(
            tf_path_select, deterministic=True, num_parallel_calls=6
        )

    def as_numpy_iterator(self, X, Y=None, batch_size=None):
        """Maps samples_to_coefficients across X using the specified batch
        size. Larger batch size is faster, but may cause memory problems
        depending on the number of coefficients.

        Parameters
        ----------
        X : arrayNd
            Input to the neural network
        Y : arrayNd, optional
            Desired outputs of the neural network. If None, the tensorflow
            dataset only yields the path selection of X. By default None.
        batch_size : int or None, optional
            Size of each batch of the path selection, by default None, meaning
            use a batch size of 32.

        Returns
        -------
        typing.Iterator
            Iterator over the path selection of X.
        """
        tf_dataset = self.as_tensorflow_dataset(
            X, Y, batch_size, shuffle=False
        )
        return tf_dataset.as_numpy_iterator()

    ##############################
    # Persistence methods
    ##############################

    def __getstate__(self):
        """Defines which variables are to be pickled when called with pickle
        and saves the rest with specific methods, e.g. keras clone_model.

        Note
        ----
        The directory where numpy and keras variables are stored is taken
        from the internal attribute _save_dir, which will be initialized
        if not previously set (normally when calling save method).

        Returns
        -------
        dict
            Variables to be pickled with python pickle module.
        """
        # capture what is normally pickled
        state = NumpyPersistanceMixin.__getstate__(self)

        # if there is no neural network, we are finished
        if self.right_model_ is None:
            return state

        # save neural network in a directory and dont try to persist it
        # with pickle
        if self.left_model_ is not None:
            state["left_model_"] = os.path.join(
                self._save_dir, "left_model.h5"
            )
            # ignore warnings about model not being compiled
            with suppress_tensorflow_warnings():
                self.left_model_.save(state["left_model_"])

        state["right_model_"] = os.path.join(self._save_dir, "right_model.h5")
        # ignore warnings about model not being compiled
        with suppress_tensorflow_warnings():
            self.right_model_.save(state["right_model_"])

        return state

    @classmethod
    def load(cls, directory_path, device=None):
        """Load KerasPathSelector from directory.

        Parameters
        ----------
        directory_path : str
            Path to directory.
        device : str, optional
            Device to use for loading the model, by default None, meaning use
            the default device of tensorflow in the current environment.

        Returns
        -------
        KerasPathSelector
            KerasPathSelector instance loaded from directory.

        Raises
        ------
        FileNotFoundError
            If the directory does not exist.
        ValueError
            If the device is not valid.
        """
        instance = PathSelector.load(directory_path)
        device = cls._check_device_or_get_default(device)
        instance._device = device

        # Currently left_model_ is a string path to the model
        if instance.left_model_ is not None:
            with suppress_tensorflow_warnings():
                with tf.device(device):
                    instance.left_model_ = kr.models.load_model(
                        instance.left_model_
                    )
        # Currently right_model_ is a string path to the model
        if instance.right_model_ is not None:
            with suppress_tensorflow_warnings():
                with tf.device("/cpu:0"):
                    instance.right_model_ = kr.models.load_model(
                        instance.right_model_
                    )

        return instance


@contextlib.contextmanager
def suppress_tensorflow_warnings():
    """Context manager to suppress tensorflow warnings.

    Yields
    ------
    None
    """
    # Get the TensorFlow logger
    logger = tf.get_logger()

    # Store the original logging level
    original_level = logger.getEffectiveLevel()

    try:
        # Set the logging level to suppress warnings
        logger.setLevel(tf.compat.v1.logging.ERROR)
        yield
    finally:
        # Restore the original logging level
        logger.setLevel(original_level)


class KerasLayerTypes:
    """
    As IntFlag is not inheritable, we need to create a new class to
    automatically set the framework for the BaseLayerTypes class,
    which allows us to redefine the __str__ method and how an class name
    is mapped to a layer type in keras.
    """

    @staticmethod
    def type_to_name(layer_type):
        """Returns the keras layer name for a given layer type

        Parameters
        ----------
        layer_type : KerasLayerTypes
            The layer type for which the keras layer name should be returned.

        Returns
        -------
        str
            The keras layer name for the given layer type.
        """
        return LAYER_TYPE_TO_KERAS_LAYER_NAME.get(layer_type)

    @staticmethod
    def name_to_type(layer_name):
        """Returns the layer type for a given keras layer name

        Parameters
        ----------
        layer_name : str
            The keras layer name for which the layer type should be returned.

        Returns
        -------
        KerasLayerTypes
            The layer type for the given keras layer name.
        """
        return KERAS_LAYER_NAME_TO_LAYER_TYPE.get(layer_name)


LAYER_TYPE_TO_KERAS_LAYER_NAME = {
    BaseLayerTypes.DROPOUT: "Dropout",
    BaseLayerTypes.SPATIAL_DROPOUT_1D: "SpatialDropout1D",
    BaseLayerTypes.SPATIAL_DROPOUT_2D: "SpatialDropout2D",
    BaseLayerTypes.SPATIAL_DROPOUT_3D: "SpatialDropout3D",
    BaseLayerTypes.GAUSSIAN_DROPOUT: "GaussianDropout",
    BaseLayerTypes.GAUSSIAN_NOISE: "GaussianNoise",
    BaseLayerTypes.ALPHA_DROPOUT: "AlphaDropout",
    BaseLayerTypes.ACTIVITY_REGULARIZATION: "ActivityRegularization",
    BaseLayerTypes.DENSE: "Dense",
    BaseLayerTypes.CONV_1D: "Conv1D",
    BaseLayerTypes.CONV_2D: "Conv2D",
    BaseLayerTypes.CONV_3D: "Conv3D",
    BaseLayerTypes.MAXPOOLING_1D: "MaxPooling1D",
    BaseLayerTypes.MAXPOOLING_2D: "MaxPooling2D",
    BaseLayerTypes.MAXPOOLING_3D: "MaxPooling3D",
    BaseLayerTypes.FLATTEN: "Flatten",
    BaseLayerTypes.RELU: "ReLU",
    BaseLayerTypes.PRELU: "PReLU",
    BaseLayerTypes.ELU: "ELU",
    BaseLayerTypes.SOFTMAX: "Softmax",
    BaseLayerTypes.LEAKY_RELU: "LeakyReLU",
    BaseLayerTypes.THRESHOLDED_RELU: "ThresholdedReLU",
    BaseLayerTypes.ACTIVATION: "Activation",
}
"""Maps layer types to keras layer names."""

for name, member in BaseLayerTypes.__members__.items():
    member.with_type_to_name_dict(LAYER_TYPE_TO_KERAS_LAYER_NAME)
    setattr(KerasLayerTypes, name, member)


KERAS_LAYER_NAME_TO_LAYER_TYPE = {
    v: getattr(KerasLayerTypes, k.name)
    for k, v in LAYER_TYPE_TO_KERAS_LAYER_NAME.items()
}
"""Maps keras layer names to layer types."""
