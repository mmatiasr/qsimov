"""Contains functionality related to path selection tasks on pytorch neural
networks.
"""
import numpy as np
from qsimov.mixins import NumpyPersistanceMixin
import os
from qsimov.path_selector import (
    DEFAULT_BATCH_SIZE,
    PathSelector,
    BaseLayerTypes,
)
import torch
import torch.nn as nn
import qsimov.paths.conv as qs_conv
import qsimov.paths.maxpooling as qs_maxpool
import qsimov.paths.dense as qs_dense
import warnings
from torch.utils.data import DataLoader, TensorDataset


class PytorchPathSelector(PathSelector):
    """Implements path selection for a subset of layers of a trained pytorch
    model. Check the parent class PathSelector for further details.

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
        PytorchLayerTypes
            Subclass of BaseLayerTypes class.
        """
        return PytorchLayerTypes

    @staticmethod
    def _check_device_or_get_default(device=None):
        """Checks if the device is valid or returns the default device.

        Parameters
        ----------
        device : torch.device or str, optional
            Device on which the left propagation should be performed. The
            default is None, which means that gpu is used if available.

        Returns
        -------
        torch.device
            Device on which the left propagation should be performed.

        Raises
        ------
        ValueError
            If the device is not valid.
        """
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if type(device) == str:
            return torch.device(device)
        if type(device) == torch.device:
            return device
        raise ValueError(
            "Device must be a torch.device or a string with the device name."
        )

    ############################
    # Class initialization
    ############################

    def __init__(
        self,
        neural_network,
        input_shape,
        initial_layer=0,
        verbose=0,
        device=None,
    ):
        """Implements path selection for a subset of layers of a trained
        pytorch model. Further details can be found in the parent class.


        Parameters
        ----------
        neural_network : torch.nn.Sequential
            As documented in the parent class.
        input_shape : int or tuple
            Integer or tuple with the shape of input data without the batch
            size, e.g. if the first layer is conv2d the tuple has to be
            (C, H, W), where C is the number of input channels, H is the
            height of input planes in pixels and W is the width of input
            planes in pixels.
        initial_layer : int, optional
            As documented in the parent class. The default is 0.
        verbose : int, optional
            As documented in the parent class. The default is 0.
        device : torch.device or str, optional
            Device on which the left propagation should be performed. The
            default is None, which means that gpu is used if available.

        Raises
        ------
        ValueError
            - If the input shape is not an integer or tuple.\n
            - If the input shapes of the right model cannot be inferred.\n
            - If the input model is not a torch.nn.Sequential.\n
            - If the right model has an unsupported layer type. Supported layer
              types are docummented in this methods notes.\n
            - If there is a convolutional layer with dilation > 1
              or padding mode not 'zeros' in the right model.\n
            - Other errors documented in the parent class.\n

        RuntimeError
            - If the dtype of the model cannot be inferred. Trials are done
              with float16 float32 and float64.\n

        Notes
        -----
        Supported layer types are: Dropout, Dropout1d, Dropout2d, Dropout3d,
        AlphaDropout, FeatureAlphaDropout, Linear, Identity, Conv1d, Conv2d,
        Conv3d, MaxPool1d, MaxPool2d, MaxPool3d, Flatten, ReLU, ReLU6,
        LeakyReLU, ELU, PReLU, Hardshrink, Hardsigmoid, Tanh, Hardtanh,
        Hardswish, Sigmoid, LogSigmoid, Threshold, Softmin, Softmax,
        Softmax2d, LogSoftmax.

        However only the ReLU or Identity activation functions are supported
        before the last layer that affects path selection.
        """
        # Check input shape
        self._check_input_shape(input_shape)

        self._device = self._check_device_or_get_default(device)

        PathSelector.__init__(self, neural_network, initial_layer, verbose)

    def _init_layer_type_sets(self):
        """Initializes the sets of layer types."""

        # Call super method to initialize the layer type sets.
        PathSelector._init_layer_type_sets(self)

        # Add pytorch specific layer types.

        self.TRAIN_ONLY_LAYERS = self.TRAIN_ONLY_LAYERS | (
            self.LayerTypes.SPATIAL_DROPOUT_1D
            | self.LayerTypes.SPATIAL_DROPOUT_2D
            | self.LayerTypes.SPATIAL_DROPOUT_3D
            | self.LayerTypes.ALPHA_DROPOUT
            | self.LayerTypes.FEATURE_ALPHA_DROPOUT
        )
        """Layers only active when training."""

        self.SUPPORTED_ACTIVATIONS = (
            self.SUPPORTED_ACTIVATIONS | self.LayerTypes.IDENTITY
        )
        """Supported activation layers for path selection."""

        self.ALL_ACTIVATIONS = self.ALL_ACTIVATIONS | (
            self.LayerTypes.IDENTITY
            | self.LayerTypes.RELU_6
            | self.LayerTypes.LEAKY_RELU
            | self.LayerTypes.ELU
            | self.LayerTypes.PRELU
            | self.LayerTypes.HARD_SHRINK
            | self.LayerTypes.HARD_SIGMOID
            | self.LayerTypes.TANH
            | self.LayerTypes.HARD_TANH
            | self.LayerTypes.HARD_SWISH
            | self.LayerTypes.SIGMOID
            | self.LayerTypes.LOGSIGMOID
            | self.LayerTypes.THRESHOLDED_RELU
            | self.LayerTypes.SOFTMIN
            | self.LayerTypes.SOFTMAX_2D
            | self.LayerTypes.LOGSOFTMAX
        )
        """All activation layers."""

        self.SUPPORTED_LAYERS |= (
            self.TRAIN_ONLY_LAYERS | self.PATH_LAYERS | self.ALL_ACTIVATIONS
        )
        """All layers that may be present in the path selector."""

    def _interpret_negative_layers(self, initial_layer, neural_network):
        """Returns the positive initial layer when the initial layer is
        negative.

        Parameters
        ----------
        initial_layer : int
            Initial layer index.
        neural_network : torch.nn.Sequential
            Neural network.

        Returns
        -------
        int
            Positive initial layer index.
        """
        if initial_layer < 0:
            initial_layer = (
                len(list(neural_network.children())) + initial_layer
            )
        self._initial_layer = initial_layer

    def _check_input_shape(self, input_shape):
        """Checks if the input shape introduced is correct

        Parameters
        ----------
        input_shape : int or tuple
            Integer or tuple with the shape of input data without the batch
            size, e.g. if the first layer is conv2d the tuple has to be
            (C, H, W), where C is the number of input channels, H is the
            height of input planes in pixels and W is the width of input
            planes in pixels.

        Raises
        ------
        ValueError
            If the input shape cannot be converted to an integer tuple.
        """
        self._input_shape = input_shape
        try:
            self._input_shape = np.array([self._input_shape], int).ravel()
            self._input_shape = tuple(self._input_shape)
        except TypeError:
            raise ValueError(
                "Input shape should be an integer iterable or integer"
            )

    def _compute_input_shapes(self):
        """Computes the input shapes of layers in the right model, and
        sets them in the attribute _input_shapes.

        Raises
        ------
        ValueError
            If the input shape is not compatible with the model.
        RuntimeError
            If no dtype compatible with the model is found (trials with
            torch.float32, torch.float64 and torch.half).
        """
        trial_dtypes = [torch.float32, torch.float64, torch.half]
        for dtype in trial_dtypes:
            try:
                # set guess dtype of the first layer
                self._left_model_input_dtype = dtype

                # create a sample input tensor with the desired shape
                sample_input = torch.randn(
                    self._input_shape, dtype=dtype, device=self._device
                )
                sample_input = sample_input.unsqueeze(0)

                # propagate input up to inputs of right model
                if self.left_model_ is not None:
                    for layer in list(self.left_model_.children()):
                        sample_input = layer(sample_input)

                # append input shape of the first layer
                self._input_shapes = [tuple(sample_input.shape)[1:]]

                # currently right model only uses cpu
                sample_input = sample_input.cpu()

                # set guess dtype of the last layer of the left model
                self._right_model_input_dtype = sample_input.dtype

                for layer in list(self.right_model_.children()):
                    # pass the input tensor through the layer and
                    # get the shape of the output tensor
                    sample_input = layer(sample_input)
                    output_shape = tuple(sample_input.shape)[1:]
                    self._input_shapes.append(output_shape)

                # get the data type of the output
                self._right_model_output_dtype = sample_input.dtype

                # remove output shape of the last layer
                self._number_outputs = np.prod(self._input_shapes[-1])
                self._input_shapes = self._input_shapes[:-1]
                return

            except RuntimeError as e:
                # check if error is due to input shape
                if "shape" in str(e) or "size" in str(e):
                    raise ValueError(
                        "Provided input shape does not match the given model"
                    )
                continue
        raise RuntimeError(
            "Could not determine input dtype of the model. After trying "
            "the following supported dtypes: {}".format(trial_dtypes)
        )

    def _get_number_of_outputs(self):
        """Returns the number of outputs of the right neural network.

        Returns
        -------
        int
            Number of outputs.
        """
        if self._number_outputs is None:
            self._compute_input_shapes()
        return self._number_outputs

    @property
    def _output_layer_dtype(self):
        """Returns the data type of the output layer.

        Returns
        -------
        str
            Data type of outputs.
        """
        return self._right_model_output_dtype

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
        return list(self.right_model_.children())

    def _layer_subset_valid(self, initial_layer, neural_network):
        """Checks that path selector uses layers in bounds of neural network.

        Parameters
        ----------
        initial_layer : int
            Layer of the model where path selection starts.
        neural_network : torch.nn.Sequential
            Pytorch sequential model.

        Returns
        -------
        bool
            True if path selector in bounds of neural network.
        """
        return 0 <= initial_layer < len(list(neural_network.children()))

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
            strides=layer.stride,
            padding=layer.padding,
            data_format="channels_first",
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
            pool_size=layer.kernel_size,
            strides=layer.stride,
            padding=layer.padding,
            data_format="channels_first",
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

        return qs_dense.get_all_paths_dense_layer(weights.T, biases)

    def _check_layer_parameters(self, layer):
        """Checks if the attributes of a layer are supported.

        Parameters
        ----------
        layer : torch.nn.Module
            Layer of the neural network.

        Returns
        -------
        torch.nn.Module
            Layer of the neural network.

        Raises
        ------
        ValueError
            If layer is not supported.
        """
        layer_type = self.LayerTypes.name_to_type(layer.__class__.__name__)

        if layer_type in self.CONVOLUTIONAL_LAYERS:
            if (
                layer.dilation is not None
                and np.ravel(layer.dilation).prod() != 1
            ):
                raise ValueError(
                    "Dilated convolutions are not supported. ", layer.dilation
                )
            if layer.padding_mode != "zeros":
                raise ValueError(
                    "Padding mode {} is not supported".format(
                        layer.padding_mode
                    )
                )

        return layer

    def _make_left_right_models(self, initial_layer, neural_network):
        """Creates left and right models by splitting neural network at
        initial layer.

        Parameters
        ----------
        initial_layer : int
            Layer of the model where path selection starts.
        neural_network : nn.Sequential
            Pytorch sequential model.
        """
        # If initial layer is the first layer, then the left model is None
        if initial_layer == 0:
            self.left_model_ = None
        else:
            # Create left model with the original weights
            self.left_model_ = nn.Sequential(
                *list(neural_network.children())[:initial_layer]
            )
            self.left_model_.eval()
            self.left_model_.to(self._device)
            self.left_model_.current_device = self._device

        # Second section of the model (right model)
        layers = []

        # Transform parameter layers and filter out training only layers
        for layer in list(neural_network.children())[initial_layer:]:
            # Get layer type
            layer_name = layer.__class__.__name__
            layer_type = self.LayerTypes.name_to_type(layer_name)

            # Unkown layer type
            if layer_type is None:
                raise ValueError(
                    f"Unsupported layer type {layer_name} in path selector."
                    f"Supported: {self.SUPPORTED_LAYERS}"
                )

            # Training only layers
            if layer_type in self.TRAIN_ONLY_LAYERS:
                continue

            # Append layers (convolutional, dense, maxpooling or
            # flatten, activation layers...)
            layers.append(self._check_layer_parameters(layer))

        self.right_model_ = nn.Sequential(*layers)

        # evaluation mode
        self.right_model_.eval()

        # as currently samples to coefficients is always done on cpu
        # it is convenient to also have the right model on cpu
        self.right_model_.to("cpu")
        self.right_model_.current_device = "cpu"

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

        # A layer that is not supported
        if layer_type not in self.SUPPORTED_ACTIVATIONS:
            return str(layer_type)

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

        weight = layer.weight.detach().numpy()
        # no bias defined, replace with zeroes
        if layer.bias is None:
            bias = np.zeros(weight.shape[0])
        else:
            bias = layer.bias.detach().numpy()

        return weight, bias

    def _flatten_model(self, neural_network):
        """Flattens a pytorch sequential model.

        Parameters
        ----------
        neural_network : nn.Sequential
            Pytorch sequential model.

        Raises
        ------
        ValueError
            If the model is not sequential.

        Returns
        -------
        torch.nn.Sequential
            Flattened pytorch sequential model.

        Notes
        -----
        The path selector only supports sequential models. However, it is
        allowed to have other models inside the sequential model.
        """
        # Ensure that the model is sequential, which disallows concatenation
        # of layers and other operations not supported by the path selector
        if not isinstance(neural_network, nn.Sequential):
            raise ValueError(
                "Path selector only supports sequential models. "
                "Please flatten your model before using it."
            )

        # Check if the model is compiled
        if hasattr(neural_network, "_torchdynamo_orig_callable"):
            warnings.warn(
                "The model introduced is compiled, but currently the path"
                " selector does not support compiled models,"
                " it will be converted to a non-compiled model."
            )

        # Create an empty list to store the flattened layers
        layers = []

        def flatten_model_recursive(model):
            # Iterate over the layers of the model
            for layer in list(model.children()):
                # If the layer is a Sequential model, flatten it recursively
                if isinstance(layer, nn.Sequential):
                    flatten_model_recursive(layer)
                else:
                    # Otherwise, add the layer to the list of flattened layers
                    layers.append(layer)

        # Update the list of layers with the flattened layers
        flatten_model_recursive(neural_network)

        # Make the model with the same configuration as the original model
        model = nn.Sequential(*layers)

        # Return a new flattened model
        return model

    ##############################
    # Path selection
    ##############################

    @staticmethod
    def _as_numpy(X):
        """Converts X to numpy array.

        Parameters
        ----------
        X : array or torch.tensor
            Input to the neural network.

        Returns
        -------
        arrayNd
            Numpy array with the same data as X.
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return X

    @staticmethod
    def _forward_inference(model, X):
        """Inference of model on X, in evaluation mode, so that layers like
        dropout are not applied.

        Parameters
        ----------
        model : torch.nn.Module
            Model to be inferred.
        X : array or torch.tensor
            Input to the neural network.

        Returns
        -------
        arrayNd
            Output of the neural network.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).requires_grad_(False)
        if hasattr(model, "current_device"):
            X = X.to(model.current_device)
        else:
            X = X.cpu()
        return model(X).detach()

    def as_pytorch_dataloader(self, X, Y=None, batch_size=None, shuffle=False):
        """Creates a pytorch dataset from the path selection of X and Y.

        Parameters
        ----------
        X : arrayNd
            Input to the neural network.
        Y : arrayNd, optional
            Desired outputs of the neural network. If None, the pytorch
            dataset only yields the path selection of X. By default None.
        batch_size : int or None, optional
            Batch size of the pytorch dataset. If None, the default batch size
            DEFAULT_BATCH_SIZE is used. By default None.
        shuffle : bool, optional
            Whether to shuffle the data. By default False.

        Returns
        -------
        torch.utils.data.DataLoader
            Pytorch dataloader over the path selection of X and Y.
        """
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE
        X_tensor = torch.as_tensor(X, dtype=self._left_model_input_dtype)

        # Create a dataset from the samples, and a collate function to
        # transform the samples into coefficients.
        # Take into account that the dataset can be only X or X and Y
        if Y is not None:
            dataset = TensorDataset(
                X_tensor,
                torch.as_tensor(Y, dtype=self._right_model_output_dtype),
            )

            def collate_fn(batch):
                X, Y = zip(*batch)
                return torch.as_tensor(
                    self.samples_to_coefficients(torch.stack(X, 0)),
                    dtype=self._right_model_input_dtype,
                ), torch.stack(Y, 0)

        else:
            dataset = TensorDataset(X_tensor)

            def collate_fn(batch):
                (X,) = zip(*batch)
                return torch.as_tensor(
                    self.samples_to_coefficients(torch.stack(X, 0)),
                    dtype=self._right_model_input_dtype,
                )

        # Create a dataloader from the dataset
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
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

        # Auxiliary function to convert to numpy, depending on whether Y is
        # None or not
        def to_numpy(batch):
            if Y is None:
                return self._as_numpy(batch)
            return self._as_numpy(batch[0]), self._as_numpy(batch[1])

        return map(
            to_numpy,
            self.as_pytorch_dataloader(X, Y, batch_size=batch_size).__iter__(),
        )

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
                self._save_dir, "left_model.pt"
            )
            torch.save(self.left_model_, state["left_model_"])

        state["right_model_"] = os.path.join(self._save_dir, "right_model.pt")
        torch.save(self.right_model_, state["right_model_"])

        return state

    @classmethod
    def load(cls, directory_path, device=None):
        """Load PytorchPathSelector from directory.

        Parameters
        ----------
        directory_path : str
            Path to directory.
        device : torch.device or str, optional
            Device to use for loading the model, by default None, meaning use
            the GPU if available.

        Returns
        -------
        PytorchPathSelector
            PytorchPathSelector instance loaded from directory.

        Raises
        ------
        ValueError
            If device is not valid.
        FileNotFoundError
            If directory_path does not exist.
        """
        instance = PathSelector.load(directory_path)
        instance._device = cls._check_device_or_get_default(device)

        # Currently left_model_ is a string path to the model
        if instance.left_model_ is not None:
            instance.left_model_ = torch.load(instance.left_model_)
            instance.left_model_.eval()
            instance.left_model_.to(instance._device)
            instance.left_model_.current_device = instance._device
        # Currently right_model_ is a string path to the model
        if instance.right_model_ is not None:
            instance.right_model_ = torch.load(instance.right_model_)
            instance.right_model_.eval()
            instance.right_model_.to("cpu")

        return instance


class PytorchLayerTypes:
    """
    As IntFlag is not inheritable, we need to create a new class to
    automatically set the framework for the BaseLayerTypes class,
    which allows us to redefine the __str__ method and how an class name
    is mapped to a layer type in pytorch.
    """

    @staticmethod
    def type_to_name(layer_type):
        """Returns the pytorch layer name for a given layer type

        Parameters
        ----------
        layer_type : PytorchLayerTypes
            The layer type for which the pytorch layer name should be returned.

        Returns
        -------
        str
            The pytorch layer name for the given layer type.
        """
        return LAYER_TYPE_TO_PYTORCH_LAYER_NAME.get(layer_type)

    @staticmethod
    def name_to_type(layer_name):
        """Returns the layer type for a given pytorch layer name

        Parameters
        ----------
        layer_name : str
            The pytorch layer name for which the layer type should be returned.

        Returns
        -------
        PytorchLayerTypes
            The layer type for the given pytorch layer name.
        """
        return PYTORCH_LAYER_NAME_TO_LAYER_TYPE.get(layer_name)


LAYER_TYPE_TO_PYTORCH_LAYER_NAME = {
    BaseLayerTypes.DROPOUT: "Dropout",
    BaseLayerTypes.SPATIAL_DROPOUT_1D: "Dropout1d",
    BaseLayerTypes.SPATIAL_DROPOUT_2D: "Dropout2d",
    BaseLayerTypes.SPATIAL_DROPOUT_3D: "Dropout3d",
    BaseLayerTypes.ALPHA_DROPOUT: "AlphaDropout",
    BaseLayerTypes.FEATURE_ALPHA_DROPOUT: "FeatureAlphaDropout",
    BaseLayerTypes.DENSE: "Linear",
    BaseLayerTypes.IDENTITY: "Identity",
    BaseLayerTypes.CONV_1D: "Conv1d",
    BaseLayerTypes.CONV_2D: "Conv2d",
    BaseLayerTypes.CONV_3D: "Conv3d",
    BaseLayerTypes.MAXPOOLING_1D: "MaxPool1d",
    BaseLayerTypes.MAXPOOLING_2D: "MaxPool2d",
    BaseLayerTypes.MAXPOOLING_3D: "MaxPool3d",
    BaseLayerTypes.FLATTEN: "Flatten",
    BaseLayerTypes.RELU: "ReLU",
    BaseLayerTypes.RELU_6: "ReLU6",
    BaseLayerTypes.LEAKY_RELU: "LeakyReLU",
    BaseLayerTypes.ELU: "ELU",
    BaseLayerTypes.PRELU: "PReLU",
    BaseLayerTypes.HARD_SHRINK: "Hardshrink",
    BaseLayerTypes.HARD_SIGMOID: "Hardsigmoid",
    BaseLayerTypes.TANH: "Tanh",
    BaseLayerTypes.HARD_TANH: "Hardtanh",
    BaseLayerTypes.HARD_SWISH: "Hardswish",
    BaseLayerTypes.SIGMOID: "Sigmoid",
    BaseLayerTypes.LOGSIGMOID: "LogSigmoid",
    BaseLayerTypes.THRESHOLDED_RELU: "Threshold",
    BaseLayerTypes.SOFTMIN: "Softmin",
    BaseLayerTypes.SOFTMAX: "Softmax",
    BaseLayerTypes.SOFTMAX_2D: "Softmax2d",
    BaseLayerTypes.LOGSOFTMAX: "LogSoftmax",
}
"""Maps layer types to pytorch layer names."""

# Mock inheritance of BaseLayerTypes class to PytorchLayerTypes class.
for name, member in BaseLayerTypes.__members__.items():
    member.with_type_to_name_dict(LAYER_TYPE_TO_PYTORCH_LAYER_NAME)
    setattr(PytorchLayerTypes, name, member)

# Create reverse mapping of LAYER_TYPE_TO_PYTORCH_LAYER_NAME.
PYTORCH_LAYER_NAME_TO_LAYER_TYPE = {
    v: getattr(PytorchLayerTypes, k.name)
    for k, v in LAYER_TYPE_TO_PYTORCH_LAYER_NAME.items()
}
"""Maps pytorch layer names to layer types."""
