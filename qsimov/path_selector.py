"""Contains functionality related to path selection tasks on keras neural
networks.
"""
import numpy as np
from qsimov.mixins import LogMixin, NumpyPersistanceMixin
import qsimov.paths.combine as qs_comb
import qsimov.paths.paths as qs_paths
import qsimov.paths.maxpooling as qs_maxpool
from enum import IntFlag, auto
from abc import ABC, abstractmethod


"""Default batch size for tensorflow dataset when None."""
DEFAULT_BATCH_SIZE = 32


class BaseLayerTypes(IntFlag):
    # Training only layers
    DROPOUT = auto()
    SPATIAL_DROPOUT_1D = auto()
    SPATIAL_DROPOUT_2D = auto()
    SPATIAL_DROPOUT_3D = auto()
    GAUSSIAN_DROPOUT = auto()
    GAUSSIAN_NOISE = auto()
    ALPHA_DROPOUT = auto()
    FEATURE_ALPHA_DROPOUT = auto()
    ACTIVITY_REGULARIZATION = auto()

    # Layers with weights
    DENSE = auto()
    CONV_1D = auto()
    CONV_2D = auto()
    CONV_3D = auto()

    # Maxpooling layers
    MAXPOOLING_1D = auto()
    MAXPOOLING_2D = auto()
    MAXPOOLING_3D = auto()

    # Activation layers
    ACTIVATION = auto()
    RELU = auto()
    SOFTMAX = auto()
    LEAKY_RELU = auto()
    PRELU = auto()
    ELU = auto()
    THRESHOLDED_RELU = auto()
    RELU_6 = auto()
    HARD_SHRINK = auto()
    HARD_SIGMOID = auto()
    TANH = auto()
    HARD_TANH = auto()
    HARD_SWISH = auto()
    SIGMOID = auto()
    LOGSIGMOID = auto()
    SOFTMIN = auto()
    SOFTMAX_2D = auto()
    LOGSOFTMAX = auto()

    # Other layers
    FLATTEN = auto()
    IDENTITY = auto()

    def to_list(self):
        """Converts a union of flags to a list of flags.

        Returns
        -------
        list
            List of flags.
        """
        return [flag for flag in self.type_to_name_dict.keys() if flag in self]

    def with_type_to_name_dict(self, type_to_name_dict):
        """Sets the type to name dictionary for string representation.

        Parameters
        ----------
        type_to_name_dict : dict
            Dictionary that maps layer types to names.

        Returns
        -------
        BaseLayerTypes
            Self.
        """
        self.type_to_name_dict = type_to_name_dict
        return self

    def __str__(self):
        """Returns a string representation of the flags.

        Returns
        -------
        str
            String representation of the flags.
        """
        return ", ".join(
            [self.type_to_name_dict[flag] for flag in self.to_list()]
        )

    def __contains__(self, other):
        """Returns whether the other flag is contained in the current flag.

        Parameters
        ----------
        other : BaseLayerTypes
            Other flag.

        Returns
        -------
        bool
            Whether the other flag is contained in the current flag.
        """
        return other is not None and super().__contains__(other)

    def __invert__(self):
        """Returns the inverse of the current flag.

        Returns
        -------
        BaseLayerTypes
            Inverse of the current flag.
        """
        return BaseLayerTypes(super().__invert__()).with_type_to_name_dict(
            self.type_to_name_dict
        )


# contains python dunders for logical operators [__or__, __ror__, ...]
logical_operations = []
for op in ["or", "and", "xor"]:
    for prefix in ["", "r"]:
        logical_operations.append(f"__{prefix}{op}__")


def _create_new_base_layer_types_method(op):
    """Creates a new method for BaseLayerTypes that overrides the given
    operator to include the type_to_name_dict attribute.

    Parameters
    ----------
    op : str
        Name of the operator to override.
    """

    def method(self, other):
        type_to_name_dict = getattr(self, "type_to_name_dict")
        return BaseLayerTypes(
            getattr(IntFlag, op)(self, other)
        ).with_type_to_name_dict(type_to_name_dict)

    return method


# monkey path or, and, xor operators of BaseLayerTypes
# to include the type_to_name_dict attribute on the result
for op in logical_operations:
    setattr(BaseLayerTypes, op, _create_new_base_layer_types_method(op))


def _flatten_and_insert_one(inputs):
    """Flattens an array and inserts a 1 at the beginning.

    Parameters
    ----------
    inputs : arrayNd
        Array to be transformed.

    Returns
    -------
    array1d
        Copy of array flattened and with an additional 1 at the beginning.
    """
    return np.insert(np.reshape(inputs, (inputs.shape[0], -1)), 0, 1, axis=1)


class PathSelector(ABC, LogMixin, NumpyPersistanceMixin):
    """Implements path selection for a subset of layers of a trained keras
    model.

    Attributes
    ----------
    output_masks_ : array2d
        Boolean indicating which coefficients correspond to each output
        in the coefficients obtained by a call to samples_to_coefficients.
        It is meant to be used as a mask for the coefficients returned by
        samples_to_coefficients.
    left_model_ : torch.nn.Sequential or tensorflow.keras.Sequential
        Model used to propagate the inputs until the specified initial layer.
    right_model_ : torch.nn.Sequential or tensorflow.keras.Sequential
        Model used to propagate the inputs from the specified initial layer
        until the end. Path selection is performed on this model.
    """

    ############################
    # Class variables
    ############################

    @property
    def LayerTypes(self):
        """Returns the BaseLayerTypes subclass for the path selector.

        Returns
        -------
        BaseLayerTypes
            BaseLayerTypes subclass implementing __str__ method and
            transformation from string to flag.
        """
        pass

    # Variables to be persisted using np.savez_compressed
    _NUMPY_VARIABLES = [
        "_all_paths",
        "_all_paths_input_neurons",
        "_layer_connections",
        "_list_weights",
        "_list_biases",
        "_partial_to_full_idxs",
        "_zero_to_zero_counts",
        "output_masks_",
    ]

    ############################
    # Class initialization
    ############################

    def __init__(self, neural_network, initial_layer=0, verbose=1):
        """Implements functionality common to all path selectors in the
        different frameworks. It is an abstract class, so it cannot be
        instantiated.

        Parameters
        ----------
        neural_network : torch.nn.Sequential or tensorflow.keras.Sequential
            Sequential model. It will be split into two models, one for the
            layers before the initial layer and one for the layers after the
            initial layer. Path selection will be performed on the second
            model. The left model may have any type of layers, but the right
            model must have at least one Dense or Convolutional layer. The
            rest of allowed layers in the right model are specified in
            the subclasses of PathSelector, as they depend on the used
            framework.
        initial_layer : int, optional
            Layer of the model where path selection starts, may be negative to
            indicate start to count from the end, e.g. -1 is last layer. By
            default 0.
        verbose : int, optional
            Degree of verbosity, by default 1, meaning some logs are printed,
            including the number of paths for the given initial layer and
            model.

        Raises
        ------
        ValueError
            When incorrect parameters are used, may be due to:\n
                - Invalid initial layer (out of bounds)\n
                - No Dense or Convolutional layers in the right model\n
                - There is an activation layer in the right model that is not
                  preceded by a layer that affects path selection (maxpooling,
                  convolutional, dense)\n
                - There is an activation layer in the right model previous to
                  the last layer that affects path selection (maxpooling,
                  convolutional, dense) that is not linear or ReLU\n
                - There is a layer in the right model that is not supported
                  by the framework\n

        Notes
        -----
        The neural network is flattened before splitting it into two models.
        This means that nested Sequential models are flattened into a single
        Sequential model. Nested Models or Modules inside the Sequential model
        are kept as is, but note that they are not supported in the right
        model. The initial layer is interpreted after flattening the model,
        so if the initial layer is negative, it is interpreted counting from
        the end of the flattened model.
        """
        # persistance settings
        NumpyPersistanceMixin.__init__(self, self._NUMPY_VARIABLES)
        # logging settings
        LogMixin.__init__(self, verbose)
        # initialize class attributes to None
        self._initialize_attributes()

        # flatten the model and check sequentiality
        neural_network = self._flatten_model(neural_network)

        # interpret negative initial layer
        self._interpret_negative_layers(initial_layer, neural_network)

        # layer subset within limits
        if not self._layer_subset_valid(self._initial_layer, neural_network):
            raise ValueError("Layer subset out of bounds")

        # make left / right models
        self._make_left_right_models(self._initial_layer, neural_network)

        # retrieve layer types
        self._retrieve_layer_types()

        # check layer types
        self._check_at_least_one_parameter_layer()

        # check every activation layer comes after a path layer
        self._check_activation_layers_after_path_layer()

        # get neural network parameters
        list_parameters = [
            self._layer_weights(layer_idx)
            for layer_idx in range(len(self._layer_types))
        ]
        self._list_weights, self._list_biases = zip(*list_parameters)

        # compile path selector
        self._compile()

    def _initialize_attributes(self):
        """Initializes attributes to their default values."""

        self._all_paths = None
        self._all_paths_input_neurons = None
        self._initial_layer = None
        self._layer_connections = None
        self._layer_types = None
        self._list_weights = None
        self._list_biases = None
        self._number_outputs = None
        self._partial_to_full_idxs = None
        self._zero_to_zero_counts = None
        self._input_shapes = None

        self.left_model_ = None
        self.right_model_ = None
        self.output_masks_ = None

        # initialize sets of layer types
        self._init_layer_type_sets()

    def _init_layer_type_sets(self):
        """Initializes the sets of layer types."""
        self.TRAIN_ONLY_LAYERS = self.LayerTypes.DROPOUT

        """Layers only active when training."""

        self.CONVOLUTIONAL_LAYERS = (
            self.LayerTypes.CONV_1D
            | self.LayerTypes.CONV_2D
            | self.LayerTypes.CONV_3D
        )
        """Supported convolutional layers."""

        self.MAXPOOLING_LAYERS = (
            self.LayerTypes.MAXPOOLING_1D
            | self.LayerTypes.MAXPOOLING_2D
            | self.LayerTypes.MAXPOOLING_3D
        )
        """Supported maxpooling layers."""

        self.PARAMETER_LAYERS = (
            self.CONVOLUTIONAL_LAYERS | self.LayerTypes.DENSE
        )
        """Layers with weights."""

        self.PATH_LAYERS = self.PARAMETER_LAYERS | self.MAXPOOLING_LAYERS
        """Layers that affect path layout."""

        self.SUPPORTED_ACTIVATIONS = self.LayerTypes.RELU
        """Supported activation layers for path selection."""

        self.ALL_ACTIVATIONS = self.LayerTypes.RELU | self.LayerTypes.SOFTMAX
        """All supported activation layers."""

        self.PATH_SELECTION_LAYERS = (
            self.MAXPOOLING_LAYERS | self.SUPPORTED_ACTIVATIONS
        )
        """Layers that affect path selection."""

        self.SUPPORTED_LAYERS = (
            self.TRAIN_ONLY_LAYERS
            | self.PATH_LAYERS
            | self.LayerTypes.FLATTEN
            | self.ALL_ACTIVATIONS
        )
        """All layers that may be present in the path selector."""

    @abstractmethod
    def _interpret_negative_layers(self, initial_layer, neural_network):
        """Returns the positive initial layer when the
        initial layer is negative"""
        pass

    @abstractmethod
    def _flatten_model(self, neural_network):
        """Flattens the model and checks sequentiality. Only nested sequential
        models are flattened, other models are kept as they are.

        Parameters
        ----------
        neural_network : torch.nn.Sequential or tensorflow.keras.Sequential
            Sequential model.

        Returns
        -------
        torch.nn.Sequential or tensorflow.keras.Sequential
            Flattened sequential model.
        """
        pass

    @abstractmethod
    def _compute_input_shapes(self):
        """Computes the input shapes of layers in the right model."""
        pass

    @abstractmethod
    def _get_number_of_outputs(self):
        """Returns the number of outputs of the right neural network.

        Returns
        -------
        int
            Number of outputs.
        """
        pass

    @property
    @abstractmethod
    def _output_layer_dtype(self):
        """Returns the data type of the output layer

        Returns
        -------
        str
            Data type of outputs
        """
        return self.__output_layer_dtype

    @_output_layer_dtype.setter
    def _output_layer_dtype(self, value):
        """Set the data type of the output layer"""
        self.__output_layer_dtype = value

    @property
    @abstractmethod
    def _layers(self):
        """Returns the layers of the right neural network.

        Returns
        -------
        list
            List of layers.
        """
        pass

    @property
    def _last_path_layer_idx(self):
        """Returns the index of the last layer that affects path selection.

        Returns
        -------
        int
            Index of the last layer that affects path selection.
        """
        cache = getattr(self, "_last_path_layer_idx_cache", None)
        if cache is not None:
            return cache
        layer_idx = len(self._layer_types) - 1
        while layer_idx >= 0:
            if self._layer_types[layer_idx] in self.PATH_LAYERS:
                self._last_path_layer_idx_cache = layer_idx
                return layer_idx
            layer_idx -= 1

    def _check_activation_layers_after_path_layer(self):
        """Checks there is at least one layer that affects path selection
        before an activation layer."""

        seen_path_layer = False

        # check every activation layer comes after a path layer
        for layer_type in self._layer_types:
            if layer_type in self.PATH_LAYERS:
                seen_path_layer = True
            elif layer_type in self.ALL_ACTIVATIONS:
                if seen_path_layer:
                    continue
                raise ValueError(
                    "Activation layer found before path layer. Set the initial"
                    " layer on one that affects  path selection (e.g. a"
                    " convolutional layer, or dense layer)."
                )

        # check for activation layers before the last path layer, which
        # must be relu or linear
        layer_idx = -1
        for layer_type in self._layer_types[: self._last_path_layer_idx]:
            layer_idx += 1

            # check if it is an unsupported activation layer
            non_linear_layer = self._get_name_if_not_linear_activation(
                layer_idx
            )
            if non_linear_layer is None:
                continue
            raise ValueError(
                "Activation layers previous to the last layer that affects"
                " path selection must be relu or linear. Found"
                f" {non_linear_layer}."
            )

    def _get_last_activation_layer_idxs(self):
        """Returns the indices of the last activation layers, meaning those
        that come after the last layer that affects path selection.

        Returns
        -------
        list
            List of indices of the last activation layers.
        """
        last_activation_layer_idxs = []
        layer_idx = self._last_path_layer_idx + 1

        # check each activation layer after the last path layer
        while layer_idx < len(self._layer_types):
            layer_type = self._layer_types[layer_idx]
            if layer_type not in self.ALL_ACTIVATIONS:
                layer_idx += 1
                continue
            last_activation_layer_idxs.append(layer_idx)
            layer_idx += 1

        return last_activation_layer_idxs

    def check_last_layer_linear(self):
        """Ensure the last activation layer is linear or relu.

        Raises
        ------
        ValueError
            If the said conditions are not met.
        """
        for layer_idx in self._get_last_activation_layer_idxs():
            # check if it is an unsupported activation layer
            non_linear_layer = self._get_name_if_not_linear_activation(
                layer_idx
            )
            if non_linear_layer is None:
                continue
            raise ValueError(
                "Activation layers after the last layer that affects path"
                " selection must be relu or linear. Found"
                f" {non_linear_layer}."
            )

    @abstractmethod
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
        pass

    def _retrieve_layer_types(self):
        """Retrieves layer types from right neural network."""
        self._layer_types = [
            self.LayerTypes.name_to_type(layer.__class__.__name__)
            for layer in self._layers
        ]

    @abstractmethod
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
        pass

    @abstractmethod
    def _layer_subset_valid(self, initial_layer, neural_network):
        """Checks that path selector uses layers in bounds of neural network.

        Parameters
        ----------
        initial_layer : int
            Layer of the model where path selection starts.
        neural_network : tensorflow.keras.Sequential or torch.nn.Sequential
            Keras sequential model.

        Returns
        -------
        bool
            True if path selector in bounds of neural network.
        """
        pass

    def _check_at_least_one_parameter_layer(self):
        """Checks that at least one parameter layer is used in the path
        selector.

        Raises
        ------
        ValueError
            If said conditions are not met.
        """
        # intersection of parameter layers and layers in path selector
        layer_types_set = 0
        for layer_type in self._layer_types:
            if layer_type is not None:
                layer_types_set = layer_types_set | layer_type

        if not (self.PARAMETER_LAYERS & layer_types_set):
            raise ValueError(
                f"At least one layer of {self.PARAMETER_LAYERS} required."
            )

    ##############################
    # Path selector compilation
    ##############################

    def _compile(self):
        """Compiles the path selector, computing all paths between layers
        and preparing transformation indices.
        """

        # get input shapes of the right model
        self._compute_input_shapes()

        # number of outputs
        self._number_outputs = self._get_number_of_outputs()

        # compile path selector
        self._compute_all_paths()
        self._make_transformation_indices()

    @abstractmethod
    def _make_left_right_models(self, initial_layer, neural_network):
        """Creates left and right models by splitting neural network at
        initial layer.

        Parameters
        ----------
        initial_layer : int
            Layer of the model where path selection starts.
        neural_network : tensorflow.keras.Sequential or torch.nn.Sequential
            Keras sequential model.
        """
        pass

    @abstractmethod
    def _get_connections_conv_layer(self, layer_idx):
        """Computes all connections in a convolutional layer.

        Parameters
        ----------
        layer_idx : int
            Layer number in right neural network.

        Returns
        -------
        array2d
            Array of paths.
        """
        pass

    @abstractmethod
    def _get_connections_maxpool_layer(self, layer_idx):
        """Computes all connections in a maxpool layer.

        Parameters
        ----------
        layer_idx : int
            Layer number in right neural network.

        Returns
        -------
        array2d
            Array of paths.
        """
        pass

    @abstractmethod
    def _get_connections_dense_layer(self, layer_idx):
        """Computes all connections in a dense layer.

        Parameters
        ----------
        layer_idx : int
            Layer number in right neural network.

        Returns
        -------
        array2d
            Array of paths.
        """
        pass

    def _get_connections(self, layer_idx):
        """Computes all possible connections to layer with specified index.

        Parameters
        ----------
        layer_idx : int
            Layer number in path selector.

        Returns
        -------
        array2d
            Sorted array of paths.
        """
        # Value for no connections
        paths = None

        # All paths for Dense layer
        if self._layer_types[layer_idx] == self.LayerTypes.DENSE:
            paths = self._get_connections_dense_layer(layer_idx)

        # All paths for Conv layer
        elif self._layer_types[layer_idx] in self.CONVOLUTIONAL_LAYERS:
            paths = self._get_connections_conv_layer(layer_idx)

        # All paths for MaxPooling layer
        elif self._layer_types[layer_idx] in self.MAXPOOLING_LAYERS:
            paths = self._get_connections_maxpool_layer(layer_idx)

        return None if paths is None else qs_paths.sort_paths(paths)

    def _make_output_masks(self):
        """Build mask that transform equations for any general output to
        the equations of a specific output according to the connection pattern
        to that output on the last layer.

        Returns
        -------
        array2d
            Array of booleans, where the nth element is the mask that
            transforms the built coefficients for any output to the
            coefficients of the nth output.
        """
        # get connection pattern of last layer
        connections = self._get_connections(self._last_path_layer_idx)

        output_masks = []
        for out_idx in range(self._number_outputs):
            # input neurons that are connected to this output
            connected = connections[connections[:, 1] == out_idx + 1, 0]

            # transform to a mask: input neurons to last layer
            # (all_paths[:, -2]) in connected neurons to this output.
            output_masks.append(np.in1d(self._all_paths[:, -2], connected))

        return np.array(output_masks)

    def _compute_all_paths(self):
        """Computes all possible paths between each layer taking weights into
        account.

        Note
        -----
        Computes as well the combinations between all paths in the path
        selection layers and the output masks by a call to _make_output_masks.
        """
        # all connections between consecutive pairs of layers
        # except last layer
        self._layer_connections = [
            self._get_connections(layer)
            for layer in range(self._last_path_layer_idx)
        ]

        # extend connections up to last layer with None
        while len(self._layer_connections) != len(self._list_weights):
            self._layer_connections.append(None)

        # add last layer connections simulating all inputs are connected to
        # a single neuron
        last_layer_inputs = np.prod(
            self._input_shapes[self._last_path_layer_idx]
        )
        connections = [[neuron, 1] for neuron in range(last_layer_inputs + 1)]
        self._layer_connections[self._last_path_layer_idx] = np.array(
            connections, np.int32
        )

        # combine paths
        connections = [x for x in self._layer_connections if x is not None]

        # compute size of combined paths to at least give the user an idea
        # of the size of the output
        combined_paths_size = qs_comb.compute_combine_paths_output_size(
            connections
        )

        self._log(f"Total number of outputs: {self._number_outputs}")
        self._log(
            f"Maximum number of paths per output: {combined_paths_size}, "
            "be mindful of the memory usage that this may imply."
        )

        self._all_paths = qs_comb.combine_paths(connections)

        # input neuron that begins each path
        self._all_paths_input_neurons = self._all_paths[:, 0]

        # get mask that transform equations for any general output to
        # the equations of a specific output according to connection pattern
        # to that output on the last layer.
        self.output_masks_ = self._make_output_masks()

        # number of paths on each output
        effective_paths = self.output_masks_.sum(axis=1)
        self._log(
            f"Number of paths on each output: {effective_paths}",
            log_level=1,
        )

    def _make_transformation_indices(self):
        """Computes necessary indices to transform all paths between two layers
        to the corresponding portion of the combined paths of all layers.

        Note
        -----
        For performance reasons, it is more efficient to make path selection
        on each layer separately and then expand the selection on each layer
        to match the combination of all the paths. For example, if we have
        two layers with the following connection pattern:

        >>> connections_layer_1 = [
        ...     [0, 1],
        ...     [0, 2],
        ...     [1, 1],
        ...     [1, 2],
        ...     [3, 1],
        ...     [3, 3]
        ... ]

        >>> connections_layer_2 = [
        ...     [0, 1],
        ...     [0, 2],
        ...     [1, 1],
        ...     [1, 2],
        ...     [2, 1],
        ...     [2, 2]
        ... ]

        The combination of paths between both layers would result in:

        >>> combination = [
        ...     [0, 0, 1],
        ...     [0, 0, 2],
        ...     [0, 1, 1],
        ...     [0, 1, 2],
        ...     [0, 2, 1],
        ...     [0, 2, 2],
        ...     [1, 1, 1],
        ...     [1, 1, 2],
        ...     [1, 2, 1],
        ...     [1, 2, 2],
        ...     [3, 1, 1],
        ...     [3, 1, 2]
        ... ]

        If we examine closely, we see that on the first two columns there are
        redundancies, as [0, 0], [0, 1], [0, 2], [1, 1], [1,2], [3, 1] are all
        repeated many times. Furthermore, aside from [0, 0], all the rest are
        connections present in connections_layer_1.

        A mapping can be made from each connection in the first two columns
        to the index of connections_layer_1 with that connections; e.g.
        the mapping of [0, 1], [0, 1], [0, 2], [0, 2], ..., [3, 1] would be
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]. We can do the analogue with the
        second and third columns and connections_layer_2, mapping the
        connections [0, 1], [0,2], [1, 1], ..., [1, 2] to the indexes
        in connections_layer_2 [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3].

        If we calculate this mapping now, we can use these 'transformation
        indices' to expand a path selection between two layers to the
        corresponding selection on the combined space. For example, if we
        have a selection for layer 2 such as:

        >>> selection_layer_2 = np.array([T, T, F, F, T, T])

        we can expand it to the the path combination space with our previous
        mapping:

        >>> expanded_selection_layer_2 = selection_layer_2[
        ...     [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3]
        ... ]
        >>> expanded_selection_layer_2
        [T, T, F, F, T, T, F, F, T, T, F, F]

        The same can be done with a selection_layer_1, but we have to keep in
        mind that the paths of type [0, 0] dont have a mapping to an index
        of connections_layer_1, so we have to calculate as well how many
        zeros to zeros are there in each consecutive pair of columns in the
        combination array. In this case, there were two [0, 0] in
        the first and second columns, so the first two positions are set to
        True and the rest are computed with selection_layer_1[[0, 0, 1,...]]:

        >>> expanded_selection_layer_1 = np.concatenate(
        ...     ([T, T], selection_layer_1[[0, 0, 1, 1, 2, 2, 3, 3, 4, 4]])
        ... )
        >>> expanded_selection_layer_1
        [T, T, ...]

        After the expansion of selection_layer_1 and selection_layer_2,
        the logical AND between both arrays gives us the path selection for
        each path in the combined space:

        >>> path_selection = (
        ...     expanded_selection_layer_1 & expanded_selection_layer_2
        ... )
        """
        self._partial_to_full_idxs = []
        self._zero_to_zero_counts = []

        # map paths between two layers to index in fully combined paths
        layer_idx = -1
        for connections in self._layer_connections:
            # layer with no path changes (e.g. dropout)
            if connections is None:
                self._partial_to_full_idxs.append(None)
                self._zero_to_zero_counts.append(None)
                continue
            else:
                layer_idx += 1

            # expanded partial paths after combination
            full_paths = self._all_paths[:, layer_idx : layer_idx + 2]

            # number of zero to zeros in path combination
            self._zero_to_zero_counts.append(
                np.sum(np.all(full_paths == np.array([0, 0]), axis=1))
            )

            # indexes that tranform partial paths to combination
            self._partial_to_full_idxs.append(
                qs_paths.partial_to_full_idxs(
                    connections, full_paths[self._zero_to_zero_counts[-1] :]
                )
            )

    ##############################
    # Path selection
    ##############################

    def _select_paths(
        self, layer_idx, flat_inputs_with_bias, pending_activation
    ):
        """Get selected paths for a batch of samples for a given layer.

        Parameters
        ----------
        layer_idx : int
            Layer number in the path selector.
        flat_inputs_with_bias : array2d
            Contains for each sample of a batch the flat inputs to the layer
            with an added 1 at the beginning of said inputs representing the
            bias.
        pending_activation : BaseLayerTypes or None
            Indicates a pending activation to be applied to the paths.

        Returns
        -------
        array2d
            Selected paths as array of booleans, where each element corresponds
            to a sample of the batch and the element is a boolean mask that
            indicates the paths that were selected.
        """
        selected_paths = None

        # apply relu activation to paths if pending
        if pending_activation is not None:
            selected_paths = qs_paths.non_zero_input_select_paths(
                flat_inputs_with_bias, self._layer_connections[layer_idx]
            )
            pending_activation = None

        # apply maxpooling activation to paths
        if self._layer_types[layer_idx] in self.MAXPOOLING_LAYERS:
            maxpool_selected_paths = qs_maxpool.select_paths_maxpool_layer(
                flat_inputs_with_bias, self._layer_connections[layer_idx]
            )
            # didnt apply relu activation before
            if selected_paths is None:
                selected_paths = maxpool_selected_paths
            else:  # combine path selection
                selected_paths = selected_paths & maxpool_selected_paths

        return selected_paths, pending_activation

    @staticmethod
    @abstractmethod
    def _forward_inference(model, X):
        """Inference of model on X, in evaluation mode, so that layers like
        dropout are not applied.

        Parameters
        ----------
        model : object
            Model to be inferred.
        X : arrayNd
            Input to the neural network.

        Returns
        -------
        arrayNd
            Output of the neural network.
        """
        pass

    def _propagate_left(self, X):
        """Propagate through the network samples in X up to first layer of
        path selector.

        Parameters
        ----------
        X : arrayNd
            Neural network samples.

        Returns
        -------
        arrayNd
            Array with the inputs to first layer of path selector.
        """
        # path selector through the whole network
        if self.left_model_ is None:
            return X

        # propagate until reach input of path selector
        return self._forward_inference(self.left_model_, X)

    @staticmethod
    @abstractmethod
    def _as_numpy(X):
        """Converts X to numpy array if it is not already.

        Parameters
        ----------
        X : arrayNd
            Input to the neural network.

        Returns
        -------
        arrayNd
            Numpy array.
        """
        pass

    def samples_to_coefficients(self, X):
        """Computes linear systems coefficients (A matrix in Ax = b) associated
        to path selector using samples in X.

        Parameters
        ----------
        X : arrayNd
            Input to the neural network.

        Notes
        -----
        Generally, it is expected that the connection pattern to the last layer
        will be similar or identical to each output neuron. Therefore, the
        coefficients generated for each output are expected to be similar or
        identical. For this reason, an assumption is made that there is only
        one output neuron and the connection pattern to this output neuron is
        dense. The coefficients are generated for this output, and the columns
        of this coefficient matrix can be filtered later on using the class
        attribute output_masks_ for any of the specific outputs of the neural
        network.

        Returns
        -------
        array2d
            Coefficients of a linear system for a generic output.
        """
        # propagate input up to inputs of path selector
        X = self._propagate_left(X)

        # path selection mask for each sample
        select_masks = np.full((X.shape[0], self._all_paths.shape[0]), True)

        # incrementally make path selection through each layer
        current_outputs = X
        first_layer_inputs = None
        layer_idx = 0
        pending_activation = None

        for layer_idx, layer_type in enumerate(self._layer_types):
            # get input of next layer if needed
            current_inputs = current_outputs
            if layer_idx < len(self._list_weights) - 1:
                current_outputs = self._forward_inference(
                    self._layers[layer_idx], current_inputs
                )

            # check for need of path selection (previously applied activation
            # or current layer is maxpooling)
            if layer_type in self.PATH_LAYERS:
                # save for later coefficient retrieval if not yet set
                pending_path_select = pending_activation is not None or (
                    layer_type in self.MAXPOOLING_LAYERS
                )

                # check if flat inputs are needed
                if first_layer_inputs is None or pending_path_select:
                    # flatten inputs and add bias for this layer
                    flat_input_with_bias = _flatten_and_insert_one(
                        self._as_numpy(current_inputs)
                    )
                    if first_layer_inputs is None:
                        first_layer_inputs = flat_input_with_bias

                # check if path selection is needed: either pending activation
                # or current layer is maxpooling
                if pending_path_select:
                    # select paths
                    path_selection, pending_activation = self._select_paths(
                        layer_idx, flat_input_with_bias, pending_activation
                    )

                    # update path selection masks:

                    # first path not zero to zero
                    first_idx = self._zero_to_zero_counts[layer_idx]

                    # update select masks with AND operator
                    select_masks[:, first_idx:] &= path_selection[
                        :, self._partial_to_full_idxs[layer_idx]
                    ]

            # mark activation as pending for next layer if needed
            if layer_type in self.SUPPORTED_ACTIVATIONS:
                pending_activation = layer_type

        # ensure that first_layer_inputs is set
        if first_layer_inputs is None:
            first_layer_inputs = _flatten_and_insert_one(self._as_numpy(X))

        # retrieve coefficients associated to path selection
        return qs_paths.retrieve_coefficients(
            select_masks, self._all_paths_input_neurons, first_layer_inputs
        )

    @abstractmethod
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
        pass
