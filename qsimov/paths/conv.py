"""Compute convolution of an input image and its atributtes.
"""
import numpy as np
import qsimov.paths.c_conv as c_conv


def _get_paddings(padding, input_shape, kernel_shape, strides):
    """Compute total minimum padding on each dimension (sum of padding at
    the beginning of dimension and at the end).

    Parameters
    ----------
    padding : str or int or array1d
        Padding type, either "valid" or "same" or int or array of ints with
        total padding for each dimension.
    input_shape : array1d
        Shape of input of convolution, without channels and batch dimension.
    kernel_shape : array1d
        Shape of kernel. As long as len(input_shape) - 1.
    strides : array1d
        Strides on each dimension. As long as len(input_shape) - 1.

    Returns
    -------
    array1d
        Array with total padding on each dimension, as long as kernel_shape.
    """
    assert len(kernel_shape) == len(strides)

    input_shape = np.asanyarray(input_shape)
    kernel_shape = np.broadcast_to(kernel_shape, (len(input_shape),))
    strides = np.broadcast_to(strides, (len(input_shape),))

    # If padding is an int, convert it to an array
    if isinstance(padding, int):
        # multiply by 2 because we need to pad on both sides
        # in pytorch convention (1, 1) means add padding 1 on both size.
        # then this function would return (2, 2), as it gives the total padding
        # on each dimension.
        return 2 * np.full(len(input_shape), padding)

    # If padding is not string, convert it to an array
    if not isinstance(padding, str):
        # multiply by 2 because we need to pad on both sides
        return 2 * np.array(padding)

    if padding == "same":  # padding to cover all input
        # https://stackoverflow.com/a/44242277
        out_dimensions = np.ceil(
            np.asfarray(input_shape) / np.asfarray(strides)
        )
        pad_along_dimensions = np.maximum(
            (out_dimensions - 1) * strides + kernel_shape - input_shape, 0
        )
        return pad_along_dimensions
    else:  # no padding
        return np.zeros_like(strides)


def _compute_conv_output_shape(
    input_shape, num_filters, kernel_shape, strides, padding, data_format
):
    """
    Computes the output shape of a convolutional layer of any dimensionality.

    Parameters
    ----------
    input_shape : array1d
        The shape of the input tensor, excluding the batch dimension and the
        number of channels.
    num_filters : array1d
        The number of filters in the convolutional layer.
    kernel_shape : int or array1d
        The size of the convolutional kernel. If an int, it is assumed to be
        the same for all dimensions.
    strides : int or array1d
        The stride of the convolution. If an int, it is assumed to be the
        same for all dimensions.
    padding : str or int or array1d
        Padding type, either "valid" or "same" or int or array of ints with
        total padding for each dimension.
    data_format : str
        The data format of the input and output tensors, either
        'channels_first' or 'channels_last'.

    Returns
    -------
    output_shape : array1d
        The shape of the output tensor, excluding the batch dimension.
    """
    # Convert kernel_size and strides to tuples if they are ints
    if isinstance(kernel_shape, int):
        kernel_shape = (kernel_shape,) * len(input_shape)
    if isinstance(strides, int):
        strides = (strides,) * len(input_shape)

    # Check the validity of the arguments
    assert len(input_shape) == len(kernel_shape) == len(strides)
    assert data_format in ("channels_first", "channels_last")

    # Compute the total padding on each dimension
    paddings = _get_paddings(padding, input_shape, kernel_shape, strides)

    # Compute the output shape for each dimension
    output_shape = np.floor_divide(
        input_shape + paddings - kernel_shape + strides, strides
    ).tolist()

    # Add the number of filters as the first or last dimension
    if data_format == "channels_first":
        output_shape.insert(0, num_filters)
    else:
        output_shape.append(num_filters)

    # Return the output shape as a tuple
    return np.array(output_shape).astype(int)


def _side_shape_to_channels_first(side_shape):
    """Converts an input or output shape without batch dimension to channels
    first.

    Parameters
    ----------
    side_shape : array1d
        Shape of input or output of convolution, without batch dimension.

    Returns
    -------
    array1d
        Shape of input or output of convolution, with channels first.
    """
    side_shape = np.asanyarray(side_shape)
    return side_shape[[-1] + list(range(len(side_shape) - 1))]


def _get_channels_last_to_channels_first_id_map(channels_last_shape):
    """Creates a map of input neurons to ids in channels first mode.

    Parameters
    ----------
    channels_last_shape : array1d
        Shape of input or output of convolution, without batch dimension.

    Returns
    -------
    dict
        Map of input neurons to ids in channels first mode.
    """
    # create a map of input neurons to ids
    channels_last_ids = (
        np.arange(np.prod(channels_last_shape)).reshape(channels_last_shape)
        + 1
    )

    # get shape in channels first mode
    channels_first_shape = _side_shape_to_channels_first(channels_last_shape)

    # create a map of ids to input neurons in channels first mode
    channels_first_ids = (
        np.arange(np.prod(channels_first_shape)).reshape(channels_first_shape)
        + 1
    )
    # move channel axis to back so the flat inputs are aligned with those
    # of the flat channels last mode
    channels_first_ids = np.moveaxis(channels_first_ids, 0, -1)

    # create a map of ids
    id_map = dict(zip(channels_last_ids.ravel(), channels_first_ids.ravel()))
    id_map[0] = 0

    return id_map


def _map_channels_last_paths_to_channels_first(
    input_shape, output_shape, paths
):
    """Maps paths from channels last to channels first.

    Parameters
    ----------
    input_shape : array1d
        Channels last shape of input of convolution, without batch dimension.
    output_shape : array1d
        Channels last shape of output of convolution, without batch dimension.
    paths : array2d
        List of connections between input and output neurons.

    Returns
    -------
    array2d
        Transformed list of connections between input and output neurons.
    """
    new_paths = np.array(paths)

    # map inputs and outputs of paths to channels first
    for side_idx, side_shape in enumerate((input_shape, output_shape)):
        if side_shape is None:
            continue

        id_map = _get_channels_last_to_channels_first_id_map(side_shape)

        # map input ids
        for idx in range(len(new_paths)):
            new_paths[idx, side_idx] = id_map[new_paths[idx, side_idx]]

    return new_paths


def get_all_paths_conv_layer(
    input_shape,
    weights,
    biases,
    strides=1,
    padding="valid",
    groups=1,
    data_format="channels_last",
):
    """Generates all paths between previous layer and current conv layer.

    Parameters
    ----------
    input_shape : array1d
        Shape of input of convolution, not including batch dimension.
    weights : arrayNd
        Numpy array of shape:
        (d1, d2, ..., dN, input_channels, output_channels) if channels_last.
        (output_channels, input_channels, d1, d2, ..., dN) if channels_first.
    biases : array1d
        Numpy array of shape (output_channels,).
    strides : int or array1d, optional
        Strides of convolution, by default 1.
    padding : str or int or array1d, optional
        Padding type, either "valid" or "same" or int or array of ints with
        total padding for each dimension. By default "valid".
    groups : int, optional
        Number of groups for grouped convolution, by default 1.
    data_format : str, optional
        Data format, one of "channels_last" or "channels_first". If
        channels_first is used, input_shape and output_shape are expected to be
        (channels, height, width), else, (height, width, channels).
        By default "channels_last".

    Returns
    -------
    array2d
        Numpy array with connections from each input neuron to each output
        neuron.
    """
    # dimensionality of the convolution (1D, 2D, 3D, etc.)
    dimension = len(input_shape) - 1

    assert len(weights.shape) == 2 + dimension
    assert type(strides) is int or hasattr(strides, "__iter__")
    assert padding in ("same", "valid") or type(padding) is not str
    assert data_format in ("channels_last", "channels_first")
    groups = int(groups)
    assert groups > 0

    # transform input arguments
    input_shape = np.asanyarray(input_shape)
    strides = np.broadcast_to(strides, (dimension,))  # to shape (dimension, )

    # transform problem to channels last
    if data_format == "channels_first":
        # map (out, in, d1, d2, ..., dN) to (d1, d2, ..., dN, in, out)
        weights = np.transpose(weights, list(range(2, 2 + dimension)) + [1, 0])
        input_shape = input_shape[list(range(1, dimension + 1)) + [0]]

    kernel_shape = weights.shape[:-2]  # grid shape

    # compute output shape
    output_shape = _compute_conv_output_shape(
        input_shape=input_shape[:-1],
        num_filters=weights.shape[-1],
        kernel_shape=kernel_shape,
        strides=strides,
        padding=padding,
        data_format="channels_last",
    )

    # get total padding on lower side of each dimension
    paddings = _get_paddings(padding, input_shape[:-1], kernel_shape, strides)
    paddings = (paddings // 2).astype(int)

    # get all paths with c extension
    if dimension == 1:
        path_function = c_conv.c_get_all_paths_conv1d_layer
    elif dimension == 2:
        path_function = c_conv.c_get_all_paths_conv2d_layer
    elif dimension == 3:
        path_function = c_conv.c_get_all_paths_conv3d_layer
    else:  # dimension >= 4
        raise NotImplementedError(
            "get_all_paths_conv_layer not implemented for dimension >= 4"
        )

    paths = path_function(
        input_shape=np.asanyarray(input_shape, dtype=np.int32),
        output_shape=np.asanyarray(output_shape, dtype=np.int32),
        weights=np.asanyarray(weights, dtype=np.float32),
        biases=np.asanyarray(biases, dtype=np.float32),
        strides=np.asanyarray(strides, dtype=np.int32),
        paddings=np.asanyarray(paddings, dtype=np.int32),
        groups=groups,
    )

    # convert neuron ids to channels first format (if necessary)
    if data_format == "channels_first":
        paths = _map_channels_last_paths_to_channels_first(
            input_shape, output_shape, paths
        )
    return paths
