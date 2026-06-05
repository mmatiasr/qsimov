"""Computes paths for a max-pooling layer.
"""
import numpy as np
from qsimov.paths.conv import get_all_paths_conv_layer
from qsimov.paths.c_maxpooling import c_select_paths_maxpool_layer


def get_all_paths_maxpool_layer(
    input_shape,
    pool_size,
    strides=None,
    padding="valid",
    data_format="channels_last",
):
    """Computes all paths for a max-pooling layer.

    Parameters
    ----------
    input_shape: array1d
        Shape of the input tensor.
    pool_size: array1d or int
        Window size over which to take the maximum.
    strides: array1d or int
        Specifies how far the pooling window moves for each pooling step.
        Defaults to None.
    padding : str or int or array1d
        Padding type, either "valid" or "same" or int or array of ints with
        total padding for each dimension.
    data_format: str
        The ordering of the dimensions in the inputs. "channels_last"
        corresponds to inputs with shape (height, width, channels) while
        "channels_first" corresponds to inputs with shape
        (channels, height, width). By default, 'channels_last' is used.

    Returns
    -------
    array2d
        Array of all possible paths through the max-pooling layer.
    """
    pool_size = np.broadcast_to(pool_size, (len(input_shape) - 1,))
    # Compute correct value for strides
    strides = pool_size if strides is None else strides

    # Transform to a call to convolutional layer
    if data_format == "channels_last":
        num_channels = input_shape[-1]
        weights_shape = list(pool_size) + [1, num_channels]
    else:
        num_channels = input_shape[0]
        weights_shape = [num_channels, 1] + list(pool_size)

    # Weights are all ones, biases are all zeros
    weights = np.ones(weights_shape)
    biases = np.zeros(num_channels)

    return get_all_paths_conv_layer(
        input_shape,
        weights,
        biases,
        strides,
        padding,
        groups=num_channels,  # One group per channel
        data_format=data_format,
    )


def select_paths_maxpool_layer(flat_inputs_with_bias, all_paths_layer):
    """Select the paths that correspond to the maximum activation for each
    output neuron in a max pooling layer.

    Parameters
    ----------
    flat_inputs_with_bias: array2d
        Flat input for each sample adding bias.
    all_paths_layer: array1d
        Array of all possible paths through the max-pooling layer.

    Returns
    -------
    array1d
        Array of all active paths through the max-pooling layer.
    """
    flat_inputs_with_bias = np.asanyarray(flat_inputs_with_bias)
    return c_select_paths_maxpool_layer(
        np.asfarray(flat_inputs_with_bias, flat_inputs_with_bias.dtype),
        np.asanyarray(all_paths_layer, np.int32),
    )
