"""General functions module for path generation.
"""

import numpy as np
from qsimov.paths.c_paths import (
    c_non_zero_input_select_paths,
    c_retrieve_coefficients,
)


def sort_paths(paths):
    """Sort paths by layer order.

    Parameters
    ----------
    paths : array2d
        Numpy array where each row is a path and each
        path element is a neuron index.

    Returns
    -------
    array2d
        Sorted paths.
    """
    if paths.shape[0] == 0:
        return paths

    return paths[np.lexsort(paths[:, ::-1].T)]


def _rowise_to_bytes(array):
    """Convert each row in array to a single bytestring.

    Parameters
    ----------
    array : array2d
        Array of numbers.

    Returns
    -------
    array1d
        Array where each element is the corresponding bytes of array row.
    """
    # necessary for conversion to numpy unicode string
    assert array.itemsize % 4 == 0

    row_size = array.shape[1] * (array.itemsize // 4)
    dtype = "U" + str(row_size)
    return np.frombuffer(array.tobytes(), dtype=dtype)


def paths_subset_of(paths, paths_set):
    """Return a boolean array indicating the paths included in paths_set.

    Parameters
    ----------
    paths : array2d
        Paths to lookup in paths_set. Must be sorted.
    paths_set : array2d
        Paths set.

    Returns
    -------
    array1d
        Boolean array of array with True for each path in paths included for
        each sample.
    """

    assert paths.shape[1] == paths_set.shape[1]

    # interpret paths as byte strings
    paths_bytes = _rowise_to_bytes(paths)
    paths_set_bytes = _rowise_to_bytes(paths_set)

    # binary search of indexes
    hits = np.searchsorted(paths_bytes, paths_set_bytes, "left")
    result = np.full((paths.shape[0]), False)
    result[hits] = True

    return result


def partial_to_full_idxs(partial, full):
    """Get indices to transform a unique sorted array "partial" into the
    array "full", made out of rows always included in "partial". Meaning:

        partial[partial_to_full_idxs(partial, full)] == full.

    Parameters
    ----------
    partial : array2d
        Sorted 2d array.
    full : array2d
        Target array.

    Example
    -------
    >>> partial = [
    ...     [0, 1],
    ...     [0, 2],
    ...     [1, 1],
    ...     [1, 2],
    ...     [3, 1],
    ...     [3, 3]
    ... ]
    >>> full = [
    ...     [0, 1],
    ...     [0, 1],
    ...     [0, 2],
    ...     [0, 2],
    ...     [1, 1],
    ...     [1, 1],
    ...     [1, 2],
    ...     [1, 2],
    ...     [3, 1],
    ...     [3, 1]
    ... ]
    >>> partial_to_full_idxs(partial, full)
    [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

    Returns
    -------
    array1d
        Transformation indices.
    """
    return np.searchsorted(_rowise_to_bytes(partial), _rowise_to_bytes(full))


def retrieve_coefficients(
    select_masks, paths_input_neurons, flat_inputs_with_bias
):
    """Retrieve coefficients from paths according to multiple path selections
    and the inputs that generated each one.

    Parameters
    ----------
    select_masks: array2d
        Boolean array indicating which paths where selected for each sample.
    paths_input_neurons: array1d
        Input neuron for each path in all the combined paths of a path
        selector.
    flat_inputs_with_bias: array2d
        Flat input for each sample adding adding a one at the beginning
        of the array, representing the bias neuron.

    Returns
    -------
    array1d
        Coefficients of nn_input associated to each path.
    """
    flat_inputs_with_bias = np.asanyarray(flat_inputs_with_bias)
    return c_retrieve_coefficients(
        np.asanyarray(select_masks, np.bool8),
        np.asanyarray(paths_input_neurons, np.int32),
        np.asfarray(flat_inputs_with_bias, flat_inputs_with_bias.dtype),
    )


def non_zero_input_select_paths(flat_inputs_with_bias, all_paths):
    """Given the connection pattern to a layer based on multiplication of
    inputs by weights (such as Dense or Conv2D) described in all_paths;
    and the inputs for said layer for many different samples, contained in
    flat_inputs_with_bias; generates a boolean mask for each sample indicating
    which paths of all_paths where selected for those inputs.

    Parameters
    ----------
    flat_inputs_with_bias : array2d
        Flat input for each sample adding 1 at the beginning to represent the
        bias neuron.
    all_paths : array2d
        Paths to filter based on input. A path is selected when the input
        neuron is zero on the input.

    Returns
    -------
    array1d
        Boolean array indicating which paths where selected.
    """
    flat_inputs_with_bias = np.asanyarray(flat_inputs_with_bias)
    return c_non_zero_input_select_paths(
        np.asfarray(flat_inputs_with_bias, flat_inputs_with_bias.dtype),
        np.asanyarray(all_paths, np.int32),
    )
