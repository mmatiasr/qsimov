"""Utilities for path combinations.
"""
import functools
import qsimov.paths.paths as paths_
from qsimov.paths.c_combine import c_combine_paths_left_right_sort_join
from collections import defaultdict
import numpy as np


def combine_paths_left_right(paths_left, paths_right):
    """Combines two sets of paths. For each p_l in paths_left and p_r in
    paths_right, if p_l[-1] == p_r[0], a new path resulting in the
    concatenation of p_l and p_r[1:] is generated for the output of this
    function. Paths p_r in paths_right with p_r[0] == 0, are always matched
    with a "bias path" of zeros with the shape paths_left.shape[1].

    Parameters
    ----------
    paths_left : array2d
        Paths on the left side.
    paths_right : array2d
        Paths on the right side.

    Returns
    -------
    array2d
        Combined paths with shape
        (number_paths, paths_left.shape[1] + paths_right.shape[1] - 1).

    Example
    -------
    Note in the following example how the path p_r = [0, 0, 1] is matched
    with a "bias path" [0, 0, 0], resulting in the first generated path
    [0, 0, 0, 0, 1]:

    >>> paths_left = np.array([[0, 0, 1], [1, 1, 1], [0, 1, 2]]),
    >>> paths_right = np.array([[0, 0, 1], [1, 1, 1], [1, 2, 2], [3, 1, 1]]),
    >>> combine_paths_left_right(paths_left, paths_right)
    np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 2, 2],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 2, 2],
        ]
    )
    """
    return c_combine_paths_left_right_sort_join(
        np.asanyarray(paths_left, np.int32),
        np.asanyarray(paths_right, np.int32),
    )


def compute_combine_paths_output_size(list_paths):
    """Computes the expected output number of paths after combining all the
    paths in list_paths.

    Parameters
    ----------
    list_paths : list[array2d]
        List of paths.

    Returns
    -------
    int
        Number of paths.
    """
    # no paths to combine
    if len(list_paths) == 0:
        return 0

    # maps to each output neuron of first layer number of input neurons
    # connected to it.
    out_neuron_path_counts_current_layer = defaultdict(lambda: 0)
    for path in list_paths[0]:
        out_neuron_path_counts_current_layer[path[-1]] += 1

    # repeats for the rest of layers, where the number of paths connected
    # to each input neuron is given by the number of paths connected to the
    # same output neuron in the last layer.
    for paths in list_paths[1:]:
        # go to next layer
        out_neuron_path_counts_last_layer = (
            out_neuron_path_counts_current_layer
        )
        out_neuron_path_counts_current_layer = defaultdict(lambda: 0)

        # Add 1 path to 0 output neuron in last layer counts, for bias
        out_neuron_path_counts_last_layer[0] = 1

        # "connect" paths from last to current
        for path in paths:
            out_neuron_path_counts_current_layer[
                path[-1]
            ] += out_neuron_path_counts_last_layer[path[0]]

    # sum total connected paths to each output neuron
    return sum(out_neuron_path_counts_current_layer.values())


def combine_paths(list_paths):
    """Combine list of paths by succesive calls to combine_paths_left_right.

    Parameters
    ----------
    list_paths : list[array2d]
        List of paths.

    Returns
    -------
    array2d
        Inner join of paths.
    """
    if len(list_paths) == 0:
        return np.empty((0, 2), np.int32)

    if len(list_paths) == 1:
        return paths_.sort_paths(np.asanyarray(list_paths[0], np.int32))

    # combine paths layer by layer recursively
    paths_combined = functools.reduce(combine_paths_left_right, list_paths)

    # sort
    paths_combined = paths_.sort_paths(paths_combined)

    return paths_combined
