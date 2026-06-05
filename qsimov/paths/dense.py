"""Compute the paths for a dense layer.
"""
import numpy as np
import qsimov.paths.c_dense as c_dense


def get_all_paths_dense_layer(weights, biases):
    """All possible paths between two dense layers.

    Parameters
    ----------
    weights : array2d
        Weight matrix between two layers.
    biases : array1d
        Bias vector between two layers.

    Returns
    -------
    array2d
        Paths between both represented layers.
    """
    return c_dense.c_get_all_paths_dense_layer(
        np.asarray(weights), np.asarray(biases)
    )
