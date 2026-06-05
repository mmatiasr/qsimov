import numpy as np


def out_to_in_adjacency_list_to_edge_list(out_to_in_adjacency_list):
    """Converts an adjacency list where the keys are the outputs and the values
    are the inputs to an edge list where the first column is the input and the
    second column is the output.

    Parameters
    ----------
    out_to_in_adjacency_list : dict[int, list[int]]
        The out-to-in adjacency list.

    Returns
    -------
    edge_list : np.ndarray
        The edge list.
    """
    return np.array(
        [
            [input, out]
            for out, inputs in out_to_in_adjacency_list.items()
            for input in inputs
            if input is not None
        ]
    )


def weights_to_channels_first(weights):
    """Converts weights from channels last to channels first.

    Parameters
    ----------
    weights : np.ndarray
        The weights.

    Returns
    -------
    weights : np.ndarray
        The weights in channels first.
    """

    return np.transpose(weights, [-1, -2, *range(len(weights.shape) - 2)])


def samples_to_channels_first(samples):
    """Converts samples from channels last to channels first.

    Parameters
    ----------
    samples : np.ndarray
        The samples.

    Returns
    -------
    samples : np.ndarray
        The samples in channels first.
    """

    return np.transpose(samples, [0, -1, *range(1, len(samples.shape) - 1)])
