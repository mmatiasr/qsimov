cimport cython
import numpy as np
import itertools
cimport numpy as np
np.import_array()


ctypedef np.uint8_t BOOL_t
DTYPE = np.int32
ctypedef fused DTYPE_t:
    np.int32_t
    np.int64_t

ctypedef fused FTYPE_t:
    np.float32_t
    np.float64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef c_get_all_paths_conv1d_layer(
    np.ndarray[DTYPE_t, ndim=1] input_shape,
    np.ndarray[DTYPE_t, ndim=1] output_shape,
    np.ndarray[FTYPE_t, ndim=3] weights,
    np.ndarray[FTYPE_t, ndim=1] biases,
    np.ndarray[DTYPE_t, ndim=1] strides,
    np.ndarray[DTYPE_t, ndim=1] paddings,
    int groups,
):
    cdef int dimension, group, in_filters_lower, out_filters_lower, in_filters_per_group, out_filters_per_group
    cdef np.ndarray[DTYPE_t, ndim=1] kernel_shape
    cdef np.ndarray[DTYPE_t, ndim=2] input_ids
    cdef np.ndarray[DTYPE_t, ndim=2] out_ids
    cdef np.ndarray[BOOL_t, ndim=3] nonzero_weights_mask
    cdef np.ndarray[DTYPE_t, ndim=2] connected_inputs
    cdef np.ndarray[DTYPE_t, ndim=3] connected_inputs_broadcast
    cdef np.ndarray[DTYPE_t, ndim=3] connected_outputs
    cdef np.ndarray[DTYPE_t, ndim=1] lower
    cdef np.ndarray[DTYPE_t, ndim=1] upper
    cdef np.ndarray[DTYPE_t, ndim=1] in_neuron_coordinates
    cdef np.ndarray[BOOL_t, ndim=3] mask
    cdef np.ndarray[DTYPE_t, ndim=1] out_cell 


    dimension = 1  # convolutional layer is 1D
    kernel_shape = np.array([weights.shape[0]], DTYPE)  # grid shape

    # ids associated to each input neuron and output neuron
    input_ids = np.arange(input_shape.prod(), dtype=DTYPE).reshape(input_shape) + 1
    out_ids = np.arange(output_shape.prod(), dtype=DTYPE).reshape(output_shape) + 1

    # add connections as specified by weights
    paths_left = []
    paths_right = []

    # Boolean mask indicating which weights are not zero, with output channels
    # first (rollaxis) for efficient indexing later on
    nonzero_weights_mask = np.rollaxis(weights != 0, -1)

    # number of channels per group of the input and output
    in_filters_per_group = input_shape[dimension] // groups
    out_filters_per_group = output_shape[dimension] // groups

    # iterates over all output neurons agnostic of channel
    # details can be found in the comments of the 2D case
    for out_cell_and_group in itertools.product(
        *[range(s) for s in output_shape[:dimension]], range(groups)
    ):
        out_cell = np.array(out_cell_and_group[:dimension], dtype=DTYPE)

        # input and output channels of current group lower bound
        group = out_cell_and_group[dimension]
        in_filters_lower = group * in_filters_per_group
        out_filters_lower = group * out_filters_per_group
        
        # top left coordinates of current input neuron grid
        in_neuron_coordinates = (np.array(out_cell) * strides - paddings).astype(DTYPE)

        # next get all input neurons in the grid considering padding.
        lower = np.maximum(0, in_neuron_coordinates)
        upper = in_neuron_coordinates + kernel_shape
        connected_inputs = input_ids[
            lower[0] : upper[0], in_filters_lower : in_filters_lower + in_filters_per_group,
        ]

        # shape of the grid after padding
        used_shape = connected_inputs.shape

        # repeat for each output channel
        connected_inputs_broadcast = np.broadcast_to(
            connected_inputs.reshape((1, used_shape[0], used_shape[1])),
            (out_filters_per_group, used_shape[0], used_shape[1]),
        )

        # map input neurons to corresponding output neurons
        connected_outputs = np.empty_like(connected_inputs_broadcast)
        connected_outputs[:] = out_ids[
            out_cell[0], out_filters_lower : out_filters_lower + out_filters_per_group,
        ].reshape(
            out_filters_per_group, 1, 1
        )

        # build slices to select corresponding portion of weights mask
        lower = np.maximum(0, -1 * in_neuron_coordinates)

        # mask indicating connections with weight not zero in current grid
        mask = nonzero_weights_mask[
            out_filters_lower : out_filters_lower + out_filters_per_group,
            lower[0] : lower[0] + used_shape[0],
            :,
        ]

        # add corresponding input and output neurons to paths
        paths_left.append(connected_inputs_broadcast[mask])
        paths_right.append(connected_outputs[mask])

    # add biases: connections [(0, output_1), (0, output_2),...]
    paths_right.append(out_ids[..., biases != 0].ravel())
    paths_left.append(np.zeros(paths_right[len(paths_right) - 1].shape[0]))

    # return paths
    return np.column_stack(
        (np.concatenate(paths_left), np.concatenate(paths_right))
    ).astype(DTYPE)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef c_get_all_paths_conv2d_layer(
    np.ndarray[DTYPE_t, ndim=1] input_shape,
    np.ndarray[DTYPE_t, ndim=1] output_shape,
    np.ndarray[FTYPE_t, ndim=4] weights,
    np.ndarray[FTYPE_t, ndim=1] biases,
    np.ndarray[DTYPE_t, ndim=1] strides,
    np.ndarray[DTYPE_t, ndim=1] paddings,
    int groups,
):
    cdef int dimension, group, in_filters_lower, out_filters_lower, in_filters_per_group, out_filters_per_group
    cdef np.ndarray[DTYPE_t, ndim=1] kernel_shape
    cdef np.ndarray[DTYPE_t, ndim=3] input_ids
    cdef np.ndarray[DTYPE_t, ndim=3] out_ids
    cdef np.ndarray[BOOL_t, ndim=4] nonzero_weights_mask
    cdef np.ndarray[DTYPE_t, ndim=3] connected_inputs
    cdef np.ndarray[DTYPE_t, ndim=4] connected_inputs_broadcast
    cdef np.ndarray[DTYPE_t, ndim=4] connected_outputs
    cdef np.ndarray[DTYPE_t, ndim=1] lower
    cdef np.ndarray[DTYPE_t, ndim=1] upper
    cdef np.ndarray[DTYPE_t, ndim=1] in_neuron_coordinates
    cdef np.ndarray[BOOL_t, ndim=4] mask
    cdef np.ndarray[DTYPE_t, ndim=1] out_cell 


    dimension = 2  # convolutional layer is 2D
    kernel_shape = np.array([weights.shape[0], weights.shape[1]], DTYPE)  # grid shape

    # ids associated to each input neuron and output neuron
    input_ids = np.arange(input_shape.prod(), dtype=DTYPE).reshape(input_shape) + 1
    out_ids = np.arange(output_shape.prod(), dtype=DTYPE).reshape(output_shape) + 1

    # add connections as specified by weights
    paths_left = []
    paths_right = []

    # Boolean mask indicating which weights are not zero, with output channels
    # first (rollaxis) for efficient indexing later on
    nonzero_weights_mask = np.rollaxis(weights != 0, -1)

    # number of channels per group of the input and output
    in_filters_per_group = input_shape[dimension] // groups
    out_filters_per_group = output_shape[dimension] // groups

    # iterates over all output neurons agnostic of channel
    for out_cell_and_group in itertools.product(
        *[range(s) for s in output_shape[:dimension]], range(groups)
    ):
        out_cell = np.array(out_cell_and_group[:dimension], dtype=DTYPE)

        # input and output channels of current group lower bound
        group = out_cell_and_group[dimension]
        in_filters_lower = group * in_filters_per_group
        out_filters_lower = group * out_filters_per_group
        
        # top left coordinates of current input neuron grid
        in_neuron_coordinates = (np.array(out_cell) * strides - paddings).astype(DTYPE)

        # next get all input neurons in the grid considering padding.

        # e.g. if due to padding, top left corner of grid is at (-1, -1),
        # lower is (0, 0) and upper is (2, 2) for a kernel shape of 3x3, as
        # we need to take the first row and column of the input.
        lower = np.maximum(0, in_neuron_coordinates)
        upper = in_neuron_coordinates + kernel_shape
        connected_inputs = input_ids[
            lower[0] : upper[0],
            lower[1] : upper[1],
            in_filters_lower : in_filters_lower + in_filters_per_group,
        ]

        # shape of the grid after padding
        used_shape = connected_inputs.shape

        # repeat for each output channel to get (out_channel, h, w, in_channel)
        connected_inputs_broadcast = np.broadcast_to(
            connected_inputs.reshape((1, used_shape[0], used_shape[1], used_shape[2])),
            (out_filters_per_group, used_shape[0], used_shape[1], used_shape[2]),
        )

        # map input neurons to corresponding output neurons
        connected_outputs = np.empty_like(connected_inputs_broadcast)
        connected_outputs[:] = out_ids[
            out_cell[0],
            out_cell[1],
            out_filters_lower : out_filters_lower + out_filters_per_group,
        ].reshape(
            out_filters_per_group, 1, 1, 1
        )

        # build slices to select corresponding portion of weights mask
        # e.g. if due to padding, top left corner of grid is at (-1, -1), then
        # lower bound is (1, 1), as the borders of the weights
        # (i.e (0, x), (x, 0)) are matched to padding cells.
        lower = np.maximum(0, -1 * in_neuron_coordinates)

        # mask indicating connections with weight not zero in current grid
        mask = nonzero_weights_mask[
            out_filters_lower : out_filters_lower + out_filters_per_group,
            lower[0] : lower[0] + used_shape[0],
            lower[1] : lower[1] + used_shape[1],
            :,
        ]

        # add corresponding input and output neurons to paths
        paths_left.append(connected_inputs_broadcast[mask])
        paths_right.append(connected_outputs[mask])

    # add biases: connections [(0, output_1), (0, output_2),...]
    paths_right.append(out_ids[..., biases != 0].ravel())
    paths_left.append(np.zeros(paths_right[len(paths_right) - 1].shape[0]))

    # return paths
    return np.column_stack(
        (np.concatenate(paths_left), np.concatenate(paths_right))
    ).astype(DTYPE)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef c_get_all_paths_conv3d_layer(
    np.ndarray[DTYPE_t, ndim=1] input_shape,
    np.ndarray[DTYPE_t, ndim=1] output_shape,
    np.ndarray[FTYPE_t, ndim=5] weights,
    np.ndarray[FTYPE_t, ndim=1] biases,
    np.ndarray[DTYPE_t, ndim=1] strides,
    np.ndarray[DTYPE_t, ndim=1] paddings,
    int groups,
):
    cdef int dimension, group, in_filters_lower, out_filters_lower, in_filters_per_group, out_filters_per_group
    cdef np.ndarray[DTYPE_t, ndim=1] kernel_shape
    cdef np.ndarray[DTYPE_t, ndim=4] input_ids
    cdef np.ndarray[DTYPE_t, ndim=4] out_ids
    cdef np.ndarray[BOOL_t, ndim=5] nonzero_weights_mask
    cdef np.ndarray[DTYPE_t, ndim=4] connected_inputs
    cdef np.ndarray[DTYPE_t, ndim=5] connected_inputs_broadcast
    cdef np.ndarray[DTYPE_t, ndim=5] connected_outputs
    cdef np.ndarray[DTYPE_t, ndim=1] lower
    cdef np.ndarray[DTYPE_t, ndim=1] upper
    cdef np.ndarray[DTYPE_t, ndim=1] in_neuron_coordinates
    cdef np.ndarray[BOOL_t, ndim=5] mask
    cdef np.ndarray[DTYPE_t, ndim=1] out_cell 


    dimension = 3  # convolutional layer is 3D
    kernel_shape = np.array([weights.shape[0], weights.shape[1], weights.shape[2]], DTYPE)  # grid shape

    # ids associated to each input neuron and output neuron
    input_ids = np.arange(input_shape.prod(), dtype=DTYPE).reshape(input_shape) + 1
    out_ids = np.arange(output_shape.prod(), dtype=DTYPE).reshape(output_shape) + 1

    # add connections as specified by weights
    paths_left = []
    paths_right = []

    # Boolean mask indicating which weights are not zero, with output channels
    # first (rollaxis) for efficient indexing later on
    nonzero_weights_mask = np.rollaxis(weights != 0, -1)

    # number of channels per group of the input and output
    in_filters_per_group = input_shape[dimension] // groups
    out_filters_per_group = output_shape[dimension] // groups

    # iterates over all output neurons agnostic of channel
    # details can be found in the comments of the 2D case
    for out_cell_and_group in itertools.product(
        *[range(s) for s in output_shape[:dimension]], range(groups)
    ):
        out_cell = np.array(out_cell_and_group[:dimension], dtype=DTYPE)

        # input and output channels of current group lower bound
        group = out_cell_and_group[dimension]
        in_filters_lower = group * in_filters_per_group
        out_filters_lower = group * out_filters_per_group
        
        # top left coordinates of current input neuron grid
        in_neuron_coordinates = (np.array(out_cell) * strides - paddings).astype(DTYPE)

        # next get all input neurons in the grid considering padding.
        lower = np.maximum(0, in_neuron_coordinates)
        upper = in_neuron_coordinates + kernel_shape
        connected_inputs = input_ids[
            lower[0] : upper[0],
            lower[1] : upper[1],
            lower[2] : upper[2],
            in_filters_lower : in_filters_lower + in_filters_per_group,
        ]

        # shape of the grid after padding
        used_shape = connected_inputs.shape

        # repeat for each output channel
        connected_inputs_broadcast = np.broadcast_to(
            connected_inputs.reshape((1, used_shape[0], used_shape[1], used_shape[2], used_shape[3])),
            (out_filters_per_group, used_shape[0], used_shape[1], used_shape[2], used_shape[3]),
        )

        # map input neurons to corresponding output neurons
        connected_outputs = np.empty_like(connected_inputs_broadcast)
        connected_outputs[:] = out_ids[
            out_cell[0],
            out_cell[1],
            out_cell[2],
            out_filters_lower : out_filters_lower + out_filters_per_group,
        ].reshape(
            out_filters_per_group, 1, 1, 1, 1
        )

        # build slices to select corresponding portion of weights mask
        lower = np.maximum(0, -1 * in_neuron_coordinates)

        # mask indicating connections with weight not zero in current grid
        mask = nonzero_weights_mask[
            out_filters_lower : out_filters_lower + out_filters_per_group,
            lower[0] : lower[0] + used_shape[0],
            lower[1] : lower[1] + used_shape[1],
            lower[2] : lower[2] + used_shape[2],
            :,
        ]

        # add corresponding input and output neurons to paths
        paths_left.append(connected_inputs_broadcast[mask])
        paths_right.append(connected_outputs[mask])

    # add biases: connections [(0, output_1), (0, output_2),...]
    paths_right.append(out_ids[..., biases != 0].ravel())
    paths_left.append(np.zeros(paths_right[len(paths_right) - 1].shape[0]))

    # return paths
    return np.column_stack(
        (np.concatenate(paths_left), np.concatenate(paths_right))
    ).astype(DTYPE)