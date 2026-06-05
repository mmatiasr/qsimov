cimport cython
import numpy as np
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


@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cpdef c_select_paths_maxpool_layer(
    np.ndarray[FTYPE_t, ndim=2] flat_inputs_with_bias,
    np.ndarray[DTYPE_t, ndim=2] all_paths_layer 
):
    cdef int num_outputs
    cdef np.ndarray[DTYPE_t, ndim=1] current_max_inputs_idxs
    cdef int path_idx, input_neuron, output_neuron, curr_max_input_idx
    cdef np.ndarray[BOOL_t, ndim=2] path_selections

    # path selection for each sample
    path_selections = np.full(
        (flat_inputs_with_bias.shape[0], all_paths_layer.shape[0]), False
    )

    # number of outputs of this layer
    num_outputs = np.unique(all_paths_layer[:, 1]).shape[0]

    for sample_idx in range(path_selections.shape[0]):

        # associates out neuron to idx of all_paths where input neuron is max
        current_max_inputs_idxs = np.full(num_outputs + 1, -1, dtype=np.int32)

        for path_idx in range(all_paths_layer.shape[0]):
            
            input_neuron = all_paths_layer[path_idx, 0]
            output_neuron = all_paths_layer[path_idx, 1]

            # index of all_paths_layer where input neuron is max for the output
            curr_max_input_idx = current_max_inputs_idxs[output_neuron]

            # max not yet set for this output neuron
            if curr_max_input_idx == -1:
                current_max_inputs_idxs[output_neuron] = path_idx
                path_selections[sample_idx, path_idx] = True
                continue

            # retrieve current maximum input neuron associated to output neuron
            curr_max_input = all_paths_layer[curr_max_input_idx, 0]
            if (flat_inputs_with_bias[sample_idx, curr_max_input] <
                flat_inputs_with_bias[sample_idx, input_neuron]):

                path_selections[sample_idx, curr_max_input_idx] = False
                path_selections[sample_idx, path_idx] = True
                current_max_inputs_idxs[output_neuron] = path_idx

    return path_selections