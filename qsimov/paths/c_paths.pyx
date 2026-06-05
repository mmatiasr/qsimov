cimport cython
import numpy as np
cimport numpy as np
np.import_array()

ctypedef fused FTYPE_t:
    np.float32_t
    np.float64_t

ctypedef fused DTYPE_t:
    np.int32_t
    np.int64_t

ctypedef np.uint8_t BOOL_t

@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cpdef c_non_zero_input_select_paths(
    np.ndarray[FTYPE_t, ndim=2] flat_inputs_with_bias,
    np.ndarray[DTYPE_t, ndim=2] all_paths
):
    # calculate the active paths between all paths
    return flat_inputs_with_bias[:, all_paths[:, 0]] != 0


@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
def c_retrieve_coefficients(
    np.ndarray[BOOL_t, ndim=2] select_masks,
    np.ndarray[DTYPE_t, ndim=1] paths_input_neurons,
    np.ndarray[FTYPE_t, ndim=2] flat_inputs_with_bias
):
    cdef np.ndarray[FTYPE_t, ndim=2] coefficients
    cdef np.ndarray[BOOL_t, ndim=1] select_mask
    cdef int sample_idx

    # initialize base coefficient matrix
    coefficients = np.zeros(
        (select_masks.shape[0], paths_input_neurons.shape[0]), dtype=flat_inputs_with_bias.dtype
    )

    for sample_idx in range(select_masks.shape[0]):
        select_mask = select_masks[sample_idx]
        coefficients[
            sample_idx, select_mask
        ] = flat_inputs_with_bias[sample_idx, paths_input_neurons[select_mask]]
    
    return coefficients