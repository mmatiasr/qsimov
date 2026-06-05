cimport cython
import numpy as np
cimport numpy as np
np.import_array()


DTYPE = np.int32

ctypedef fused FTYPE_t:
    np.float32_t
    np.float64_t


@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cpdef c_get_all_paths_dense_layer(
    np.ndarray[FTYPE_t, ndim=2] weights,
    np.ndarray[FTYPE_t, ndim=1] biases
):
    cdef np.ndarray[FTYPE_t, ndim=2] full_weights
    cdef np.ndarray[np.int64_t, ndim=1] zeros_left, zeros_right
    cdef int num_paths

    # combine biases and weights in a single matrix
    full_weights = np.vstack((biases[None, :], weights))

    # we have as many paths as zero valued weights
    zeros_left, zeros_right = np.where(full_weights != 0)
    np.add(zeros_right, 1, out=zeros_right)
    num_paths = zeros_left.shape[0]

    # return if there are no paths to compute
    if num_paths == 0:
        return np.empty((0, 2), dtype=DTYPE)

    return np.hstack((zeros_left[:, None], zeros_right[:, None])).astype(DTYPE)


@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cpdef c_get_paths_dense_layer(
    np.ndarray[FTYPE_t, ndim=1] input,
    np.ndarray[FTYPE_t, ndim=2] weights,
    np.ndarray[FTYPE_t, ndim=1] biases
):
    cdef np.ndarray[FTYPE_t, ndim=2] weights_aux

    # auxiliar weights and biases indicating with 0 removed paths
    weights_aux = weights.copy()

    # all connections from 0 inputs are removed
    weights_aux[input == 0] = 0

    # all paths to this layer with not null weight or bias
    return c_get_all_paths_dense_layer(weights_aux, biases)