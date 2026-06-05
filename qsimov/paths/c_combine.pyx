cimport cython
import numpy as np
from itertools import repeat

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
np.import_array()

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.int32

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef fused DTYPE_t:
    np.int32_t
    np.int64_t

@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cpdef c_combine_paths_left_right_hash_join(
    np.ndarray[DTYPE_t, ndim=2] paths_left,
    np.ndarray[DTYPE_t, ndim=2] paths_right
):
    cdef int neuron
    cdef list matches_right_idxs
    cdef int path_left_width = paths_left.shape[1]

    # add auxiliar path with zeros to create bias when joining
    # left: [[0,0,0,...],...] right: [[0,0,1], [0,0,2],...]
    # join: [[0,0,0,0,0,1],...]

    paths_left = np.concatenate(
        (
            np.zeros([1, paths_left.shape[1]], dtype=DTYPE),
            paths_left,
        )
    )

    # associate each neuron to indexes in paths_right
    cdef dict indexes = dict()
    for idx in range(paths_right.shape[0]):
        indexes.setdefault(paths_right[idx, 0], []).append(idx)

    # connect paths
    cdef list left_join_aux = list()
    cdef list right_join = list()
    cdef list right_sizes = list()
    for index_left in range(paths_left.shape[0]):
        neuron = paths_left[index_left, path_left_width - 1]
        
        # lookup neuron in right path
        matches_right_idxs = indexes.get(neuron, None)

        # no match
        if matches_right_idxs is None:
            continue

        # match dimensions
        left_join_aux.append(index_left)
        right_sizes.append(len(matches_right_idxs))
        right_join.extend(matches_right_idxs)

    cdef np.ndarray[DTYPE_t, ndim=1] left_join = np.repeat(
        left_join_aux, right_sizes, axis=0
    ).astype(DTYPE, copy=False)

    if len(left_join) == 0:
        return np.empty(
            (0, path_left_width + paths_right.shape[1] - 1), dtype=DTYPE
        )

    return np.hstack((paths_left[left_join], paths_right[right_join, 1:]))


@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cpdef c_make_join_indices_sorted(
    np.ndarray[DTYPE_t, ndim=1] connections_left,
    np.ndarray[DTYPE_t, ndim=1] connections_right
):

    cdef list indices_left_aux = list()
    cdef list indices_right = list()
    cdef list sizes_right = list()
    cdef int l_idx = 0 
    cdef int r_idx = 0
    cdef int l_idx_next = 0
    cdef int r_idx_next = 0
    cdef int size_left
    cdef int connection_neuron
    cdef int left_len = len(connections_left)
    cdef int right_len = len(connections_right)


    while l_idx < left_len and r_idx < right_len:
        if connections_left[l_idx] < connections_right[r_idx]:
            l_idx += 1
        elif connections_left[l_idx] > connections_right[r_idx]:
            r_idx += 1
        # l_idx and r_idx pointing to same connection neuron
        else:
            # traverse left until no match
            connection_neuron = connections_left[l_idx]
            l_idx_next = l_idx + 1
            while l_idx_next < left_len and connections_left[
                l_idx_next] == connection_neuron:
                l_idx_next += 1
            # number of instances of connection neuron on left paths
            size_left = l_idx_next - l_idx

            # indices where connection neuron ocurred
            indices_left_aux.extend(range(l_idx, l_idx_next))

            # traverse right until no match
            r_idx_next = r_idx + 1
            while r_idx_next < right_len and connections_right[
                r_idx_next] == connection_neuron:
                r_idx_next += 1

            # add right match indices to left connection neuron
            indices_right.extend(list(range(r_idx, r_idx_next)) * size_left)

            # number of instances of connection neuron on the right
            # times size_left
            sizes_right.extend(repeat(r_idx_next - r_idx, size_left))

            l_idx, r_idx = l_idx_next, r_idx_next

    indices_left = np.repeat(indices_left_aux, sizes_right, axis=0)
    return indices_left.astype(DTYPE), np.asarray(indices_right, DTYPE)


@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cpdef c_combine_paths_left_right_sort_join(
    np.ndarray[DTYPE_t, ndim=2] paths_left,
    np.ndarray[DTYPE_t, ndim=2] paths_right
):
    cdef int output_cols
    cdef int [:] indices_left
    cdef int [:] indices_right
    cdef int paths_left_width = paths_left.shape[1] - 1
    cdef np.ndarray[DTYPE_t, ndim=1] connections_left,
    cdef np.ndarray[DTYPE_t, ndim=1] connections_right

    # add auxiliar path with zeros to create bias when joining
    # left: [[0,0,0,...],...] right: [[0,0,1], [0,0,2],...]
    # join: [[0,0,0,0,0,1],...]
    paths_left = np.concatenate(
        (np.zeros((1, paths_left.shape[1]), dtype=DTYPE), paths_left)
    )

    # right paths empty special case
    output_cols = paths_left_width + paths_right.shape[1]
    if len(paths_right) == 0:
        return np.empty((0, output_cols), dtype=np.int32)

    # sort paths
    paths_left = paths_left[paths_left[:, paths_left_width].argsort()]
    paths_right = paths_right[paths_right[:, 0].argsort()]

    # get intersection indexes and number of ocurrences on each array
    connections_left = paths_left[:, paths_left_width]
    connections_right = paths_right[:, 0]
    indices_left, indices_right = c_make_join_indices_sorted(
        connections_left, connections_right
    )

    if indices_left.shape[0] == 0:
        return np.empty((0, output_cols), dtype=np.int32)

    return np.hstack(
        (paths_left[indices_left], paths_right[indices_right, 1:])
    )