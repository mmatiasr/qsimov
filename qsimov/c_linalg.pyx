cimport cython
import numpy as np
import warnings
from scipy.linalg import solve_triangular
from cython cimport floating
cimport numpy as np
np.import_array()

ctypedef fused FTYPE_t:
    np.float32_t
    np.float64_t

ctypedef np.uint8_t BOOL_t


@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cpdef c_make_square_system(
    np.ndarray[FTYPE_t, ndim=2] A,
    np.ndarray[FTYPE_t, ndim=1] b,
):
    cdef np.ndarray[BOOL_t, ndim=2] is_zero
    cdef np.ndarray[BOOL_t, ndim=1] all_zero
    cdef np.ndarray[np.intp_t, ndim=1] consecutive_zeros
    cdef np.ndarray[FTYPE_t, ndim=2] A_new
    cdef np.ndarray[FTYPE_t, ndim=1] b_new
    cdef int previous_unknown
    cdef int unknown
    cdef int n_unknowns
    cdef int curr_idx

    n_unknowns = A.shape[1]

    # counts of first consecutive zeros for each row, however, when the row
    # has all zeros, it returns zero as well, instead of A.shape[1]
    is_zero = A == 0
    consecutive_zeros = np.argmin(is_zero, axis=1)

    # square linear system
    A_new = np.zeros((n_unknowns, n_unknowns))
    b_new = np.zeros(n_unknowns)

    # find first row not all zeros
    curr_idx = consecutive_zeros.shape[0] - 1
    while (curr_idx >= 0
           and consecutive_zeros[curr_idx] == 0
           and is_zero[curr_idx].all()
    ):
        curr_idx -= 1

    # reverse iterate rest of consecutive zero counts
    previous_unknown = -1
    for A_idx in range(curr_idx, -1, -1):
        unknown = consecutive_zeros[A_idx]
        # first row in A (counting from the bottom) that has a certain
        # number of zeros has a matching row in A_new. The rest are zero.
        if unknown != previous_unknown:
            A_new[unknown], b_new[unknown] = A[A_idx], b[A_idx]
        previous_unknown = unknown

    return A_new, b_new


@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cpdef c_back_substitution(
    np.ndarray[FTYPE_t, ndim=2] A,
    np.ndarray[FTYPE_t, ndim=1] b,
    float absolute_cutoff,
    float relative_cutoff,
):
    cdef float relative_cutoff_val
    cdef float zero_threshold
    cdef np.ndarray[FTYPE_t, ndim=1] solution
    cdef np.ndarray[BOOL_t, ndim=1] solvable_unknowns

    # underdetermined system
    if A.shape[0] < A.shape[1]:
        warnings.warn(
            "System is underdetermined, will set zero for some unknowns"
        )
        # use square system for rest of procedure
        A, b = c_make_square_system(A, b)

    # values for which diagonal elements are interpreted as zero
    if relative_cutoff < 0:
        relative_cutoff_val = 0
    else:
        relative_cutoff_val = np.max(np.abs(A)) / relative_cutoff
    zero_threshold = max(absolute_cutoff, relative_cutoff_val)

    # solve unkowns guessing zero for unkowns with zero in diagonal
    solution = np.zeros_like(b)
    solvable_unknowns = np.abs(np.diag(A)) > zero_threshold

    # perhaps all diagonal elements where zero...
    if np.any(solvable_unknowns):
        solution[solvable_unknowns] = solve_triangular(
            A[np.ix_(solvable_unknowns, solvable_unknowns)],
            b[solvable_unknowns],
        )
    else:
        warnings.warn(
            "All elements in diagonal almost zero, result will be all zeros"
        )

    return solution