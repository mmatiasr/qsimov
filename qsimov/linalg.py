"""Utilities related to linear algebra."""
import numpy as np
from qsimov.c_linalg import c_make_square_system, c_back_substitution


def r_transform(AB):
    """Transform linear system AB to R matrix in QR factorization.

    Parameters
    ----------
    AB: array2d
        Input matrix of shape (M, N)
        where A|B dependent/independent terms of a system.

    Returns
    -------
    array2d
        Upper triangular matrix of shape (K, N) having K=min(M, N).

    Raises
    ------
    np.linalg.LinAlgError
        If factoring fails.
    """
    # add new equations and r transform
    return np.linalg.qr(AB, mode="r")


def _make_square_system(A, b):
    """Transform the system (A, b) into a square linear system by removing
    incompatible equations and adding rows of zeros to make it square.

    Parameters
    ----------
    A: array2d
        Matrix of coefficients.
    b: array1d
        Vector of independent terms.

    Returns
    -------
    A: array2d
        Matrix of coefficients of the square system.
    b: array1d
        Vector of constant of square system.

    Examples
    --------
    >>> A = np.array([[1, 2, 3, 4], [0, 1, 0, 0], [0, 0, 0, 2]])
    >>> b = np.array([1, 2, 3])
    >>> _make_square_system(A, b)
    (
        array([
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
        ]),
        array([1.0, 2.0, 3.0, 0.0])
    )

    Notes
    -----
    To build the system of equations, it is necessary to transform the (M, N)
    matrix into a square matrix (N, N). To achieve this, the number of
    consecutive zeros in each row of the original matrix is counted. This
    number represents the position that each row should occupy in the square
    matrix. If many rows have the same number of consecutive zeros, the last of
    them is chosen. If a row does not appear in the original matrix, a row of
    zeros is added to the square matrix. For example:

    >>> A = [
    ...     [x x x x x] # <- 0
    ...     [0 x x x x] # <- 1
    ...     [0 y y y y] # <- 1
    ...     [0 0 0 x x] # <- 3
    ... ]

    >>> b = [x x y x]

    >>> number_consecutives_zeros = [0, 1, 1, 3]

    >>> result = (
    ...     A = [
    ...         [x x x x x] <- 0
    ...         [0 y y y y] <- 1
    ...         [0 0 0 0 0] <- 2
    ...         [0 0 0 x x] <- 3
    ...         [0 0 0 0 0] <- 4
    ...     ],
    ...     b = [x y 0 x 0]
    ... )
    """
    A = np.asanyarray(A)
    b = np.asanyarray(b)
    return c_make_square_system(
        np.asfarray(A, A.dtype), np.asfarray(b, b.dtype)
    )


def back_substitution(A, b, absolute_cutoff=0, relative_cutoff=np.inf):
    """Solve the linear system Ax = b using back substitution.

    Parameters
    ----------
    A : array2d
        Coefficient matrix of shape (M, N).
    b : array1d
        Right-hand side vector of shape (M,).
    absolute_cutoff : float, optional
        Absolute cutoff value for the magnitude of entries in the solution.
        Entries with magnitude smaller than this value will be set to zero.
        Default is 0.
    relative_cutoff : float, optional
        Relative cutoff value for the magnitude of entries in the solution.
        Entries with magnitude smaller than this value times the magnitude of
        the largest entry will be set to zero. Default is infinity
        (i.e. no relative cutoff).

    Returns
    -------
    x : array1d
        Solution vector of shape (N,).

    Note
    -----
    This function assumes that the matrix A is upper triangular but maybe not
    square, which will happen when A, b is the R in a QR factorization of an
    underdetermined system. If A is not a square matrix, some unknowns are set
    to zero and ignored to make it square by a call to _make_square_system.
    If an unknwown has zeros in its corresponding diagonal, it is assigned to
    zero.
    """

    if relative_cutoff == np.inf:
        relative_cutoff = -1
    A, b = np.asanyarray(A), np.asanyarray(b)
    return c_back_substitution(
        np.asfarray(A, A.dtype),
        np.asfarray(b, b.dtype),
        absolute_cutoff,
        relative_cutoff,
    )


def solve(AB, include_last_row=True, solver="lstsq", **kwargs):
    """Solve the linear system Ax = b using least-squares or back-substitution
    method.

    Parameters
    ----------
    AB : array2d
        Coefficient matrix of shape(M, N) where M is the number
        of equations and N is the number of variables.
    include_last_row : bool, optional
        Include last row in the matrix AB. Default is True.
    solver : string, optional
        Method to solve the system of equations. Possible values are
        "lstsq" (default) for least-squares or "back_substitution" for
        back-substitution method.
    **kwargs : dict
        Arbitrary keyword arguments that can be passed to either
        np.linalg.lstsq() or back_substitution() function.

    Returns
    -------
    (array2d, array1d)
        Tuple (A, b) with A matrix of coefficients of the square system and
        b the vector of independent terms.

    Notes
    -----
    If include_last_row is False, which should be set when the input system
    is the R matrix in a QR factorization of an original system (A|B), the
    last row of the system is deleted if (A|B) was overdetermined.

    Raises
    ------
    np.linalg.LinAlgError
        If computation does not converge (only for least-squares method).
    """
    if not include_last_row:
        # overdetermined system
        if AB.shape[0] == AB.shape[1]:
            A, b = AB[:-1, :-1], AB[:-1, -1]
        # underdetermined or determinate system, do not remove last row
        else:
            A, b = AB[:, :-1], AB[:, -1]

    # should not remove last row if underdetermined
    else:
        A, b = AB[:, :-1], AB[:, -1]

    # get parameters for back-substitution
    absolute_cutoff = kwargs.pop("absolute_cutoff", 0)
    relative_cutoff = kwargs.pop("relative_cutoff", np.inf)

    # least squares solution
    if solver == "lstsq":
        return np.linalg.lstsq(A, b, **kwargs)[0]

    return back_substitution(
        A, b, absolute_cutoff=absolute_cutoff, relative_cutoff=relative_cutoff
    )


def _qr_transform_linear_system(AB):
    """QR transform A and output R, Q.T · B.

    Parameters
    ----------
    AB : array2d
        Horizontal concatenation of dependent and independent term.

    Returns
    -------
    (array2d, array1d)
        R, Q.T · B after where Q, R is QR transformation of A.

    Raises
    ------
    np.linalg.LinAlgError
        If factoring fails.
    """
    A, B = AB[:, :-1], AB[:, -1]
    Q, R = np.linalg.qr(A)
    return np.hstack((R, (Q.T @ B).reshape(-1, 1)))


def _qr_update(AB_old, AB_new):
    """Update a previously QR transformation A_old, B_old, where
    A_old, B_old = R, Q.T · B with Q,R a QR transformation of an initial
    coefficient matrix A and B the independent terms.

    Parameters
    ----------
    AB_old : array2d
        Concatenated A_old (coefficients) and B_old (independent term) with:
            A_old = R with Q, R a QR transformation of an initial
            coefficient matrix A.
            B_old = Q.T · B Q, R a QR transformation of an initial
            coefficient matrix A and B the independent term for A.
    AB_new : array2d
        New dependent/independent terms to update linear system.

    Returns
    -------
    array2d
        An updated linear system with the same properties as A_old, B_old.
    Raises
    ------
    np.linalg.LinAlgError
        If factoring fails.
    """
    # add new equations and qr transform
    return _qr_transform_linear_system(np.vstack((AB_old, AB_new)))
