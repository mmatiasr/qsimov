"""Contains functionality related to the qsimov neural network algorithm
applying linear algebra.
"""

import numpy as np
from qsimov.mixins import LogMixin, NumpyPersistanceMixin
from qsimov.path_selector import DEFAULT_BATCH_SIZE
import qsimov.linalg as qs_alg
from abc import ABC


class QsimovLinearSystem(ABC, LogMixin, NumpyPersistanceMixin):
    """Applies Qsimov algorithm with linear algebra to a neural network
    to replace a subset of the last layers with a flat layer.

    Attributes
    ----------
    equations_: list[array2d]
        List that contains for each output the generated linear systems to be
        solved.
    solutions_: list[array1d]
        List that contains for each output the solutions of the linear system.
    """

    # Variables to be persisted using np.savez_compressed
    _NUMPY_VARIABLES = [
        "equations_",
        "solutions_",
    ]

    def __init__(
        self,
        path_selector,
        solver="back_substitution",
        qr_shrinkage_factor=2,
        verbose=0,
        **kwargs,
    ):
        """Create a QsimovLinearSystem instance. Not to be used directly, as
        this is an abstract class.

        Parameters
        ----------
        path_selector : PathSelector
            Path selector to use for the path selection, already built with
            a neural network.
        solver: str, optional
            Solving strategy to use for the linear systems. 'back_substitution'
            is faster than 'lstsq', but may generate incorrect solutions if
            there are much more generated paths than training samples. By
            default 'back_substitution'.
        qr_shrinkage_factor: float, optional
            Specifies how many times taller (number of equations) than wider
            (number of paths per output) an updated linear system has to be
            after processing a batch on the training process before performing
            a QR factorization to shrink and triangulize it. A QR factorization
            is done at the end of the path selection regardless before solving
            the systems, unless when processing the last batch a QR
            factorization was made.

            Values closer to 1 may be recommendable when there are much more
            samples than paths per output and memory, as the linear system may
            grow too large to fit in memory. By default 2.
        verbose : int, optional
            Degree of verbosity, by default 0, meaning no logs are printed.
        **kwargs: dict
            Arguments passed to the `QsimovLinearSystem.solve` method:
            absolute_cutoff, relative_cutoff, rcond_max, rcond_min.

        """
        # initialize default parameters
        LogMixin.__init__(self, verbose)
        # persistance settings
        NumpyPersistanceMixin.__init__(self, self._NUMPY_VARIABLES)

        self._solver = solver
        self._kwargs = kwargs
        self._qr_shrinkage_factor = qr_shrinkage_factor
        self._path_selector = path_selector

        # allow only linear activation on last weight layer
        self._path_selector.check_last_layer_linear()

        # initialize equation, solution set
        self.reset_equations()

    def solve(self, rcond_max, rcond_min, absolute_cutoff, relative_cutoff):
        """Solve with specified solver all the generated linear systems.

        Parameters
        ----------
        absolute_cutoff: float, optional
            Cutoff value for small elements in generated linear system diagonal
            such that if the absolute value is smaller than absolute_cutoff, it
            is treated as zero and the unknown is set to zero. Only applied
            when solver is back_substitution. By default 1e-6.
        relative_cutoff: float, optional
            Relative cutoff value for small elements in generated linear system
            diagonal such that if the absolute value is relative_cutoff times
            smaller than the largest absolute value in the system, it is
            treated as zero and the unknown is set to zero. Only applied when
            solver is back_substitution. By default 1e6.
        rcond_max : float, optional
            Max value of rcond in numpy.lstsq, by default 1e-3.
        rcond_min : float, optional
            Min value of rcond in numpy.lstsq, by default 1e-8.

        Note
        ----
        When the solver is lstsq, the parameter rcond is estimated as:
        >>> rcond = 1e8 * np.finfo(float).eps * linear_systems_width
        >>> rcond = max(min(rcond, rcond_max), rcond_min)

        Returns
        -------
        list[array1d]
            Solutions of the linear system for each output.
        """
        # compute numerical stability parameter rcond
        rcond = 1e8 * np.finfo(float).eps * max(self.equations_[0][0].shape)

        # always keep in provided range
        rcond = max(min(rcond, rcond_max), rcond_min)

        # solve equations
        return [
            qs_alg.solve(
                equation,
                include_last_row=False,
                solver=self._solver,
                rcond=rcond,
                absolute_cutoff=absolute_cutoff,
                relative_cutoff=relative_cutoff,
            )
            for equation in self.equations_
        ]

    def _process_batch(self, X_path_selection, Y):
        """Process the path selection of X, concatenating it with Y and
        evaluating whether a QR factorization is needed regarding the
        parameter qr_shrinkage_factor stablished on the __init__ method.

        Parameters
        ----------
        X_path_selection : arrayNd
            Path selection of a batch in X.
        Y : array2d
            Expected outputs for each sample in the batch: (samples, outputs).
        """
        # masks of paths for each output
        masks = self._path_selector.output_masks_

        # add independent terms for each output
        new_equations = [
            np.hstack((X_path_selection[:, masks[out]], Y[:, out, None]))
            for out in range(self._path_selector._number_outputs)
        ]

        # update equation set
        self.equations_ = [
            self._r_update(new_equations, out_idx)
            for out_idx in range(self._path_selector._number_outputs)
        ]

    def _r_update(self, new_equations, out_idx):
        """Concatenate new_equations to previously computed equation dataset
        and shrink using QR if the number of equations is larger than
        self.qr_shrinkage_factor times the number of unknowns.

        Parameters
        ----------
        new_equations : list[array2d]
            List of equations for each output.
        out_idx : int
            Which output the update will be perforemed on.

        Returns
        -------
        array2d
            Updated linear system for the output.
        """
        if self.equations_[out_idx] is None:
            expanded = new_equations[out_idx]
        else:
            expanded = np.vstack(
                (self.equations_[out_idx], new_equations[out_idx])
            )

        # check condition to triangularize linear system
        if expanded.shape[0] > self._qr_shrinkage_factor * expanded.shape[1]:
            self._r_transformed_equations[out_idx] = True
            return qs_alg.r_transform(expanded)

        # no compression yet
        self._r_transformed_equations[out_idx] = False
        return expanded

    def fit(self, X, Y, batch_size=None):
        """Fit parameters of path selector, including generation and solving
        of linear systems.

        Parameters
        ----------
        X : arrayNd
            Samples.
        Y : arrayNd
            Expected outputs.
        batch_size : int or None, optional
            Size of each batch to update linear systems. By default None,
            equivalent to 32. Setting this parameter may be necessary when
            there are many samples to avoid using excessive memory. However,
            there will be a computational overhead to compute the linear
            systems one batch at a time. It is recommended using the higher
            batch_size that the memory can fit for faster results.

        Note
        -----
        After fitting each batch, it is evaluated whether a QR factorization is
        needed regarding the parameter qr_shrinkage_factor stablished on the
        __init__ method, which is generaly the most costly operation of the
        training process.

        This method may be called many times to retrain the model.
        """
        # save for prediction
        self._Y_shape = Y.shape

        # reshape if necessary
        if len(Y.shape) != 2:
            Y = Y.reshape(len(Y), -1)

        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE

        # Compute number of batches
        n_batches = (len(X) // batch_size) + int(len(X) % batch_size != 0)

        self._log(f"Computing equations ({n_batches} batches)...")
        for X_batch, Y_batch in self._path_selector.as_numpy_iterator(
            X, Y, batch_size=batch_size
        ):
            self._process_batch(X_batch, Y_batch)

        # r transform linear systems if necessary
        for out_idx, equation in enumerate(self.equations_):
            if not self._r_transformed_equations[out_idx]:
                self.equations_[out_idx] = qs_alg.r_transform(equation)
                self._r_transformed_equations[out_idx] = True

        # solve with least squares
        self._log("Computing solutions...")
        self.solutions_ = self.solve(
            rcond_max=self._kwargs.get("rcond_max", 1e-3),
            rcond_min=self._kwargs.get("rcond_max", 1e-8),
            absolute_cutoff=self._kwargs.get("absolute_cutoff", 1e-6),
            relative_cutoff=self._kwargs.get("relative_cutoff", 1e6),
        )

    def reset_equations(self):
        """Delete generated linear systems after a call to fit."""
        self.equations_ = [None] * self._path_selector._number_outputs
        self._r_transformed_equations = np.full(
            self._path_selector._number_outputs, False
        )
        self.solutions_ = None
        self._Y_shape = None

    def predict(self, X, batch_size=None):
        """Predicts an output for given samples.

        Parameters
        ----------
        X : arrayNd
            Samples.

        batch_size : int or None, optional
            Size of each batch when making path selection. Smaller sizes may be
            required if there are many paths, or if the layers of the neural
            network previous to path selection are very large. When None, a
            size of 32 is used, which is the default.

        Returns
        -------
        arrayNd
            Array of predictions with shape of each prediction as seen during
            fit.

        Raises
        ------
        ValueError
            When the model is not fitted.
        """
        if self.solutions_ is None:
            raise ValueError("Model not fitted")

        # make prediction for each batch
        masks = self._path_selector.output_masks_
        Y_pred = np.empty((len(X), self._path_selector._number_outputs))
        current_idx = 0  # Current batch being processed
        for batch_idx, X_batch in enumerate(
            self._path_selector.as_numpy_iterator(X, batch_size=batch_size)
        ):
            self._log(f"Batch {batch_idx + 1}", log_level=1)

            # predict
            for out_idx in range(self._path_selector._number_outputs):
                Y_pred[
                    current_idx : current_idx + len(X_batch), out_idx
                ] = np.dot(
                    X_batch[:, masks[out_idx]], self.solutions_[out_idx]
                )
            current_idx += len(X_batch)

        # use shape seen during fit for return value
        return Y_pred.reshape(len(Y_pred), *self._Y_shape[1:])
