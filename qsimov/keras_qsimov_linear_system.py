"""Contains functionality related to the qsimov neural network algorithm
applying linear algebra.
"""

import os
from qsimov.keras_path_selector import KerasPathSelector
from qsimov.mixins import NumpyPersistanceMixin
from qsimov.qsimov_linear_system import QsimovLinearSystem


class KerasQsimovLinearSystem(QsimovLinearSystem):
    """Applies Qsimov algorithm with linear algebra to a keras neural network
    to replace a subset of the last layers with a flat layer.

    Attributes
    ----------
    equations_: list[array2d]
        Documented in QsimovLinearSystem.
    solutions_: list[array1d]
        Documented in QsimovLinearSystem.
    """

    def __init__(
        self,
        path_selector,
        solver="back_substitution",
        qr_shrinkage_factor=2,
        verbose=0,
        **kwargs,
    ):
        """Create a KerasQsimovLinearSystem instance.

        Parameters
        ----------
        path_selector : KerasPathSelector
            Instance of KerasPathSelector that will be used to select the
            paths that will be replaced by a flat layer.
        solver: str, optional
            Documented in QsimovLinearSystem.
        qr_shrinkage_factor: float, optional
            Documented in QsimovLinearSystem.
        verbose : int, optional
            Degree of verbosity, by default 0, meaning no logs are printed.
        **kwargs: dict
            Arguments passed to the _solve method. Documented in
            QsimovLinearSystem.
        """
        QsimovLinearSystem.__init__(
            self,
            path_selector,
            solver=solver,
            qr_shrinkage_factor=qr_shrinkage_factor,
            verbose=verbose,
            **kwargs,
        )

    @classmethod
    def load(cls, directory_path, path_selector_device=None):
        """Load KerasQsimovLinearSystem instance from directory.

        Parameters
        ----------
        directory_path : str
            Path to directory.
        path_selector_device : str, optional
            Device to use for the path selector, by default None, which means
            the GPU will be used if available.

        Returns
        -------
        KerasQsimovLinearSystem
            KerasQsimovLinearSystem instance.
        Raises
        ------
        ValueError
            If the specified device is invalid.
        """
        instance = NumpyPersistanceMixin.load(directory_path)

        # replace path selector with actual path selector
        instance._path_selector = KerasPathSelector.load(
            instance._path_selector, device=path_selector_device
        )
        return instance

    def __getstate__(self):
        """Defines which variables are to be pickled.

        Note
        ----
        The directory where numpy and keras variables are stored is taken
        from the internal attribute _save_dir, which will be initialized
        if not previously set (normally when calling save method). This
        method also saves the path selector to a directory inside the
        _save_dir directory and return in the state the path to that
        directory.

        Returns
        -------
        dict
            Variables to be pickled with python pickle module.
        """
        # capture what is normally pickled
        state = NumpyPersistanceMixin.__getstate__(self)

        # persist path selector as a path to a directory
        path_selector_dir = os.path.join(self._save_dir, "path_selector.qsi")
        os.makedirs(path_selector_dir, exist_ok=True)
        KerasPathSelector.save(
            self._path_selector, path_selector_dir, verbose=0
        )
        state["_path_selector"] = path_selector_dir

        return state
