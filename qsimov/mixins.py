"""General util functions for the qsimov library."""
import numpy as np
import os
import tempfile
import pickle


class LogMixin:
    """Allows logging messages to stdout filtering them based on a verbosity
    parameter.
    """

    def __init__(self, verbose=0):
        """Creates a LogMixin object, which allows printing messages to stdout
        when their log_level is lower than the threshold set by the verbose
        parameter.

        Parameters
        ----------
        verbose : int, optional
            Threshold such that if the _log method is called with
            log_level < verbose, the message is printed, by default 0.
        """
        self._verbose = verbose

    def _log(self, *message, log_level=0):
        """Log one or many elements to stdout.

        Parameters
        ----------
        *message : tuple
            Parameters to be printed, as in calls to print().
        log_level : int, optional
            Verbosity level of elements to be printed, by default 0.
        """
        if self._verbose > log_level:
            print(*message)


class NumpyPersistanceMixin:
    """Defines methods __getstate__ and __setstate__ for the pickle module
    so that a list of provided numpy variables can be saved separately
    instead of using the pickle protocol.
    """

    def __init__(self, numpy_variables, default_save_dir=None):
        """Creates NumpyPersistanceMixin instance.

        Parameters
        ----------
        numpy_variables : list
            Name of the variables which are either numpy arrays or lists
            of numpy arrays.
        default_save_dir : str, optional
            Default directory where to save compressed numpy variables, if set
            to None, will use a temporary directory. By
            default None.
        """
        # where variables are persisted
        self._save_dir = default_save_dir
        self._numpy_variables = numpy_variables

    def __getstate__(self):
        """Defines which variables are to be pickled when called with pickle
        and saves the rest with specific methods, e.g. keras clone_model.

        Note
        ----
        The directory where numpy and keras variables are stored is taken
        from the internal attribute _save_dir, which will be initialized
        if not previously set (normally when calling save method).

        Returns
        -------
        dict
            Variables to be pickled with python pickle module.
        """
        # if not previously defined, create a persistance directory
        if self._save_dir is None:
            self._save_dir = tempfile.mkdtemp()
        self._save_dir = os.path.abspath(self._save_dir)

        # capture what is normally pickled
        state = self.__dict__.copy()

        # save numpy arrays
        savez_kwargs = {}
        for name in self._NUMPY_VARIABLES:
            variable = getattr(self, name)
            # list or tuple
            if type(variable) is not np.ndarray and variable is not None:
                savez_kwargs.update(
                    {
                        f"__iter_{idx}_{name}": variable[idx]
                        for idx in range(len(variable))
                    }
                )
                state[name] = [None] * len(variable)
            else:
                savez_kwargs[name] = variable
                # dont pickle this variable contents
                state[name] = None

        # save to file
        np.savez_compressed(
            os.path.join(self._save_dir, "numpy_variables"), **savez_kwargs
        )

        return state

    def __setstate__(self, state):
        """Updates class attributes according to state.

        Parameters
        ----------
        state : dict
            Maps class attributes to values.

        Raises
        ------
        ValueError
            Internal attribute _save_dir not in state.
        """
        save_dir = state["_save_dir"]
        if save_dir is None:
            raise ValueError("Persistance directory was not set.")

        # retrieve numpy variables
        numpy_variables = np.load(
            os.path.join(save_dir, "numpy_variables.npz"), allow_pickle=True
        )
        for name in list(numpy_variables):
            value = numpy_variables[name]

            # possibly an object when list of numpy arrays
            if value.shape == ():
                value = value.item()

            # if it was a list of numpy arrays, decode name
            if name.startswith("__iter_"):
                # retrieve original variable name and position in the iterator
                idx = len("__iter_")
                position = ""
                while name[idx] != "_":
                    position += name[idx]
                    idx += 1

                state[name[idx + 1 :]][int(position)] = value
            else:
                state[name] = value

        # update state
        self.__dict__.update(state)

    def save(self, directory_path, verbose=1):
        """Save class to a directory.

        Parameters
        ----------
        directory_path : str
            Path to the directory where the class will be saved.
        verbose : int, optional
            Verbosity level, by default 1.
        """
        if not directory_path.endswith(".qsi"):
            directory_path += ".qsi"

        os.makedirs(directory_path, exist_ok=True)
        self._save_dir = directory_path

        with open(os.path.join(directory_path, "py_objects.pkl"), "wb") as f:
            pickle.dump(self, f)

        if verbose > 0:
            class_name = self.__class__.__name__
            print(f"Saved {class_name} instance to {self._save_dir}")

    @classmethod
    def load(cls, directory_path):
        """Load class instance from directory.

        Parameters
        ----------
        directory_path : str
            Path to directory.

        Returns
        -------
        object
            Object from a type that inherits from this mixin.
        Raises
        ------
        FileNotFoundError
            If the directory does not exist.
        """
        with open(os.path.join(directory_path, "py_objects.pkl"), "rb") as f:
            saved_object = pickle.load(f)
        return saved_object
