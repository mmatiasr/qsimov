"""Contains functionality related to the qsimov neural network algorithm
applying gradient descent.
"""


from qsimov.mixins import LogMixin, NumpyPersistanceMixin
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import time

from qsimov.pytorch_path_selector import PytorchPathSelector


class PytorchQsimovGradient(LogMixin, NumpyPersistanceMixin):
    """Applies Qsimov algorithm with gradient descent to a pytorch neural
    network to replace a subset of the last layers with a flat layer.

    Attributes
    ----------
    model_ : nn.Sequential
        Model that replaces layers where the algorithm is applied.
    """

    # Variables to be persisted using np.savez_compressed
    _NUMPY_VARIABLES = ["_Y_shape"]

    def __init__(
        self,
        path_selector,
        training_loop=None,
        verbose=0,
    ):
        """Create a PytorchQsimovGradient instance.

        Parameters
        ----------
        path_selector : PytorchPathSelector
            Instance of PytorchPathSelector that will be used to select the
            paths that will be replaced by a flat layer.
        training_loop : typing.Callable, optional
            Used to train the model that will replace the original model's
            layers where the algorithm will be applied. The function must
            receive the parameters documented on the fit method of this class.
            If None, the default training loop will be used, which is
            described in the
            `qsimov.pytorch_qsimov_gradient.default_training_loop` function.
        verbose : int, optional
            Verbosity level, by default 0.
        """
        LogMixin.__init__(self, verbose)
        # persistance settings
        NumpyPersistanceMixin.__init__(self, self._NUMPY_VARIABLES)

        # path selector
        self._path_selector = path_selector

        # model that represents the algorithm.
        self.model_ = None

        # training loop
        if training_loop is None:
            self._training_loop = default_training_loop
        else:
            self._training_loop = training_loop

        # output shape
        self._Y_shape = None

    def _make_model(self):
        """Create model that solves path selection dataset using pytorch API.

        Returns
        -------
        torch.nn.Sequential
            One layer model.
        """
        if self.model_ is not None:
            return self.model_

        # get activation layers after last layer that affects path selection
        activation_layers = list(
            map(
                lambda idx: self._path_selector._layers[idx],
                self._path_selector._get_last_activation_layer_idxs(),
            )
        )

        # create model
        connection_mask = self._path_selector.output_masks_
        model = nn.Sequential(
            CustomConnectedLinear(
                in_features=connection_mask.shape[1],
                out_features=connection_mask.shape[0],
                connection_mask=connection_mask,
                bias=False,
                dtype=self._path_selector._right_model_input_dtype,
            ),
            *activation_layers,
        )

        return model

    def fit(
        self,
        X,
        Y,
        X_val=None,
        Y_val=None,
        batch_size=None,
        shuffle=True,
        **kwargs,
    ):
        """Fits weights of the qsimov model to the specified data.

        Parameters
        ----------
        X : arrayNd
            Inputs.
        Y : arrayNd
            Outputs.
        X_val : arrayNd, optional
            Validation inputs.
        Y_val : arrayNd, optional
            Validation outputs.
        batch_size : int, optional
            Batch size to use during training. If not specified, the default
            batch size of the path selector is used.
        shuffle : bool, optional
            Whether to shuffle the data during training.
        **kwargs
            Additional arguments passed to the training loop. The training loop
            will always receive the following arguments:
            - model: the model to be trained.
            - train_dataloader: the dataloader to be used for training, which
            maps inputs to path selection coefficients.
            - validation_dataloader: if validation data is provided, the
            dataloader to be used for validation, which maps inputs to path
            selection coefficients.

        Returns
        -------
        The output of the training loop.
        """
        # save for prediction
        self._Y_shape = Y.shape

        # reshape if necessary
        if len(Y.shape) != 2:
            Y = Y.reshape(len(Y), -1)

        # create model if not previously on previous fit
        self.model_ = self._make_model()

        # map inputs to coefficients
        path_selection_dataset = self._path_selector.as_pytorch_dataloader(
            X, Y, batch_size=batch_size, shuffle=shuffle
        )

        # validation data
        if X_val is not None and Y_val is not None:
            validation_dataset = self._path_selector.as_pytorch_dataloader(
                X_val, Y_val, batch_size=batch_size, shuffle=False
            )
        else:
            validation_dataset = None

        # train
        return self._training_loop(
            model=self.model_,
            train_dataloader=path_selection_dataset,
            validation_dataloader=validation_dataset,
            **kwargs,
        )

    def predict(self, X, batch_size=None, device=None):
        """Uses fitted model to predict outputs for specified inputs.

        Parameters
        ----------
        X : arrayNd
            Inputs.
        batch_size : int, optional
            Batch size to use during prediction. If not specified, the default
            batch size of the path selector is used.
        device : torch.device or str, optional
            Device where the model is placed during prediction. By default
            None, which means that if a GPU is available it will be used,
            otherwise CPU will be used.

        Returns
        -------
        arrayNd
            Predicted outputs, with same shape as seen during train.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")

        # map inputs to coefficients
        dataloader = self._path_selector.as_pytorch_dataloader(
            X, batch_size=batch_size
        )

        # move model to device
        device = PytorchPathSelector._check_device_or_get_default(
            device=device
        )
        self.model_.to(device)

        # predict
        self._log(f"Predicting {len(dataloader)} batches of size {batch_size}")
        predictions = []
        for X_batch in dataloader:
            X_batch = X_batch.to(next(self.model_.parameters()).device)
            predictions.append(self.model_(X_batch).detach().cpu().numpy())

        # concatenate predictions and reshape to original shape
        Y = np.concatenate(predictions)
        return Y.reshape(len(Y), *self._Y_shape[1:])

    @classmethod
    def load(cls, directory_path, path_selector_device=None):
        """Load PytorchQsimovGradient instance from directory.

        Parameters
        ----------
        directory_path : str
            Path to directory.
        path_selector_device : torch.device or str, optional
            Device the path selector uses. By default None, which means that
            if a GPU is available it will be used, otherwise CPU will be used.

        Returns
        -------
        PytorchQsimovGradient
            Qsimov gradient instance.

        Raises
        ------
        ValueError
            If the specified device is invalid.
        """
        instance = NumpyPersistanceMixin.load(directory_path)

        # Load path selector:
        instance._path_selector = PytorchPathSelector.load(
            instance._path_selector, path_selector_device
        )

        # The object was saved with a model
        if instance.model_ is not None:
            # Build model structure:
            instance.model_ = None  # reset current model cache
            instance.model_ = instance._make_model()

            # Load weights:
            path_to_weights = os.path.join(directory_path, "model_weights.pt")
            instance.model_.load_state_dict(torch.load(path_to_weights))

        return instance

    def __getstate__(self):
        """Defines which variables are to be pickled when called with pickle
        and saves the rest with specific methods, e.g. pytorch save.

        Note
        ----
        The directory where numpy and pytorch variables are stored is taken
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

        # save path selector as a path to the directory where it is saved
        path_selector_dir = os.path.join(self._save_dir, "path_selector.qsi")
        os.makedirs(path_selector_dir, exist_ok=True)
        PytorchPathSelector.save(
            self._path_selector, path_selector_dir, verbose=0
        )
        state["_path_selector"] = path_selector_dir

        # if there is no neural network, we are finished
        if self.model_ is None:
            return state

        # save neural network in a directory, persist the attribute as a path
        model_dir = os.path.join(self._save_dir, "model_weights.pt")
        state["model_"] = model_dir

        # save model weights
        torch.save(self.model_.state_dict(), model_dir)

        return state


###############################################################################
# CUSTOM LAYERS
###############################################################################


class CustomConnectedLinear(nn.Linear):
    """Implements a pytorch linear layer with a specific connection pattern."""

    def __init__(
        self, in_features, out_features, connection_mask=None, **kwargs
    ):
        """Create a linear layer with a custom connection pattern.

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        connection_mask : array2d or None, optional
            A boolean array indicating which weights are kept, by default None.
        **kwargs : dict
            Keyword arguments to be passed to pytorch Linear class.
        """
        super().__init__(in_features, out_features, **kwargs)

        assert connection_mask is None or connection_mask.ndim == 2

        # set default connection mask
        if connection_mask is None:
            connection_mask = np.ones((out_features, in_features))

        # convert to torch tensor
        connection_mask = torch.tensor(
            connection_mask, dtype=self.weight.dtype, requires_grad=False
        )

        # register as a parameter
        self.connection_mask = nn.Parameter(
            connection_mask, requires_grad=False
        )

    def forward(self, inputs):
        """Use the layer to map an input.

        Parameters
        ----------
        inputs : array2d
            Sets of inputs.

        Returns
        -------
        array2d
            Outputs.
        """
        return F.linear(inputs, torch.mul(self.weight, self.connection_mask))


###############################################################################
# DEFAULT TRAINING LOOP
###############################################################################


def train_one_epoch(
    model, dataloader, loss_function, optimizer, metrics, device
):
    """
    Trains a PyTorch model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be trained.
    dataloader : torch.utils.data.DataLoader
        DataLoader for the training data.
    loss_function : torch.nn.Module
        The loss function.
    optimizer : torch.optim.Optimizer
        The optimizer for the model.
    metrics : list of namedcallable
        Metrics functions to evaluate during training.
    device : str
        Device where the model is stored.

    Returns
    -------
    float
        Average loss for the epoch.
    dict
        Dictionary of average metrics for this epoch. The keys are the names of
        the metrics and the values are the average metric values. If no metrics
        were provided, returns None.

    """
    # set model to train mode
    model.train()

    # initialize metrics and loss
    metrics_sum = {metric.__name__: 0 for metric in metrics}
    loss_sum = 0
    total_batches = len(dataloader)

    # iterate over batches
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        for metric in metrics:
            metrics_sum[metric.__name__] += metric(targets, outputs)

    avg_loss = loss_sum / total_batches
    avg_metrics = {
        metric: metrics_sum[metric] / total_batches for metric in metrics_sum
    }

    return avg_loss, avg_metrics


def validate_one_epoch(model, dataloader, loss_function, metrics, device):
    """
    Validates a PyTorch model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be validated.
    dataloader : torch.utils.data.DataLoader
        DataLoader for the validation data.
    loss_function : torch.nn.Module
        The loss function.
    metrics : list of namedcallable
        Metrics functions to evaluate during validation.
    device : str
        Device where the model is stored.

    Returns
    -------
    float
        Average loss for the epoch.
    dict
        Dictionary of average metrics for this epoch. The keys are the names of
        the metrics and the values are the average metric values. If no metrics
        were provided, returns None.

    """
    # set model to evaluation mode
    model.eval()

    # initialize metrics and loss
    metrics_sum = {metric.__name__: 0 for metric in metrics}
    loss_sum = 0
    total_batches = len(dataloader)

    # compute metrics and loss
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss_sum += loss.item()

            for metric in metrics:
                metrics_sum[metric.__name__] += metric(targets, outputs)

    # compute average metrics and loss
    avg_loss = loss_sum / total_batches
    avg_metrics = {
        metric: metrics_sum[metric] / total_batches for metric in metrics_sum
    }

    return avg_loss, avg_metrics


def print_progress(
    epoch, epochs, train_loss, train_metrics, val_loss=None, val_metrics=None
):
    """
    Prints progress after each epoch.

    Parameters
    ----------
    epoch : int
        The current epoch.
    epochs : int
        The total number of epochs.
    train_loss : float
        Training loss for the current epoch.
    train_metrics : dict
        Dictionary of training metrics for the current epoch.
    val_loss : float, optional
        Validation loss for the current epoch.
    val_metrics : dict, optional
        Dictionary of validation metrics for the current epoch.

    """
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train loss: {train_loss}")
    if train_metrics:
        for metric, value in train_metrics.items():
            print(f"Train {metric}: {value}")
    if val_loss:
        print(f"Validation loss: {val_loss}")
        for metric, value in val_metrics.items():
            print(f"Validation {metric}: {value}")
    print("-----------------")


# accumulated epoch time
class AccumulatedEpochTimeTracker:
    """
    Accumulates the time taken for each epoch.
    """

    def __init__(self):
        """
        Initializes the tracker.
        """
        self.times = []
        self.total_time = 0

    def on_epoch_begin(self):
        """
        Starts the timer for the epoch.
        """
        self.start_time = time.time()

    def on_epoch_end(self):
        """
        Stops the timer for the epoch and accumulates the time taken.
        """
        end_time = time.time()
        self.total_time = self.total_time + end_time - self.start_time
        self.times.append(self.total_time)


def default_training_loop(
    epochs,
    model,
    loss_function,
    optimizer,
    train_dataloader,
    validation_dataloader=None,
    metrics=None,
    verbose=True,
    device=None,
):
    """
    Trains a PyTorch model for a specified number of epochs.

    Parameters
    ----------
    epochs : int
        Number of epochs to train the model.
    model : torch.nn.Module
        The PyTorch model to be trained.
    loss_function : torch.nn.Module
        The loss function. Should reduce with reduction='mean'.
    optimizer : lambda params: torch.optim.Optimizer
        A function that takes the model parameters and returns an optimizer.
    train_dataloader : torch.utils.data.DataLoader
        DataLoader for the training data.
    validation_dataloader : torch.utils.data.DataLoader, optional
        DataLoader for validation data. If None, no validation is performed.
    metrics : list of namedcallable, optional
        Metrics functions to evaluate during training and validation, should
        have a __name__ attribute. Should take the targets and outputs as
        arguments and return a float.
    verbose : bool, optional
        If True, prints information after each epoch.
    device : torch.device or str, optional
        Device to use for training and validation. If None, defaults to
        cuda if available, otherwise cpu.

    Returns
    -------
    dict
        History of training and validation loss, and any additional metrics.

    """
    # create history dictionary
    history = {"loss": [], "val_loss": []}

    # set device
    device = PytorchPathSelector._check_device_or_get_default(device)
    model.to(device)

    # create optimizer from model parameters
    optimizer = optimizer(model.parameters())
    if metrics is None:
        metrics = []

    # add metrics to history
    for metric in metrics:
        history[metric.__name__] = []

        # add validation metrics to history
        if validation_dataloader:
            history["val_" + metric.__name__] = []

    # stores the timestamps with t_0 = 0
    train_times = AccumulatedEpochTimeTracker()

    for epoch in range(epochs):
        # initialize epoch time
        train_times.on_epoch_begin()

        # train for one epoch
        train_loss, train_metrics = train_one_epoch(
            model, train_dataloader, loss_function, optimizer, metrics, device
        )

        # add training metrics and loss to history
        history["loss"].append(train_loss)
        for metric, value in train_metrics.items():
            history[metric].append(value)

        # validate model
        if validation_dataloader:
            val_loss, val_metrics = validate_one_epoch(
                model, validation_dataloader, loss_function, metrics, device
            )
            history["val_loss"].append(val_loss)
            for metric, value in val_metrics.items():
                history["val_" + metric].append(value)
        else:
            val_loss, val_metrics = None, None

        # update epoch time
        train_times.on_epoch_end()

        # print progress
        if verbose:
            print_progress(
                epoch, epochs, train_loss, train_metrics, val_loss, val_metrics
            )

    history["time(s)"] = train_times.times
    return history
