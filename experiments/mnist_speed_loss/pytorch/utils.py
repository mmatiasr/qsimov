import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from qsimov.pytorch_qsimov_gradient import AccumulatedEpochTimeTracker
import tempfile
import os

SEED = 42


def init_torch(device, seed=SEED):
    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # set device
    return torch.device("cuda" if device != "cpu" else "cpu")


# create dataloader
def create_dataloader(x, y, batch_size):
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# samples from channels last to channels first
def samples_to_channels_first(samples):
    return np.transpose(samples, [0, -1, *range(1, len(samples.shape) - 1)])


# Accuracy for mse models
def accuracy(y_true, y_pred):
    y_true = y_true.argmax(1)
    y_pred = y_pred.argmax(1)
    return (y_true == y_pred).sum().item() / len(y_true)


def adam(params):
    return optim.Adam(params, lr=0.001)


def build_crossentropy_cnn(image_shape, dataset):
    from experiments.mnist_speed_loss.pytorch.model_factory import (
        load_model,
    )

    # load model based on the loss function and dataset
    return load_model(
        in_channels=image_shape,
        loss="categorical_crossentropy",
        name=dataset,
    )


def build_mse_cnn(image_shape, dataset):
    from experiments.mnist_speed_loss.pytorch.model_factory import (
        load_model,
    )

    # load model based on the loss function and dataset
    return load_model(
        in_channels=image_shape,
        loss="mse",
        name=dataset,
    )


# Cross entropy custom loss (for models WITH Softmax output)
class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(CustomCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        input = torch.clamp(input, self.epsilon, 1.0 - self.epsilon)
        loss = torch.sum(-target * torch.log(input), dim=1)
        return torch.mean(loss)


# Cross entropy loss for models WITHOUT Softmax (raw logits output)
class CrossEntropyFromLogitsLoss(nn.Module):
    def forward(self, input, target):
        import torch.nn.functional as F
        log_probs = F.log_softmax(input, dim=1)
        return torch.mean(-torch.sum(target * log_probs, dim=1))


def train_one_epoch(dataloader, model, loss_function, optimizer, device):
    # training mode (e.g. switches dropout and batchnorm on)
    model.train()
    loss = 0
    corrects = 0
    # fit each batch
    for X, y in dataloader:
        # move data to device
        X, y = X.to(device), y.to(device)

        # compute loss and prediction
        pred = model(X)
        batch_loss = loss_function(pred, y)

        # zero the parameter gradients
        optimizer.zero_grad()
        batch_loss.double().backward()
        optimizer.step()

        # calculate loss and accuracy on this batch batch
        loss += batch_loss.item()
        corrects += (
            (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
        )

    epoch_loss = loss / len(dataloader)
    epoch_accuracy = corrects / len(dataloader.dataset)
    print("Train Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_accuracy))
    return epoch_loss, epoch_accuracy


def validate_one_epoch(dataloader, model, loss_function, device):
    model.eval()
    loss = 0
    corrects = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            now_batch_size = X.shape[0]
            pred = model(X)
            batch_loss = loss_function(pred, y)
            # calculate loss and accuracy by batch
            loss += batch_loss.item() * now_batch_size
            corrects += (
                (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            )
    epoch_loss = loss / len(dataloader.dataset)
    epoch_accuracy = corrects / len(dataloader.dataset)
    print("Test Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_accuracy))
    return epoch_loss, epoch_accuracy


def fit(
    model,
    loss_function,
    optimizer,
    train_dataloader,
    test_dataloader,
    device,
    epochs,
):
    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    model.to(device)
    time_tracker = AccumulatedEpochTimeTracker()
    for epoch in range(epochs):
        time_tracker.on_epoch_begin()
        print(f"Epoch {epoch+1} \n-------------------------------")

        train_loss, train_accuracy = train_one_epoch(
            train_dataloader, model, loss_function, optimizer, device
        )
        history["loss"].append(train_loss)
        history["accuracy"].append(train_accuracy)

        val_loss, val_accuracy = validate_one_epoch(
            test_dataloader, model, loss_function, device
        )
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        time_tracker.on_epoch_end()
    history["time(s)"] = time_tracker.times
    return history


def clone_model_with_weights(model):
    # Save the model to a temporary location
    temp_path = tempfile.mkdtemp()
    torch.save(model, os.path.join(temp_path, "model.pt"))

    # Load the model from the temporary location
    return torch.load(os.path.join(temp_path, "model.pt"), weights_only=False)
