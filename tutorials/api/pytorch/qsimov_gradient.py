import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import dataloader, TensorDataset
from torchvision import datasets, transforms
from qsimov.pytorch_qsimov_gradient import PytorchQsimovGradient
from qsimov.pytorch_path_selector import PytorchPathSelector
import tempfile
import os
import random

# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# Load the MNIST dataset

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)

test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transforms.ToTensor()
)

# Convert the datasets to numpy arrays
x_train = train_dataset.data.numpy()
y_train = train_dataset.targets.numpy()
x_test = test_dataset.data.numpy()
y_test = test_dataset.targets.numpy()

# Normalize the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add a color channel dimension to the data
x_train = np.expand_dims(x_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)

# Convert the labels to one-hot encodings
num_classes = 10
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

# Split the training data into two parts
split_idx = len(x_train) // 2
x_train_1, x_train_2 = x_train[:split_idx], x_train[split_idx:]
y_train_1, y_train_2 = y_train[:split_idx], y_train[split_idx:]


# Print the shapes of the arrays
print("x_train_1 shape:", x_train_1.shape)
print("y_train_1 shape:", y_train_1.shape)
print("x_train_2 shape:", x_train_2.shape)
print("y_train_2 shape:", y_train_2.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 16, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10),
    nn.Softmax(dim=1),
)

# Intialize weights for layers
for layer_idx, layer in enumerate(model.children()):
    # Gain = 1 appropiate for Conv2d and softmax
    if (
        isinstance(layer, nn.Conv2d)
        or layer_idx == len(list(model.children())) - 2
    ):
        gain = 1
    # Gain = sqrt(2) appropiate for ReLU
    elif isinstance(layer, nn.Linear):
        gain = nn.init.calculate_gain("relu")

    # Initialize the weights and biases according to the gain
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

# Print a summary of the model architecture
print(model)


# Define a custom cross entropy loss function that emulates the
# behaviour of the cross entropy loss function in keras, by accepting
# probabilities as input and not logits
class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(CustomCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        input = torch.clamp(input, self.epsilon, 1.0 - self.epsilon)
        loss = torch.sum(-target * torch.log(input), dim=1)
        return torch.mean(loss)


# Train the model as usual
batch_size = 32
epochs = 5
loss_function = CustomCrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Make a torch dataset over the numpy arrays
train_ds_1 = TensorDataset(
    torch.as_tensor(x_train_1), torch.as_tensor(y_train_1)
)
train_loader_1 = dataloader.DataLoader(train_ds_1, batch_size=batch_size)
test_ds = TensorDataset(torch.as_tensor(x_test), torch.as_tensor(y_test))
test_loader = dataloader.DataLoader(test_ds, batch_size=512, shuffle=False)

# Training loop
for epoch in range(epochs):
    # Training on first half of the data
    model.train()
    for data, target in train_loader_1:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

    # Evaluate on test dataset
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    total_batches = 0
    with torch.no_grad():
        for data, target in test_loader:
            # loss
            output = model(data)
            test_loss += loss_function(output, target).item()

            # accuracy
            pred = torch.argmax(output, dim=1)
            target = torch.argmax(target, dim=1)
            correct += torch.eq(pred, target).float().sum().item()

            total_batches += 1
            total += len(data)

        test_loss /= total_batches
        test_accuracy = (100.0 * correct) / total

    print(
        f"Epoch {epoch+1}: Test Loss = {test_loss:.4f}, "
        f"Accuracy = {test_accuracy:.2f}%"
    )

pred = model(torch.as_tensor(x_test)).argmax(1)
print(
    "Accuracy on test set:",
    100.0
    * (pred.eq(torch.as_tensor(y_test).argmax(1))).type(torch.float).mean(),
)

# Train the model with Qsimov

# Create a QsimovGradient object using the previously trained model as
# path selector:

epochs = 5
qsimov_gradient = PytorchQsimovGradient(
    PytorchPathSelector(
        neural_network=model,
        input_shape=(1, 28, 28),
        initial_layer=-4,
    )
)
"""
We use the last two layers of the model to apply the Qsimov algorithm.
Using more layers may be counterproductive because the Qsimov algorithm
works better when there are more samples than paths. Note that the
initial layer is specified as -4 beacuse we have to take into account
the activation layers.

We also specify the input shape of the model. This is necessary because
the Qsimov algorithm needs to know the input shape of the model to
generate the paths, and the pytorch model does not have an input shape
attribute.
"""


# accuracy in pytorch
def accuracy(y_true, y_pred):
    y_true = torch.argmax(y_true, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    return torch.mean((y_true == y_pred).float())


history = qsimov_gradient.fit(
    x_train,
    y_train,
    X_val=x_test,
    Y_val=y_test,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
    loss_function=CustomCrossEntropyLoss(),
    metrics=[accuracy],
)

print("Structure of the one layer model:")
print(qsimov_gradient.model_)


# Make predictions
y_pred = torch.as_tensor(qsimov_gradient.predict(x_test))
y_test = torch.as_tensor(y_test)

# Evaluate the predictions
test_accuracy = accuracy(y_test, y_pred)

print("Test accuracy:", test_accuracy)

# We can save the model in a file and load it later

# Save the model to a temporary directory
tempdir = os.path.join(tempfile.mkdtemp(), "qsimov_gradient.qsi")
qsimov_gradient.save(tempdir)

# Load the model from the temporary directory
qsimov_gradient = PytorchQsimovGradient.load(tempdir)

# Make predictions
y_pred = torch.as_tensor(qsimov_gradient.predict(x_test))

# Evaluate the predictions
test_accuracy = accuracy(y_test, y_pred)
print("Test accuracy:", test_accuracy)
