import os
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
from qsimov.pytorch_qsimov_linear_system import PytorchQsimovLinearSystem
from qsimov.pytorch_path_selector import PytorchPathSelector

# Set the random seed for reproducibility
torch.manual_seed(42)

# Load the Boston Housing dataset
california = fetch_california_housing()
x, y = california.data, california.target

# Normalize the data
scaler = StandardScaler()
x = scaler.fit_transform(x)

print("x.shape:", x.shape)
print("y.shape:", y.shape)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Convert data to tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Split the training data into two parts
split_idx = len(x_train) // 2
x_train_1, x_train_2 = x_train[:split_idx], x_train[split_idx:]
y_train_1, y_train_2 = y_train[:split_idx], y_train[split_idx:]

# Define the model architecture
model = nn.Sequential(
    nn.Linear(x_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)

# Define the loss function
loss_function = nn.MSELoss()

# Define the optimizer
optimizer = optim.RMSprop(model.parameters())

# Print a summary of the model architecture
print(model)

# Train the model as usual
batch_size = 32
epochs = 50

for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    running_test_loss = 0.0
    num_train_batches = math.ceil(len(x_train_1) / batch_size)
    num_test_batches = math.ceil(len(x_test) / batch_size)

    # Training phase
    for i in range(0, len(x_train_1), batch_size):
        batch_x = x_train_1[i : i + batch_size]
        batch_y = y_train_1[i : i + batch_size]

        # Forward pass
        pred = model(batch_x)

        # Compute the training loss
        train_loss = loss_function(pred, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        running_train_loss += train_loss.item()

    # Test phase
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            batch_x = x_test[i : i + batch_size]
            batch_y = y_test[i : i + batch_size]

            # Forward pass
            pred = model(batch_x)

            # Compute the test loss
            test_loss = loss_function(pred, batch_y)

            running_test_loss += test_loss.item()

    # Calculate average losses
    train_epoch_loss = running_train_loss / num_train_batches
    test_epoch_loss = running_test_loss / num_test_batches

    # Print progress update
    print(f"Epoch [{epoch + 1}/{epochs}]")
    print(f"  Train Loss: {train_epoch_loss:.4f}")
    print(f"  Test Loss: {test_epoch_loss:.4f}")

# Evaluate the model on the test data
model.eval()
with torch.no_grad():
    pred_test = model(x_test)
    test_loss = loss_function(pred_test, y_test)

print("Test loss:", test_loss.item())


# Create a PytorchQsimovLinearSystem object
qsimov_linear = PytorchQsimovLinearSystem(
    PytorchPathSelector(
        neural_network=model,
        input_shape=x_train.shape[1],
        initial_layer=-1,
        verbose=1,
    ),
    verbose=1,
    qr_shrinkage_factor=10,
)
"""
We used only the last layer of the model as path selector (initial_layer=-1)
because the training data is small (only 404 samples), and it is not convenient
to have more paths than samples as it may lead to numerical instability.

We used back substitution as a solver, but we can also use lstsq, which is much
slower, but may be more accurate or even necessary when the number of paths
is larger than the number of training samples.

We used absolute and relative cutoffs to avoid numerical instability. Some
paths may have very small coefficients, which may lead to elements close to
zero after applying QR in the diagonal of the matrix. The absolute cutoff
and the relative cutoff are separate criteria that stablish the threshold
at which a coefficient is considered zero. Elements in the diagonal of the
matrix set to zero after applying this criteria are handled by setting the
corresponding unknown to zero.

We used a QR shrinkage factor to set the threshold at which the algorithm
applies QR decomposition. Setting it to 10 will apply QR decomposition every
time the current state of the matrix has more than 10 times the number of
samples than the number of unknowns. As QR decomposition is a costly operation,
we can set this value to a larger number if there arent many samples, to avoid
unnecessary QR decompositions.
"""


qsimov_linear.fit(x_train, y_train, batch_size=512)
"""
Everytime a batch is processed, the algorithm adds the corresponding equations
to the internal linear system (or systems if many targets), and applies
QR decomposition if necessary according to the QR shrinkage factor.
"""


pred_test = torch.tensor(qsimov_linear.predict(x_test))

# Evaluate the model on the test data
test_loss = loss_function(pred_test, y_test)
print("Test loss:", test_loss.item())

# Save the qsimov model to a file

# Create a temporary directory
tempdir = os.path.join(tempfile.mkdtemp(), "qsimov_linear_system.qsi")
qsimov_linear.save(tempdir)

# Load the qsimov model from a file
qsimov_linear_system = PytorchQsimovLinearSystem.load(tempdir)

# Evaluate the model on the test data

pred_test = torch.tensor(qsimov_linear_system.predict(x_test))
test_loss = loss_function(pred_test, y_test)
print("Test loss:", test_loss.item())
