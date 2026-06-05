from qsimov.keras_qsimov_linear_system import KerasQsimovLinearSystem
from qsimov.keras_path_selector import KerasPathSelector
import tempfile
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras as kr

# Set the random seed for reproducibility
kr.utils.set_random_seed(42)

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

# Split the training data into two parts
split_idx = len(x_train) // 2
x_train_1, x_train_2 = x_train[:split_idx], x_train[split_idx:]
y_train_1, y_train_2 = y_train[:split_idx], y_train[split_idx:]

# Define the model architecture
model = kr.Sequential(
    [
        kr.layers.Dense(
            64, activation="relu", input_shape=(x_train.shape[1],)
        ),
        kr.layers.Dense(64, activation="relu"),
        kr.layers.Dense(1),
    ]
)

# Compile the model
model.compile(
    loss="mse",
    optimizer="rmsprop",
    metrics=["mae"],
)

# Print a summary of the model architecture
model.summary()

# Train the model as usual
batch_size = 32
epochs = 50
history = model.fit(
    x_train_1,
    y_train_1,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)

# Evaluate the model on the test data
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test mae:", score[1])

# Now, we can apply Qsimov linear system to the model

# Create a QsimovLinearSystem object using previously trained model
# as path selector:
qsimov_linear_system = KerasQsimovLinearSystem(
    path_selector=KerasPathSelector(
        neural_network=model, initial_layer=-1, verbose=1
    ),
    solver="back_substitution",  # alternatively, "lstsq"
    absolute_cutoff=1e-2,
    relative_cutoff=1e6,
    qr_shrinkage_factor=10,
    verbose=1,
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

# Train the model using Qsimov linear system on the whole training data

qsimov_linear_system.fit(x_train, y_train, batch_size=256)
"""
Everytime a batch is processed, the algorithm adds the corresponding equations
to the internal linear system (or systems if many targets), and applies
QR decomposition if necessary according to the QR shrinkage factor.
"""

# Make predictions on the test data
y_pred = qsimov_linear_system.predict(x_test)

# Evaluate the mse and mae on the test data
mse = kr.losses.mean_squared_error(y_test, y_pred)
mae = kr.losses.mean_absolute_error(y_test, y_pred)

print("Test loss (Qsimov):", mse)
print("Test mae (Qsimov):", mae)

# Save the qsimov model to a file

# Create a temporary directory
tempdir = os.path.join(tempfile.mkdtemp(), "qsimov_linear_system.qsi")
qsimov_linear_system.save(tempdir)

# Load the qsimov model from a file
qsimov_linear_system = KerasQsimovLinearSystem.load(tempdir)

# Make predictions on the test data
y_pred = qsimov_linear_system.predict(x_test)

# Evaluate the mse and mae on the test data
mse = kr.losses.mean_squared_error(y_test, y_pred)
mae = kr.losses.mean_absolute_error(y_test, y_pred)

print("Test loss (Qsimov):", mse)
print("Test mae (Qsimov):", mae)
