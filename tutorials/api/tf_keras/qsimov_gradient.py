import numpy as np
import tensorflow as tf
from qsimov.keras_qsimov_gradient import KerasQsimovGradient
from qsimov.keras_path_selector import KerasPathSelector
import tempfile
import os
from tensorflow import keras as kr

# Set the random seed for reproducibility
tf.keras.utils.set_random_seed(42)

# Load the MNIST dataset
mnist = kr.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Normalize the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert the labels to one-hot encodings
y_train = kr.utils.to_categorical(y_train, 10)
y_test = kr.utils.to_categorical(y_test, 10)

# Add a color channel dimension to the data
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Split the training data into two parts
split_idx = len(x_train) // 2
x_train_1, x_train_2 = x_train[:split_idx], x_train[split_idx:]
y_train_1, y_train_2 = y_train[:split_idx], y_train[split_idx:]

# Define the model architecture
model = kr.Sequential(
    [
        kr.Input(shape=(28, 28, 1)),
        kr.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
        ),
        kr.layers.MaxPooling2D(pool_size=(2, 2)),
        kr.layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        kr.layers.MaxPooling2D(pool_size=(2, 2)),
        kr.layers.Flatten(),
        kr.layers.Dense(32, activation="relu"),
        kr.layers.Dense(16, activation="relu"),
        kr.layers.Dense(10, activation="softmax"),
    ],
)

# Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer=kr.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

# Print a summary of the model architecture
model.summary()

# Train the model as usual
batch_size = 32
epochs = 5
history = model.fit(
    x_train_1,
    y_train_1,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
)

# Evaluate the model on the test data
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Train the model with Qsimov

# Create a QsimovGradient object using the previously trained model as
# path selector:

epochs = 3
qsimov_gradient = KerasQsimovGradient(
    KerasPathSelector(model, initial_layer=-2)
)
"""
We use the last two layers of the model to apply the Qsimov algorithm.
Using more layers may be counterproductive because the Qsimov algorithm
works better when there are more samples than paths.

The rest of the arguments are similar to those of keras compile and fit,
and are used to train the one layer model that the Qsimov algorithm generates
from the last two layers of the original model.
"""

qsimov_gradient.compile(
    loss="categorical_crossentropy",
    optimizer=kr.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

history = qsimov_gradient.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
)

# Make predictions
y_pred = qsimov_gradient.predict(x_test)

# Evaluate the predictions
loss = tf.keras.losses.categorical_crossentropy(y_test, y_pred)
accuracy = tf.keras.metrics.categorical_accuracy(y_test, y_pred)

print("Test loss:", loss.numpy().mean())
print("Test accuracy:", accuracy.numpy().mean())

# We can save the model in a file and load it later

# Save the model to a temporary directory
tempdir = os.path.join(tempfile.mkdtemp(), "qsimov_gradient.qsi")
qsimov_gradient.save(tempdir)

# Load the model from the temporary directory setting again objects that
# cannot be persisted with pickle

qsimov_gradient = KerasQsimovGradient.load(tempdir)

# Make predictions
y_pred = qsimov_gradient.predict(x_test)

# Evaluate the predictions
loss = tf.keras.losses.categorical_crossentropy(y_test, y_pred)
accuracy = tf.keras.metrics.categorical_accuracy(y_test, y_pred)

print("Test loss:", loss.numpy().mean())
print("Test accuracy:", accuracy.numpy().mean())
