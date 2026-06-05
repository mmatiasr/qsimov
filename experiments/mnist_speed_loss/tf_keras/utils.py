import os
import time
import tensorflow as tf
import tempfile


keras = tf.keras

SEED = 42


def init_tensorflow(tf, device, seed=SEED):
    # set random seed
    tf.keras.utils.set_random_seed(seed)
    # set device
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class AccumulatedEpochTimeTracker(keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        self.total_time = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        end_time = time.time()
        self.total_time = self.total_time + end_time - self.start_time
        self.times.append(self.total_time)


# Accuracy for mse models
def accuracy(y_true, y_pred):
    y_true = keras.backend.argmax(y_true, axis=1)
    y_pred = keras.backend.argmax(y_pred, axis=1)
    return keras.backend.mean(y_true == y_pred)


def clone_model_with_weights(model):
    # Save the model to a temporary location
    temp_path = tempfile.mkdtemp()
    model.save(temp_path, save_format="tf")

    # Load the model from the temporary location
    return keras.models.load_model(
        temp_path, custom_objects={"accuracy": accuracy}
    )
