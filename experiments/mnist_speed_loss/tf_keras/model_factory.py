import tensorflow as tf
from tensorflow import keras as kr

# set random seed
SEED = 42
tf.keras.utils.set_random_seed(SEED)


# Accuracy for mse models
def accuracy(y_true, y_pred):
    y_true = kr.backend.argmax(y_true, axis=1)
    y_pred = kr.backend.argmax(y_pred, axis=1)
    return kr.backend.mean(y_true == y_pred)


def build_mnist(image_shape=(28, 28, 1), loss="categorical_crossentropy"):
    cnn = kr.Sequential()

    # Convolution
    cnn.add(
        kr.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=image_shape,
        )
    )

    # Pooling
    cnn.add(kr.layers.MaxPooling2D(pool_size=(2, 2)))

    # 2nd Convolution
    cnn.add(kr.layers.Conv2D(16, (3, 3), activation="relu"))

    # 2nd Pooling layer
    cnn.add(kr.layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten the layer
    cnn.add(kr.layers.Flatten())
    cnn.add(kr.layers.Dropout(0.5))

    # Fully Connected Layers
    cnn.add(kr.layers.Dense(activation="relu", units=32))
    cnn.add(kr.layers.Dropout(0.5))
    cnn.add(kr.layers.Dense(activation="relu", units=16))
    cnn.add(
        kr.layers.Dense(
            activation="softmax"
            if loss == "categorical_crossentropy"
            else "linear",
            units=10,
        )
    )

    # Compile the Neural network
    cnn.compile(
        optimizer="adam",
        loss=loss,
        metrics=[
            "accuracy" if loss == "categorical_crossentropy" else accuracy
        ],
    )
    cnn.summary()
    return cnn
