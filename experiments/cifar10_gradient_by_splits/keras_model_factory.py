import tensorflow as tf
from tensorflow import keras as kr
from keras.applications.vgg16 import VGG16
import argparse
from experiments.cifar10_gradient_by_splits.train_keras_models import (
    TrainModelsParser,
)
from experiments.path_utils import (
    get_cifar10_gradient_by_splits_results_initial_weights_dir,
)
import pickle
import numpy as np
import os

# set random seed
SEED = 42
tf.keras.utils.set_random_seed(SEED)


def pretrained_build_vgg16():
    vgg16 = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(32, 32, 3),
        classes=10,
    )

    def get_top_layers():
        return [
            kr.layers.Flatten(),
            kr.layers.Dense(units=512, activation="relu"),
            kr.layers.Dropout(0.5),
            kr.layers.Dense(units=512, activation="relu"),
            kr.layers.Dropout(0.5),
            kr.layers.Dense(units=10, activation="softmax"),
        ]

    cnn = kr.Sequential(vgg16.layers + get_top_layers())
    cnn.summary()

    return cnn


def build_vgg16(include_top=True):
    model = kr.models.Sequential()
    model.add(kr.layers.Input(shape=(32, 32, 3)))

    # Block 1
    model.add(
        kr.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="block1_conv1"
        )
    )
    model.add(
        kr.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="block1_conv2"
        )
    )
    model.add(
        kr.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")
    )

    # Block 2
    model.add(
        kr.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same", name="block2_conv1"
        )
    )
    model.add(
        kr.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same", name="block2_conv2"
        )
    )
    model.add(
        kr.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")
    )

    # Block 3
    model.add(
        kr.layers.Conv2D(
            256, (3, 3), activation="relu", padding="same", name="block3_conv1"
        )
    )
    model.add(
        kr.layers.Conv2D(
            256, (3, 3), activation="relu", padding="same", name="block3_conv2"
        )
    )
    model.add(
        kr.layers.Conv2D(
            256, (3, 3), activation="relu", padding="same", name="block3_conv3"
        )
    )
    model.add(
        kr.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")
    )

    # Block 4
    model.add(
        kr.layers.Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block4_conv1"
        )
    )
    model.add(
        kr.layers.Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block4_conv2"
        )
    )
    model.add(
        kr.layers.Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block4_conv3"
        )
    )
    model.add(
        kr.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")
    )

    # Block 5
    model.add(
        kr.layers.Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block5_conv1"
        )
    )
    model.add(
        kr.layers.Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block5_conv2"
        )
    )
    model.add(
        kr.layers.Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block5_conv3"
        )
    )
    model.add(
        kr.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")
    )

    if include_top:
        model.add(kr.layers.Flatten(name="flatten"))
        model.add(kr.layers.Dense(8192, activation="relu", name="fc1"))
        model.add(kr.layers.Dense(128, activation="relu", name="fc2"))
        model.add(
            kr.layers.Dense(10, activation="softmax", name="predictions")
        )
    model.summary()

    return model


def _build_path_selector_vgg16_body(model):
    """Add blocks 1-5 (with reduced block5 channels) to an existing Sequential."""
    # Block 1
    model.add(kr.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1"))
    model.add(kr.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2"))
    model.add(kr.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool"))
    # Block 2
    model.add(kr.layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1"))
    model.add(kr.layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2"))
    model.add(kr.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool"))
    # Block 3
    model.add(kr.layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1"))
    model.add(kr.layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2"))
    model.add(kr.layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3"))
    model.add(kr.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool"))
    # Block 4
    model.add(kr.layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1"))
    model.add(kr.layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2"))
    model.add(kr.layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3"))
    model.add(kr.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool"))
    # Block 5 (reduced channels to keep n_paths manageable)
    model.add(kr.layers.Conv2D(12, (3, 3), activation="relu", padding="same", name="block5_conv1"))
    model.add(kr.layers.Conv2D(12, (3, 3), activation="relu", padding="same", name="block5_conv2"))
    model.add(kr.layers.Conv2D(8, (3, 3), activation="relu", padding="same", name="block5_conv3"))
    model.add(kr.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool"))
    return model


def build_path_selector_vgg16_linear(include_top=True):
    """Path selector VGG16 with linear (no activation) output.

    Required by QsimovLinearSystem, which rejects softmax last layers.
    Architecture is identical to build_path_selector_vgg16() except
    the final Dense uses activation='linear'.
    """
    model = kr.models.Sequential()
    model.add(kr.layers.Input(shape=(32, 32, 3)))
    _build_path_selector_vgg16_body(model)
    if include_top:
        model.add(kr.layers.Flatten(name="flatten"))
        model.add(kr.layers.Dense(10, activation="linear", name="predictions"))
    model.summary()
    return model


def build_path_selector_vgg16(include_top=True):
    model = kr.models.Sequential()
    model.add(kr.layers.Input(shape=(32, 32, 3)))

    # Block 1
    model.add(
        kr.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="block1_conv1"
        )
    )
    model.add(
        kr.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="block1_conv2"
        )
    )
    model.add(
        kr.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")
    )

    # Block 2
    model.add(
        kr.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same", name="block2_conv1"
        )
    )
    model.add(
        kr.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same", name="block2_conv2"
        )
    )
    model.add(
        kr.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")
    )

    # Block 3
    model.add(
        kr.layers.Conv2D(
            256, (3, 3), activation="relu", padding="same", name="block3_conv1"
        )
    )
    model.add(
        kr.layers.Conv2D(
            256, (3, 3), activation="relu", padding="same", name="block3_conv2"
        )
    )
    model.add(
        kr.layers.Conv2D(
            256, (3, 3), activation="relu", padding="same", name="block3_conv3"
        )
    )
    model.add(
        kr.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")
    )

    # Block 4
    model.add(
        kr.layers.Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block4_conv1"
        )
    )
    model.add(
        kr.layers.Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block4_conv2"
        )
    )
    model.add(
        kr.layers.Conv2D(
            512, (3, 3), activation="relu", padding="same", name="block4_conv3"
        )
    )
    model.add(
        kr.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")
    )

    # Block 5
    model.add(
        kr.layers.Conv2D(
            12, (3, 3), activation="relu", padding="same", name="block5_conv1"
        )
    )
    model.add(
        kr.layers.Conv2D(
            12, (3, 3), activation="relu", padding="same", name="block5_conv2"
        )
    )
    model.add(
        kr.layers.Conv2D(
            8, (3, 3), activation="relu", padding="same", name="block5_conv3"
        )
    )
    model.add(
        kr.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")
    )

    if include_top:
        model.add(kr.layers.Flatten(name="flatten"))
        model.add(
            kr.layers.Dense(10, activation="softmax", name="predictions")
        )

    model.summary()
    return model


def build_lenet():
    model = kr.Sequential()

    model.add(
        kr.layers.Conv2D(
            filters=20,
            kernel_size=5,
            padding="same",
            activation="relu",
            input_shape=(32, 32, 3),
        )
    )
    model.add(kr.layers.MaxPooling2D())
    model.add(
        kr.layers.Conv2D(
            filters=50, kernel_size=5, padding="same", activation="relu"
        )
    )
    model.add(kr.layers.MaxPooling2D())
    model.add(kr.layers.Flatten())
    model.add(kr.layers.Dense(500, activation="relu"))
    model.add(kr.layers.Dense(32, activation="relu"))
    model.add(kr.layers.Dense(10, activation="softmax"))
    model.summary()

    return model


def build_alexnet():
    model = kr.Sequential()
    model.add(
        kr.layers.Conv2D(
            filters=96,
            kernel_size=(1, 1),
            strides=(1, 1),
            input_shape=(32, 32, 3),
            activation="relu",
        )
    )
    model.add(kr.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(kr.layers.Conv2D(256, (5, 5), padding="same", activation="relu"))
    model.add(kr.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(kr.layers.Conv2D(384, (3, 3), padding="same", activation="relu"))
    model.add(kr.layers.Conv2D(384, (3, 3), padding="same", activation="relu"))
    model.add(kr.layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(kr.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(kr.layers.Flatten())
    model.add(kr.layers.Dense(512, activation="relu"))
    model.add(kr.layers.Dropout(0.4))
    model.add(kr.layers.Dense(120, activation="relu"))
    model.add(kr.layers.Dropout(0.4))
    model.add(kr.layers.Dense(10, activation="softmax"))
    model.summary()

    return model


def get_optimizer(model):
    # Define optimizer for each keras model
    if model == "vgg16" or "alexnet":
        optimizer = kr.optimizers.Adam(learning_rate=1e-4)
    elif model == "lenet":
        optimizer = kr.optimizers.SGD(
            learning_rate=1e-2, momentum=0.9, nesterov=True
        )
    else:
        raise ValueError("Optimizer not supported")

    return optimizer


def create_model(name, path_selector):
    # build keras model
    if name == "alexnet":
        model = build_alexnet()
    elif name == "lenet":
        model = build_lenet()
    elif name == "vgg16":
        if path_selector:
            model = build_path_selector_vgg16()
        else:
            model = build_vgg16()
    else:
        raise ValueError("Unknown model name: {}".format(name))

    # compile model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=get_optimizer(model=model),
        metrics=["accuracy"],
    )
    return model


def load_model(model_name, model_type):
    # load model
    models_dir = get_cifar10_gradient_by_splits_results_initial_weights_dir()
    model = kr.models.load_model(
        f"{models_dir}/{model_name}{model_type}_model.tf"
    )
    return model


def save_weights_pytorch_format(
    keras_model, model_name, model_type, save_path
):
    weights = {"conv": [], "dense": []}

    for layer in keras_model.layers:
        if type(layer) == kr.layers.Conv2D:
            kernel_weights, bias_weights = layer.get_weights()
            # convert to channels first format
            kernel_weights = np.transpose(kernel_weights, (3, 2, 0, 1))
            weights["conv"].append((kernel_weights, bias_weights))
            last_conv_layer_shape = layer.output_shape[1:]

        elif type(layer) == kr.layers.Dense:
            kernel_weights, bias_weights = layer.get_weights()
            # first dense layer must shift input neuron weights to match last
            # conv / maxpool layer change of channels last to channels first
            if len(weights["dense"]) == 0:
                # indices of input neurons shaped as last conv layer output
                kernel_weights_input_idxs = np.arange(
                    kernel_weights.shape[0]
                ).reshape(last_conv_layer_shape)

                # move indices from channels last to channels first
                kernel_weights_input_idxs = np.transpose(
                    kernel_weights_input_idxs, (2, 0, 1)
                )

                # use these indices to shift the input neuron weights
                kernel_weights = kernel_weights[
                    np.ravel(kernel_weights_input_idxs)
                ]
            # weights in pytorch dense are (output, input)
            weights["dense"].append((kernel_weights.T, bias_weights))

        elif "Flatten" in str(type(layer)):
            last_conv_layer_shape = layer.input_shape[1:]

    # Save the weights and biases in the specified path
    with open(f"{save_path}/{model_name}{model_type}_weights.pkl", "wb") as f:
        pickle.dump(weights, f, protocol=5)


def main(args):
    for model_type in ("", "_path_selector"):
        model = create_model(args.model_name, model_type)

        weights_dir = (
            get_cifar10_gradient_by_splits_results_initial_weights_dir()
        )

        os.makedirs(weights_dir, exist_ok=True)

        # save model
        model.save(
            f"{weights_dir}/{args.model_name}{model_type}_model.tf",
            save_format="tf",
        )

        # save weigths
        save_weights_pytorch_format(
            model, args.model_name, model_type, weights_dir
        )


###############################################################################
# CLI Parser
###############################################################################


class ModelFactoryParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_arguments()

    def add_arguments(self):
        TrainModelsParser.add_model_name_argument(self)


if __name__ == "__main__":
    args = ModelFactoryParser().parse_args()
    main(args)
