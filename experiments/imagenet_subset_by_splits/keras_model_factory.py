from tensorflow import keras as kr
from keras.applications.vgg16 import VGG16
from experiments.imagenet_subset_by_splits.preprocess_data import NUM_LABELS

ORIGINAL_SHAPE = (224, 224, 3)


def preprocess_input(x):
    x /= 127.5
    x -= 1.0
    return x


# original vgg16 model
def original_vgg16(weights="imagenet"):
    model = kr.Sequential(
        [kr.layers.Lambda(preprocess_input, input_shape=ORIGINAL_SHAPE)]
        + VGG16(
            include_top=False,
            weights=weights,
            input_shape=ORIGINAL_SHAPE,
            classes=NUM_LABELS,
        ).layers[1:]
    )

    model.add(kr.layers.Flatten(name="flatten"))
    model.add(kr.layers.Dense(4096, activation="relu", name="fc1"))
    model.add(kr.layers.Dense(4096, activation="relu", name="fc2"))
    model.add(
        kr.layers.Dense(NUM_LABELS, activation="softmax", name="predictions")
    )

    for layer in model.layers[:-3]:
        layer.trainable = False
    model.summary()
    return model


# vgg16 model adapted to more sparse connections
def path_selector_vgg_16(weights="imagenet"):
    model = kr.Sequential(
        [kr.layers.Lambda(preprocess_input, input_shape=ORIGINAL_SHAPE)]
        + VGG16(
            include_top=False,
            weights=weights,
            input_shape=ORIGINAL_SHAPE,
            classes=NUM_LABELS,
        ).layers[1:]
    )

    model.add(kr.layers.Flatten(name="flatten"))
    model.add(kr.layers.Dense(2048, activation="relu", name="fc1"))
    model.add(kr.layers.Dense(128, activation="relu", name="fc2"))
    model.add(
        kr.layers.Dense(NUM_LABELS, activation="softmax", name="predictions")
    )

    for layer in model.layers[:-3]:
        layer.trainable = False
    model.summary()
    return model


def get_optimizer(model, is_qsimov=False):
    # Define optimizer for each keras model
    if model == "vgg16":
        if is_qsimov:
            optimizer = kr.optimizers.Adam(learning_rate=1e-5)
        else:
            optimizer = kr.optimizers.Adam(learning_rate=1e-4)
    else:
        raise ValueError("Optimizer not supported")

    return optimizer


def load_model(model, path_selector=False, **kwargs):
    # Define optimizer for each keras model
    if model == "vgg16":
        if path_selector:
            model_instance = path_selector_vgg_16(
                weights=kwargs.get("weights", "imagenet")
            )
        else:
            model_instance = original_vgg16(
                weights=kwargs.get("weights", "imagenet")
            )
    else:
        raise ValueError("Model not supported")

    model_instance.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=get_optimizer(model),
        metrics=["accuracy"],
    )

    return model_instance
