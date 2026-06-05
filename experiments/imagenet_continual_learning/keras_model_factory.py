"""Keras model factory for the continual learning experiment.

Two path-selector architectures are provided:

  path_selector_vgg16_softmax  — VGG16 frozen base + small dense top with
      softmax output.  Used by QsimovGradient and standard fine-tuning.

  path_selector_vgg16_linear   — Same architecture but with LINEAR (no
      activation) on the last Dense layer.  Required by QsimovLinearSystem
      because check_last_layer_linear() raises ValueError on softmax.

Both architectures share the same convolutional base (imagenet weights) and
the same Dense dimensions (2048 → 128 → NUM_LABELS) so results are comparable.
"""

from experiments.imagenet_subset_by_splits.keras_model_factory import (
    preprocess_input,
    get_optimizer,
    ORIGINAL_SHAPE,
)
from experiments.imagenet_continual_learning.preprocess_data import NUM_LABELS
from tensorflow import keras as kr
from keras.applications.vgg16 import VGG16


def _build_vgg16_top(last_activation):
    """Build VGG16 + small dense top with a configurable last activation."""
    vgg = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=ORIGINAL_SHAPE,
    )

    model = kr.Sequential(
        [kr.layers.Lambda(preprocess_input, input_shape=ORIGINAL_SHAPE)]
        + vgg.layers[1:]
    )

    model.add(kr.layers.Flatten(name="flatten"))
    model.add(kr.layers.Dense(2048, activation="relu", name="fc1"))
    model.add(kr.layers.Dense(128, activation="relu", name="fc2"))
    model.add(
        kr.layers.Dense(
            NUM_LABELS,
            activation=last_activation,
            name="predictions",
        )
    )

    # freeze conv base, train only the top
    for layer in model.layers[:-3]:
        layer.trainable = False

    model.summary()
    return model


def path_selector_vgg16_softmax():
    """VGG16 + dense top with softmax.  Used for QsimovGradient / fine-tuning."""
    return _build_vgg16_top("softmax")


def path_selector_vgg16_linear():
    """VGG16 + dense top with linear output.  Required for QsimovLinearSystem."""
    return _build_vgg16_top("linear")


def load_model(results_dir, tag):
    """Load a saved model.  tag ∈ {'path_selector_softmax', 'path_selector_linear', 'standard'}."""
    return kr.models.load_model(f"{results_dir}/vgg16_{tag}_model.tf")
