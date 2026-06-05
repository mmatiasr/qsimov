import pytest
from qsimov.keras_path_selector import KerasPathSelector
import numpy as np
from tensorflow import keras as kr

krl = kr.layers


###############################################################################
# Test handling of nested sequential models
###############################################################################


# make a nested sequential model
def make_nested_sequential_model_1():
    # make a nested sequential model
    # initialize bias with ones so that connections arent removed
    model = kr.Sequential()
    model.add(krl.Dense(5, input_shape=(6,), bias_initializer="ones"))
    model.add(
        kr.Sequential(
            [
                krl.Dense(4, bias_initializer="ones"),
                krl.Dense(3, bias_initializer="ones"),
            ]
        )
    )
    model.add(krl.Dense(2, bias_initializer="ones"))

    # compile
    model.compile(optimizer="adam", loss="mse")

    return model


def test__flatten_model_from_layer_0():
    model = make_nested_sequential_model_1()

    # create a path selector
    path_selector = KerasPathSelector(model, 0, 0)

    # check number of outputs and initial layer match
    assert path_selector._number_outputs == 2
    assert path_selector._initial_layer == 0

    # ensure that the left model is None
    assert path_selector.left_model_ is None

    # ensure that the right model has 4 layers
    assert len(path_selector._layers) == 4

    # ensure that all layers in the right model are dense
    assert all(
        [isinstance(layer, krl.Dense) for layer in path_selector._layers]
    )

    # ensure the layer types were correctly extracted
    assert all(
        [
            layer_type == path_selector.LayerTypes.DENSE
            for layer_type in path_selector._layer_types
        ]
    )
    assert len(path_selector._layer_types) == 4

    # ensure the layer connections were correctly extracted
    connection_shapes_true = [
        (5 * (6 + 1), 2),  # layer 0
        (4 * (5 + 1), 2),  # layer 1
        (3 * (4 + 1), 2),  # layer 2
        (1 * (3 + 1), 2),  # layer 3, done as if there is a single output
    ]
    connection_shapes_true = np.array(connection_shapes_true)

    # connection shapes
    connection_shapes = [
        connections.shape for connections in path_selector._layer_connections
    ]
    np.testing.assert_array_equal(connection_shapes, connection_shapes_true)


def test__flatten_model_from_layer_2():
    model = make_nested_sequential_model_1()

    # create a path selector using last two layers
    path_selector = KerasPathSelector(model, -2, 0)

    # check number of outputs and initial layer match
    assert path_selector._number_outputs == 2
    assert path_selector._initial_layer == 2

    # ensure that the left model is not None
    assert path_selector.left_model_ is not None

    # ensure that the left model has 2 layers
    assert len(path_selector.left_model_.layers) == 2

    # ensure that the right model has 2 layers
    assert len(path_selector._layers) == 2

    # ensure that all layers in the right model are dense
    assert all(
        [isinstance(layer, krl.Dense) for layer in path_selector._layers]
    )

    # ensure the layer types were correctly extracted
    assert all(
        [
            layer_type == path_selector.LayerTypes.DENSE
            for layer_type in path_selector._layer_types
        ]
    )
    assert len(path_selector._layer_types) == 2

    # ensure the layer connections were correctly extracted
    connection_shapes_true = [
        (3 * (4 + 1), 2),  # layer 2
        (1 * (3 + 1), 2),  # layer 3, done as if there is a single output
    ]
    connection_shapes_true = np.array(connection_shapes_true)

    # connection shapes
    connection_shapes = [
        connections.shape for connections in path_selector._layer_connections
    ]
    np.testing.assert_array_equal(connection_shapes, connection_shapes_true)


def test__flatten_model_reject_non_sequential_non_compiled():
    # not sequential model
    input_layer_1 = kr.Input(shape=(10,))
    hidden_layer_1 = krl.Dense(32, activation="relu")(input_layer_1)

    input_layer_2 = kr.Input(shape=(20,))
    hidden_layer_2 = krl.Dense(32, activation="relu")(input_layer_2)

    concatenated_output = krl.concatenate([hidden_layer_1, hidden_layer_2])
    output_layer = krl.Dense(1, activation="sigmoid")(concatenated_output)

    model = kr.Model(
        inputs=[input_layer_1, input_layer_2], outputs=output_layer
    )
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    with pytest.raises(ValueError):
        KerasPathSelector(model, 0, 0)

    # sequential model but not built
    model = kr.Sequential([krl.Dense(2)])

    with pytest.raises(ValueError):
        KerasPathSelector(model, 0, 0)


###############################################################################
# Test the make_left_right_models method
###############################################################################


def make_complex_path_selector(
    training_layer,
    left_model_activation,
    right_model_activation,
    last_activation,
    left_model_unsupported_layer=None,
    right_model_unsupported_layer=None,
):
    # make a dense layer with activation
    def make_dense_layer(units, activation, input_shape=None):
        kwargs = {"units": units, "bias_initializer": "ones"}
        if input_shape is not None:
            kwargs["input_shape"] = input_shape

        if type(activation) == str:
            return krl.Dense(activation=activation, **kwargs)
        else:
            return kr.Sequential([krl.Dense(**kwargs), activation()])

    # where to split the model
    initial_layer_index = 2

    # make a nested sequential model
    # initialize bias with ones so that connections arent removed

    model = kr.Sequential()
    model.add(make_dense_layer(5, left_model_activation, input_shape=(6,)))

    # add an unsupported layer to the left model (doesn't give an error)
    if left_model_unsupported_layer is not None:
        model.add(left_model_unsupported_layer())
        initial_layer_index += 1

    # add a training layer to the left model
    model.add(training_layer())

    # add an unsupported layer to the right model (gives an error)
    if right_model_unsupported_layer is not None:
        model.add(right_model_unsupported_layer())

    model.add(
        kr.Sequential(
            [
                make_dense_layer(4, right_model_activation),
                # add the training layer
                training_layer(),
                make_dense_layer(3, last_activation),
            ]
        )
    )

    # compile
    model.compile(optimizer="adam", loss="mse")

    return KerasPathSelector(model, initial_layer_index, 0)


def train_only_layers():
    return [
        lambda: krl.Dropout(0.5),
        lambda: krl.AlphaDropout(0.5),
        lambda: krl.GaussianNoise(0.5),
        lambda: krl.GaussianDropout(0.5),
        lambda: krl.ActivityRegularization(0.5),
    ]


def test__make_left_right_models_remove_training_layers():
    activation = "linear"

    for training_layer in train_only_layers():
        path_selector = make_complex_path_selector(
            training_layer, activation, activation, activation
        )

        # ensure that the left model is not None
        assert path_selector.left_model_ is not None

        # ensure that the left model has 2 layers (not removed training layer)
        assert len(path_selector.left_model_.layers) == 2

        # ensure that the right model has 2 layers (removed training layer)
        assert len(path_selector._layers) == 2

        # ensure all layer types are dense, as we used linear activation
        assert all(
            [
                layer_type == path_selector.LayerTypes.DENSE
                for layer_type in path_selector._layer_types
            ]
        )


def test__make_left_right_models_parsing_activation_layers():
    left_activation = "tanh"
    right_activation = "relu"
    last_activation = "sigmoid"

    path_selector = make_complex_path_selector(
        lambda: krl.Dropout(0.5),
        left_activation,
        right_activation,
        last_activation,
    )

    # ensure that the left model is not None
    assert path_selector.left_model_ is not None

    # ensure that the left model has 2 layers, not parsed activation layer
    assert len(path_selector.left_model_.layers) == 2

    # ensure that the right model has 4 layers, with parsed activation layers
    assert len(path_selector._layers) == 4

    # ensure dense layers do not have activation
    assert all(
        [
            layer_type != path_selector.LayerTypes.DENSE
            or layer.activation.__name__ == "linear"
            for layer_type, layer in zip(
                path_selector._layer_types, path_selector._layers
            )
        ]
    )

    # ensure activation layers were placed correctly
    assert path_selector._layer_types[1] == path_selector.LayerTypes.ACTIVATION
    assert path_selector._layer_types[3] == path_selector.LayerTypes.ACTIVATION

    assert path_selector._layers[1].activation.__name__ == right_activation
    assert path_selector._layers[3].activation.__name__ == last_activation

    assert path_selector._layer_types[0] == path_selector.LayerTypes.DENSE
    assert path_selector._layer_types[2] == path_selector.LayerTypes.DENSE


def make_unsupported_layers():
    # define a custom model with a non sequential structure
    class NonSequentialModel(kr.Model):
        def __init__(self):
            super(NonSequentialModel, self).__init__()
            self.flatten_layer1 = krl.Flatten()
            self.dense_layer1 = krl.Dense(units=128, activation="relu")
            self.dense_layer2 = krl.Dense(units=32, activation="relu")
            self.dense_layer3 = krl.Dense(units=16, activation="relu")
            self.dense_layer4 = krl.Dense(units=1, activation="sigmoid")

        def call(self, inputs):
            x = self.flatten_layer1(inputs)
            x1 = self.dense_layer1(x)
            x2 = self.dense_layer2(x)

            x = krl.concatenate([x1, x2])
            x = self.dense_layer3(x)
            x = self.dense_layer4(x)
            return x

    return [
        lambda: NonSequentialModel(),
        lambda: krl.Lambda(lambda x: x),
        lambda: krl.Normalization(),
    ]


def test__make_left_right_models_detecting_unsupported_layers():
    for non_supported_layer in make_unsupported_layers():
        # should raise an error if the right model has an unsupported layer
        with pytest.raises(ValueError):
            make_complex_path_selector(
                lambda: krl.Dropout(0.5),
                "linear",
                "linear",
                "linear",
                right_model_unsupported_layer=non_supported_layer,
                left_model_unsupported_layer=non_supported_layer,
            )
        # will not raise an error if the left model has an unsupported layer
        make_complex_path_selector(
            lambda: krl.Dropout(0.5),
            "linear",
            "linear",
            "linear",
            left_model_unsupported_layer=non_supported_layer,
        )


def make_path_selector_activations():
    # layers that are supported by the path selector in different formats
    # lambda are used to avoid reusing the same layer
    return [
        ["linear", lambda: krl.Activation("linear")],
        ["relu", lambda: krl.Activation("relu"), lambda: krl.ReLU()],
    ]


def make_path_selector_unsupported_activations():
    # layers that are not supported by the path selector in different formats
    # lambda are used to avoid reusing the same layer
    return [
        ["sigmoid", lambda: krl.Activation("sigmoid")],
        ["tanh", lambda: krl.Activation("tanh")],
        [lambda: krl.ReLU(negative_slope=0.1)],
    ]


def test__check_activation_layers_after_path_layer_unsupported_activations():
    for activations in make_path_selector_activations():
        for activation in activations:
            # should not raise an error if always using supported activations
            # in the path selector
            make_complex_path_selector(
                lambda: krl.Dropout(0.5), activation, activation, activation
            )

            # should not raise an error even if the final activation or an
            # activation in the left model is not supported in the path
            # selector
            make_complex_path_selector(
                lambda: krl.Dropout(0.5), "tanh", activation, "softmax"
            )

    for activations in make_path_selector_unsupported_activations():
        for activation in activations:
            # should raise an error even if the final activation or an
            # activation in the left model is supported by the path selector
            with pytest.raises(ValueError):
                make_complex_path_selector(
                    lambda: krl.Dropout(0.5), "linear", activation, "linear"
                )


def test__check_activation_layers_after_path_layer():
    # should raise an error if there is an activation before the first path
    # layer
    model = kr.Sequential(
        [
            krl.Input(shape=(10,)),
            krl.Dense(units=128),
            krl.Dropout(0.5),
            krl.ReLU(),
            krl.Dense(units=32, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")

    with pytest.raises(ValueError):
        # create a path selector starting from the second layer
        KerasPathSelector(model, 1)

    with pytest.raises(ValueError):
        # create a path selector starting from the third layer
        KerasPathSelector(model, 2)
