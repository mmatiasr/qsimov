import pytest
from qsimov.pytorch_path_selector import PytorchPathSelector
import numpy as np
import torch
import torch.nn as nn


###############################################################################
# Test handling of nested sequential models
###############################################################################


# make a nested sequential model
def make_nested_sequential_model_1():
    # make a nested sequential model
    # initialize bias
    model = nn.Sequential(
        nn.Linear(6, 5, bias=True),
        nn.Sequential(
            nn.Linear(5, 4, bias=True),
            nn.Linear(4, 3, bias=True),
        ),
        nn.Linear(3, 2, bias=True),
    )

    return model


def test__flatten_model_from_layer_0():
    model = make_nested_sequential_model_1()

    # create a path selector
    path_selector = PytorchPathSelector(model, (6), 0, 0)

    # check number of outputs and initial layer match
    assert path_selector._number_outputs == 2
    assert path_selector._initial_layer == 0

    # ensure that the left model is None
    assert path_selector.left_model_ is None

    # ensure that the right model has 4 layers
    assert len(path_selector._layers) == 4

    # ensure that all layers in the right model are dense
    assert all(
        [isinstance(layer, nn.Linear) for layer in path_selector._layers]
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
    path_selector = PytorchPathSelector(model, (6), -2, 0)

    # check number of outputs and initial layer match
    assert path_selector._number_outputs == 2
    assert path_selector._initial_layer == 2

    # ensure that the left model is not None
    assert path_selector.left_model_ is not None

    # ensure that the left model has 2 layers
    assert len(list(path_selector.left_model_.children())) == 2

    # ensure that the right model has 2 layers
    assert len(path_selector._layers) == 2

    # ensure that all layers in the right model are dense
    assert all(
        [isinstance(layer, nn.Linear) for layer in path_selector._layers]
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


def test__flatten_model_reject_non_sequential():
    # not sequential model and compile
    model = nn.Linear(6, 2)
    model = torch.compile(model)

    with pytest.raises(ValueError):
        PytorchPathSelector(model, (6), 0, 0)

    # sequential model and compile
    model = nn.Sequential(nn.Linear(6, 2))
    model = torch.compile(model)

    with pytest.raises(ValueError):
        PytorchPathSelector(model, (6), 0, 0)


def test__check_input_shape():
    # check if the input shape introduced is correct
    # the input is empty (gives an error)
    model = nn.Sequential(nn.Linear(6, 2))
    with pytest.raises(ValueError):
        PytorchPathSelector(model, input_shape=(), initial_layer=0, verbose=0)

    # the input is integer (doesn't give an error)
    model = nn.Sequential(nn.Linear(6, 2))
    path_selector = PytorchPathSelector(
        model, input_shape=6, initial_layer=0, verbose=0
    )
    assert path_selector._input_shape == (6,)

    # the input is a list (doesn't give an error)
    model = nn.Sequential(nn.Linear(6, 2))
    path_selector = PytorchPathSelector(
        model, input_shape=[6], initial_layer=0, verbose=0
    )
    assert path_selector._input_shape == (6,)

    # the input is a tuple (doesn't give an error)
    model = nn.Sequential(nn.Linear(6, 2))
    path_selector = PytorchPathSelector(
        model, input_shape=(6,), initial_layer=0, verbose=0
    )
    assert path_selector._input_shape == (6,)


def test__compute_input_shape():
    # check if the input shape is not compatible with the model
    # the input is incorrect (gives an error)
    model = nn.Sequential(nn.Linear(6, 2))
    with pytest.raises(ValueError):
        PytorchPathSelector(
            model, input_shape=(6, 1), initial_layer=0, verbose=0
        )
    # check if the dtype is compatible with the model
    # the dtype is complex (gives an error)
    model = nn.Sequential(nn.Linear(6, 2))
    weights = torch.randn(2, 6, dtype=torch.complex128, requires_grad=False)
    biases = torch.randn(6, dtype=torch.complex128, requires_grad=False)
    new_weights = torch.nn.Parameter(
        weights.clone().detach(), requires_grad=True
    )
    new_biases = torch.nn.Parameter(
        biases.clone().detach(), requires_grad=True
    )
    model[0].weight = new_weights
    model[0].bias = new_biases
    with pytest.raises(RuntimeError):
        PytorchPathSelector(
            model, input_shape=(6,), initial_layer=0, verbose=0
        )


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
    def make_dense_layer(out_features, activation, in_features):
        return nn.Sequential(
            *[nn.Linear(in_features, out_features), activation]
        )

    # where to split the model
    initial_layer_index = 2

    # make a nested sequential model
    # initialize bias
    # add an unsupported layer to the left model (doesn't give an error)
    # add a training layer to the left model
    # add an unsupported layer to the right model (gives an error)
    if left_model_unsupported_layer is not None:
        initial_layer_index += 1

    model = nn.Sequential(
        *[
            make_dense_layer(5, left_model_activation, in_features=6),
            left_model_unsupported_layer()
            if left_model_unsupported_layer is not None
            else None,
            training_layer(),
            right_model_unsupported_layer()
            if right_model_unsupported_layer is not None
            else None,
            nn.Sequential(
                *[
                    make_dense_layer(4, right_model_activation, in_features=5),
                    # add the training layer
                    training_layer(),
                    make_dense_layer(3, last_activation, in_features=4),
                ]
            ),
        ]
    )

    return PytorchPathSelector(model, (6), initial_layer_index, 0)


def train_only_layers():
    return [
        lambda: nn.Dropout(0.5),
        lambda: nn.AlphaDropout(0.5),
        lambda: nn.Dropout1d(0.5),
        lambda: nn.AlphaDropout(0.5),
        lambda: nn.FeatureAlphaDropout(0.5),
    ]


def test__make_left_right_models_remove_training_layers():
    activation = nn.Identity()

    for training_layer in train_only_layers():
        path_selector = make_complex_path_selector(
            training_layer, activation, activation, activation
        )

        # ensure that the left model is not None
        assert path_selector.left_model_ is not None

        # ensure that the left model has 2 layers (not removed training layer)
        assert len(list(path_selector.left_model_.children())) == 2

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
    left_activation = nn.Tanh()
    right_activation = nn.ReLU()
    last_activation = nn.Sigmoid()

    path_selector = make_complex_path_selector(
        lambda: nn.Dropout(0.5),
        left_activation,
        right_activation,
        last_activation,
    )

    # ensure that the left model is not None
    assert path_selector.left_model_ is not None

    # ensure that the left model has 2 layers, not parsed activation layer
    assert len(list(path_selector.left_model_.children())) == 2

    # ensure that the right model has 4 layers, with parsed activation layers
    assert len(path_selector._layers) == 4


def make_unsupported_layers():
    # define a custom model with a non sequential structure
    class NonSequentialModel(nn.Module):
        def __init__(self):
            super(NonSequentialModel, self).__init__()
            self.flatten_layer1 = nn.Flatten()
            self.dense_layer1 = nn.Linear(in_features=5, out_features=128)
            self.relu1 = nn.ReLU()
            self.dense_layer2 = nn.Linear(in_features=5, out_features=32)
            self.relu2 = nn.ReLU()
            self.dense_layer3 = nn.Linear(in_features=160, out_features=16)
            self.relu3 = nn.ReLU()
            self.dense_layer4 = nn.Linear(in_features=16, out_features=5)
            self.sigmoid = nn.Sigmoid()

        def forward(self, inputs):
            x = self.flatten_layer1(inputs)
            x1 = self.dense_layer1(x)
            x1 = self.relu1(x1)
            x2 = self.dense_layer2(x)
            x2 = self.relu2(x2)
            x = torch.cat((x1, x2), dim=1)
            x = self.dense_layer3(x)
            x = self.relu3(x)
            x = self.dense_layer4(x)
            x = self.sigmoid(x)
            return x

    return [
        lambda: NonSequentialModel(),
        lambda: nn.Tanh(),
        lambda: nn.BatchNorm1d(5),
    ]


def test__make_left_right_models_detecting_unsupported_layers():
    for non_supported_layer in make_unsupported_layers():
        # should raise an error if the right model has an unsupported layer
        with pytest.raises(ValueError):
            make_complex_path_selector(
                lambda: nn.Dropout(0.5),
                nn.Identity(),
                nn.Identity(),
                nn.Identity(),
                right_model_unsupported_layer=non_supported_layer,
                left_model_unsupported_layer=non_supported_layer,
            )
        # will not raise an error if the left model has an unsupported layer
        make_complex_path_selector(
            lambda: nn.Dropout(0.5),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            left_model_unsupported_layer=non_supported_layer,
        )


def make_path_selector_activations():
    # layers that are supported by the path selector
    # lambda are used to avoid reusing the same layer
    return [
        nn.Identity(),
        nn.ReLU(),
    ]


def make_path_selector_unsupported_activations():
    # layers that are not supported by the path selector
    # lambda are used to avoid reusing the same layer
    return [
        nn.Sigmoid(),
        nn.Tanh(),
        nn.Threshold(0.1, 5),
    ]


def test__check_activation_layers_after_path_layer_unsupported_activations():
    for activation in make_path_selector_activations():
        # should not raise an error if always using supported activations
        # in the path selector
        make_complex_path_selector(
            lambda: nn.Dropout(0.5), activation, activation, activation
        )

        # should not raise an error even if the final activation or an
        # activation in the left model is not supported in the path
        # selector
        make_complex_path_selector(
            lambda: nn.Dropout(0.5), nn.Tanh(), activation, nn.Softmax(dim=1)
        )

    for activation in make_path_selector_unsupported_activations():
        # should raise an error even if the final activation or an
        # activation in the left model is supported by the path selector
        with pytest.raises(ValueError):
            make_complex_path_selector(
                lambda: nn.Dropout(0.5),
                nn.Identity(),
                activation,
                nn.Identity(),
            )


def test__check_activation_layers_after_path_layer():
    # should raise an error if there is an activation before the first path
    # layer
    model = nn.Sequential(
        *[
            nn.Linear(in_features=10, out_features=128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.Sigmoid(),
        ]
    )

    with pytest.raises(ValueError):
        # create a path selector starting from the second layer
        PytorchPathSelector(model, 10, 1)

    with pytest.raises(ValueError):
        # create a path selector starting from the third layer
        PytorchPathSelector(model, 10, 2)
