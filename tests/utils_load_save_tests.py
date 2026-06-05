import numpy as np
import torch


def list_assert_almost_equal(x, y, **kwargs):
    if x is None:
        assert y is None
        return

    assert len(x) == len(y)

    for idx in range(len(x)):
        if type(x[idx]) is not np.ndarray:
            assert x[idx] == y[idx]
        else:
            np.testing.assert_almost_equal(x[idx], y[idx], **kwargs)


def assert_torch_models_equal(result, expected):
    if result is None:
        assert expected is None
        return
    # move to same device
    result = result.to(next(expected.parameters()).device)
    assert result.state_dict().keys() == expected.state_dict().keys()
    for key in result.state_dict().keys():
        assert torch.equal(
            result.state_dict()[key], expected.state_dict()[key]
        )


def assert_path_selector_equals(result, expected):
    # output masks
    np.testing.assert_array_almost_equal(
        result.output_masks_, expected.output_masks_
    )

    # paths
    np.testing.assert_array_almost_equal(
        result._all_paths, expected._all_paths
    )
    np.testing.assert_array_almost_equal(
        result._all_paths_input_neurons, expected._all_paths_input_neurons
    )

    # initial layer
    assert result._initial_layer == expected._initial_layer

    # layer connections
    list_assert_almost_equal(
        result._layer_connections, expected._layer_connections
    )

    # layer types
    assert result._layer_types == expected._layer_types

    # parameters
    list_assert_almost_equal(result._list_biases, expected._list_biases)
    list_assert_almost_equal(result._list_weights, expected._list_weights)

    # number of outputs
    assert result._number_outputs == expected._number_outputs

    # transformation indices
    list_assert_almost_equal(
        result._partial_to_full_idxs, expected._partial_to_full_idxs
    )

    # zero to zero counts
    list_assert_almost_equal(
        result._zero_to_zero_counts, expected._zero_to_zero_counts
    )


def assert_keras_path_selector_equals(result, expected):
    # compare attributes of two path selectors
    # neural network
    list_assert_almost_equal(
        result.right_model_.get_weights(),
        expected.right_model_.get_weights(),
    )
    if expected.left_model_ is None:
        assert result.left_model_ is None
    else:
        list_assert_almost_equal(
            result.left_model_.get_weights(),
            expected.left_model_.get_weights(),
        )

    # layers
    for layer_idx in range(len(result._layers)):
        if result._layers[layer_idx] is None:
            assert expected._layers[layer_idx] is None
        else:
            assert (
                result._layers[layer_idx].get_config()
                == expected._layers[layer_idx].get_config()
            )
    assert len(result._layers) == len(expected._layers)

    # rest of attributes
    return assert_path_selector_equals(result, expected)


def assert_pytorch_path_selector_equals(result, expected):
    # compare attributes of two path selectors
    # neural network
    assert_torch_models_equal(result.right_model_, expected.right_model_)
    assert_torch_models_equal(result.left_model_, expected.left_model_)

    # layers
    assert len(result._layers) == len(expected._layers)

    # rest of attributes
    return assert_path_selector_equals(result, expected)


def assert_qsimov_linear_system_equals(result, expected):
    # equations and solutions
    list_assert_almost_equal(result.equations_, expected.equations_)
    list_assert_almost_equal(result.solutions_, expected.solutions_)

    # keyword arguments
    assert result._kwargs == expected._kwargs

    # shrinkage factor
    assert result._qr_shrinkage_factor == expected._qr_shrinkage_factor

    # qr flags
    np.testing.assert_array_equal(
        result._r_transformed_equations,
        expected._r_transformed_equations,
    )

    # solver
    assert result._solver == expected._solver

    # shape of output
    np.testing.assert_array_equal(result._Y_shape, expected._Y_shape)


def assert_keras_qsimov_linear_system_equals(result, expected):
    # path selector
    assert_keras_path_selector_equals(
        result._path_selector, expected._path_selector
    )
    return assert_qsimov_linear_system_equals(result, expected)


def assert_pytorch_qsimov_linear_system_equals(result, expected):
    # path selector
    assert_pytorch_path_selector_equals(
        result._path_selector, expected._path_selector
    )
    return assert_qsimov_linear_system_equals(result, expected)


def assert_qsimov_gradient_equals(result, expected):
    np.testing.assert_array_equal(result._Y_shape, expected._Y_shape)


def assert_keras_qsimov_gradient_equals(result, expected):
    assert_qsimov_gradient_equals(result, expected)

    # neural network
    if result.model_ is None:
        assert expected.model_ is None
    else:
        list_assert_almost_equal(
            result.model_.get_weights(),
            expected.model_.get_weights(),
        )

    # path selector
    return assert_keras_path_selector_equals(
        result._path_selector, expected._path_selector
    )


def assert_pytorch_qsimov_gradient_equals(result, expected):
    assert_qsimov_gradient_equals(result, expected)

    # neural network
    assert_torch_models_equal(result.model_, expected.model_)

    # path selector
    return assert_pytorch_path_selector_equals(
        result._path_selector, expected._path_selector
    )
