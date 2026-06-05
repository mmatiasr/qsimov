import numpy as np
import pytest
from qsimov.keras_path_selector import KerasPathSelector
import qsimov.paths.paths as paths_
import qsimov.paths.combine as combine
from tensorflow import keras as kr

krl = kr.layers


@pytest.fixture(name="path_selector")
def make_path_selector():
    # create model
    model = kr.Sequential()
    model.add(
        krl.Conv2D(2, kernel_size=2, input_shape=(3, 3, 1), activation="relu")
    )
    model.add(krl.Conv2D(1, kernel_size=2, padding="same", use_bias=False))
    model.compile(optimizer="adam", loss="mse")

    # "train" model
    weights1 = np.array([[[[0, -2]], [[3, 1]]], [[[2, 1]], [[-1, 1]]]])
    weights2 = np.array(
        [[[[1], [-1]], [[1], [-1]]], [[[1], [-1]], [[1], [-1]]]]
    )
    biases1 = np.array([0, 1])
    model.layers[0].set_weights([weights1, biases1])
    model.layers[1].set_weights([weights2])

    # use for path selection
    return KerasPathSelector(neural_network=model, initial_layer=0)


@pytest.fixture(name="X")
def make_samples():
    sample1 = np.array([[1, 2, 0], [3, -1, 4], [0, 2, 2]]).reshape(3, 3, 1)
    return np.array([sample1]).astype(float)


def test__compute_all_paths(path_selector):
    # check full path list
    all_paths = path_selector._all_paths

    # all paths between layer 0 and 1
    all_paths_layer1_true = np.vstack(
        [
            # insert(input neurons, _, output_neuron,_)
            np.insert([[2], [4], [5]], 1, 1, axis=1),
            np.insert([[3], [5], [6]], 1, 3, axis=1),
            np.insert([[5], [7], [8]], 1, 5, axis=1),
            np.insert([[6], [8], [9]], 1, 7, axis=1),
            np.insert([[0], [1], [2], [4], [5]], 1, 2, axis=1),
            np.insert([[0], [2], [3], [5], [6]], 1, 4, axis=1),
            np.insert([[0], [4], [5], [7], [8]], 1, 6, axis=1),
            np.insert([[0], [5], [6], [8], [9]], 1, 8, axis=1),
        ]
    ).astype(np.int32)

    # all paths between layer 1 and 2
    all_paths_layer2_true = np.vstack(
        [
            np.insert([[1], [2], [3], [4], [5], [6], [7], [8]], 1, 1, axis=1),
            np.insert([[3], [4], [7], [8]], 1, 2, axis=1),
            np.insert([[5], [6], [7], [8]], 1, 3, axis=1),
            np.insert([[7], [8]], 1, 4, axis=1),
        ]
    ).astype(np.int32)

    # compute all paths and split by output neuron
    all_paths_true = combine.combine_paths(
        [all_paths_layer1_true, all_paths_layer2_true]
    )
    all_paths_true = [
        all_paths_true[all_paths_true[:, 2] == output_neuron]
        for output_neuron in range(1, 5)
    ]

    # compare paths to each output against masked paths to any output
    out_masks = path_selector.output_masks_
    for output_idx in range(4):
        # get correct paths only to this output
        paths_true = np.array(paths_.sort_paths(all_paths_true[output_idx]))
        paths_true[:, -1] = 1
        np.testing.assert_array_equal(
            all_paths[out_masks[output_idx]], paths_true
        )


def test_samples_to_coefficients(X, path_selector):
    coefficients = path_selector.samples_to_coefficients(X)

    coefficients_true = [
        [
            1,
            0,
            0,
            1,
            1,
            2,
            2,
            0,
            0,
            0,
            3,
            3,
            0,
            -1,
            -1,
            0,
            0,
            0,
            0,
            -1,
            0,
            0,
            4,
            4,
            0,
            0,
            0,
            0,
            2,
            2,
            2,
            2,
        ],
        [0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 4, 4, 2, 2, 2, 2],
        [0, 1, 0, 0, 0, -1, 4, 4, 0, 0, 0, 0, 2, 2, 2, 2],
        [1, -1, 4, 4, 2, 2, 2, 2],
    ]
    coefficients_true = [
        np.asfarray(coefficients) for coefficients in coefficients_true
    ]

    for output_idx in range(4):
        np.testing.assert_array_equal(
            coefficients[:, path_selector.output_masks_[output_idx]],
            [coefficients_true[output_idx]],
        )
    return
