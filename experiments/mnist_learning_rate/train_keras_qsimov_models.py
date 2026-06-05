import pickle
import os
from experiments.path_utils import (
    get_qsimov_dataset_dir,
    get_mnist_learning_rate_results_dir,
)
import numpy as np
from experiments.mnist_speed_loss.preprocess_data import (
    get_train_test_data,
    concat_batches,
)


# Possibly slow imports
def make_imports():
    global tf, keras, AccumulatedEpochTimeTracker, concat_batches, accuracy
    global init_tensorflow, KerasQsimovGradient, KerasPathSelector
    global clone_model_with_weights, build_mnist

    from qsimov.keras_qsimov_gradient import KerasQsimovGradient
    from qsimov.keras_path_selector import KerasPathSelector
    import tensorflow as tf
    from tensorflow import keras
    from experiments.mnist_speed_loss.tf_keras.utils import (
        init_tensorflow,
        AccumulatedEpochTimeTracker,
        accuracy,
        clone_model_with_weights,
    )

    from experiments.mnist_speed_loss.tf_keras.model_factory import build_mnist


BATCH_SIZE = 256
BASE_MODEL = None


def load_base_model(input_shape, loss):
    global BASE_MODEL
    if BASE_MODEL is None:
        BASE_MODEL = build_mnist(input_shape, loss)
    return clone_model_with_weights(BASE_MODEL)


def save_results(
    output_dir,
    keras_stats,
    qsimov_stats,
    number_paths,
    learning_rates,
    multiples,
    base_learning_rate,
):
    stats = [(qsimov_stats, "qsimov"), (keras_stats, "keras")]
    for _, (history, method) in enumerate(stats):
        with open(f"{output_dir}/learning_rate_{method}_stats.pkl", "wb") as f:
            pickle.dump(history, f)
    np.save(os.path.join(output_dir, "learning_rates.npy"), learning_rates)
    np.save(
        os.path.join(output_dir, "learning_rate_multiples.npy"),
        multiples,
    )
    np.save(
        os.path.join(output_dir, "base_learning_rate.npy"),
        base_learning_rate,
    )

    with open(os.path.join(output_dir, "number_of_paths.txt"), "w") as f:
        f.write(str(number_paths))


def train_qsimov_model(
    output_dir,
    initial_layer,
    loss,
    train_x,
    train_y,
    test_x,
    test_y,
    epochs,
    learning_rate,
):
    # Load a model trained with galf of the dataset
    partial_model = keras.models.load_model(
        f"{output_dir}/crossentropy_half_model.tf"
    )
    path_selector = KerasPathSelector(
        partial_model, initial_layer=initial_layer, verbose=1
    )
    qsimov_gradient = KerasQsimovGradient(
        path_selector,
    )
    qsimov_gradient.compile(
        metrics=["accuracy"],
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    )

    # train the qsimov model
    stats = qsimov_gradient.fit(
        train_x,
        train_y,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        validation_data=(test_x, test_y),
        verbose=1,
    ).history
    # compute number of paths for naming of output files
    number_paths = np.sum(
        qsimov_gradient._path_selector.output_masks_, axis=1
    ).max()
    return stats, number_paths


def train_keras_model(
    input_shape,
    loss,
    train_x,
    train_y,
    test_x,
    test_y,
    epochs,
    learning_rate,
):
    # load the starting model, but with same initial random weights
    model = load_base_model(input_shape=input_shape, loss=loss)
    model.compile(
        metrics=["accuracy"],
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    )
    return model.fit(
        train_x,
        train_y,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        validation_data=(test_x, test_y),
        verbose=1,
    ).history


def train_models(
    initial_layer,
    train_x,
    train_y,
    test_x,
    test_y,
    loss,
    epochs,
    output_dir,
):
    # define differents learning rates
    base_learning_rate = 0.001
    multiples = np.power(4.0, np.arange(-2, 4))
    learning_rates = base_learning_rate * multiples

    # initialize the stats
    keras_stats = []
    qsimov_stats = []

    # get image shape
    input_shape = test_x.shape[1:]

    # separator for printing
    separator = "\n" + "-" * 50 + "\n"
    for idx, learning_rate in enumerate(learning_rates):
        print(
            f"Learning rate {1+idx}/{len(learning_rates)}: {learning_rate}"
            + separator
        )
        # train the keras model
        keras_stats.append(
            train_keras_model(
                input_shape,
                loss,
                train_x,
                train_y,
                test_x,
                test_y,
                epochs,
                learning_rate,
            )
        )

        # train the qsimov model

        stats, number_paths = train_qsimov_model(
            output_dir,
            initial_layer,
            loss,
            train_x,
            train_y,
            test_x,
            test_y,
            epochs,
            learning_rate,
        )
        qsimov_stats.append(stats)

    save_results(
        output_dir,
        keras_stats,
        qsimov_stats,
        number_paths,
        learning_rates,
        multiples,
        base_learning_rate,
    )


def execute_logic(data_dir, output_dir, epochs, initial_layer):
    # load dataset
    train_xs, train_ys, test_x, test_y = get_train_test_data(data_dir)

    train_models(
        initial_layer=initial_layer,
        train_x=concat_batches(train_xs),
        train_y=concat_batches(train_ys),
        test_x=test_x,
        test_y=test_y,
        loss="categorical_crossentropy",
        epochs=epochs,
        output_dir=output_dir,
    )


def main(args):
    # make slow imports (those that need to import tensorflow)
    make_imports()

    # arguments
    processor = args.processor
    epochs = args.epochs
    initial_layer = args.initial_layer

    # init tensorflow
    init_tensorflow(tf, processor)

    # input and output directories
    data_dir = get_qsimov_dataset_dir("mnist")
    output_dir = get_mnist_learning_rate_results_dir("keras", processor)
    os.makedirs(output_dir, exist_ok=True)

    # train the different combinations of learning rates
    execute_logic(data_dir, output_dir, epochs, initial_layer)


if __name__ == "__main__":
    from experiments.mnist_speed_loss.tf_keras.train_qsimov_models import (
        TrainQsimovModelsParser,
    )

    main(TrainQsimovModelsParser().parse_args())
