from experiments.mnist_speed_loss.tf_keras.train_keras_models import (
    TrainModelsParser,
)
import time
import pickle
import os
import numpy as np
from experiments.path_utils import (
    get_qsimov_dataset_dir,
    get_mnist_speed_loss_results_dir as get_results_dir,
)


# Possibly slow imports
def make_imports():
    global accuracy, keras, tf, get_train_test_data, concat_batches
    global init_tensorflow, KerasQsimovGradient, AccumulatedEpochTimeTracker
    global KerasQsimovLinearSystem, KerasPathSelector

    import tensorflow as tf
    from tensorflow import keras

    from experiments.mnist_speed_loss.tf_keras.utils import (
        init_tensorflow,
        accuracy,
    )
    from experiments.mnist_speed_loss.preprocess_data import (
        get_train_test_data,
        concat_batches,
    )
    from qsimov.keras_qsimov_gradient import KerasQsimovGradient
    from experiments.mnist_speed_loss.tf_keras.utils import (
        AccumulatedEpochTimeTracker,
    )
    from qsimov.keras_qsimov_linear_system import KerasQsimovLinearSystem
    from qsimov.keras_path_selector import KerasPathSelector


BATCH_SIZE = 32


# Calculate mse
def compute_mse(Y_pred, Y_true):
    return np.mean((Y_pred - Y_true) ** 2)


# Qsimov gradient descent
def train_qsimov_gradient_models(
    path_selector,
    model_name,
    device,
    train_x,
    train_y,
    test_x,
    test_y,
    loss_name,
    epochs,
    output_dir,
):
    # set callback to track time
    times_partial_model = AccumulatedEpochTimeTracker()

    # create qsimov gradient model
    qsimov_gradient = KerasQsimovGradient(path_selector)

    # compile model
    qsimov_gradient.compile(
        metrics=[accuracy], loss=loss_name, optimizer="adam", device=device
    )

    # fit model taking time
    history_qsimov_gradient = qsimov_gradient.fit(
        train_x,
        train_y,
        callbacks=[times_partial_model],
        batch_size=BATCH_SIZE,
        validation_data=(test_x, test_y),
        epochs=epochs,
    )

    # set timestamps on history
    history_qsimov_gradient.history["time(s)"] = times_partial_model.times

    # save results based on number of paths
    number_paths = np.sum(
        qsimov_gradient._path_selector.output_masks_, axis=1
    ).max()

    # save results
    output_file = f"{output_dir}/qsimov_gradient_{model_name}_history.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(history_qsimov_gradient.history, f)

    with open(f"{output_dir}/number_of_paths.txt", "w") as f:
        f.write(str(number_paths))


# Qsimov linear system
def train_qsimov_linear_model(
    path_selector,
    model_name,
    train_x,
    train_y,
    test_x,
    test_y,
    output_dir,
):
    # create qsimov linear system
    qsimov_linear = KerasQsimovLinearSystem(path_selector, verbose=1)

    # fit model taking time
    times_qsimov_linear = time.time()
    qsimov_linear.fit(train_x, train_y, batch_size=BATCH_SIZE)
    times_qsimov_linear = time.time() - times_qsimov_linear

    # predict and compute accuracy/mse on train and test
    y_train_pred = qsimov_linear.predict(train_x, batch_size=512)
    y_test_pred = qsimov_linear.predict(test_x, batch_size=BATCH_SIZE)
    train_accuracy = accuracy(y_train_pred, train_y)
    train_mse = compute_mse(y_train_pred, train_y)
    test_accuracy = accuracy(y_test_pred, test_y)
    test_mse = compute_mse(y_test_pred, test_y)

    # create history dictionary
    history_qsimov_linear = {
        "loss": [train_mse],
        "val_loss": [test_mse],
        "accuracy": [train_accuracy],
        "val_accuracy": [test_accuracy],
        "time(s)": [times_qsimov_linear],
    }

    # save history

    output_file = f"{output_dir}/qsimov_linear_{model_name}_history.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(history_qsimov_linear, f)


def train_qsimov_models(
    output_dir,
    initial_layer,
    device,
    train_x,
    train_y,
    test_x,
    test_y,
    epochs,
):
    # dataset portion names
    partial_datas = ["half", "quarter"]

    # loss names
    loss_functions = ["categorical_crossentropy", "mse"]

    # base models names for qsimov gradient/linear system
    crossentropy_models = ["crossentropy_half", "crossentropy_quarter"]
    mse_models = ["mse_half", "mse_quarter"]
    models = [crossentropy_models, mse_models]

    # separator for printing
    separator = "\n" + "-" * 50 + "\n"
    for partial_idx, name in enumerate(partial_datas):
        print(f"Training {name} models" + separator)
        for loss_idx, loss_name in enumerate(loss_functions):
            print(f"with {loss_name} loss function" + separator)

            # load model based on loss function and dataset portion
            current_models = models[loss_idx]
            model_name = current_models[partial_idx]
            model = keras.models.load_model(
                f"{output_dir}/{model_name}_model.tf"
            )

            # create path selector
            path_selector = KerasPathSelector(
                model, initial_layer=initial_layer, device=device, verbose=1
            )

            # set kwargs for training
            kwargs = {
                "path_selector": path_selector,
                "model_name": model_name,
                "train_x": train_x,
                "train_y": train_y,
                "test_x": test_x,
                "test_y": test_y,
                "output_dir": output_dir,
            }

            # train qsimov model using loaded model
            train_qsimov_gradient_models(
                **kwargs, epochs=epochs, loss_name=loss_name, device=device
            )

            # train qsimov linear system only for mse loss
            if loss_name == "mse":
                train_qsimov_linear_model(**kwargs)


def main(args):
    make_imports()
    # arguments
    processor = args.processor
    epochs = args.epochs
    initial_layer = args.initial_layer

    # init tensorflow
    init_tensorflow(tf, processor)

    # input and output directories
    data_dir = get_qsimov_dataset_dir("mnist")
    output_dir = get_results_dir("keras", processor)
    os.makedirs(output_dir, exist_ok=True)

    # load dataset
    train_xs, train_ys, test_x, test_y = get_train_test_data(data_dir)

    train_qsimov_models(
        output_dir=output_dir,
        initial_layer=initial_layer,
        device="/cpu:0" if processor == "cpu" else "/gpu:0",
        train_x=concat_batches(train_xs),
        train_y=concat_batches(train_ys),
        test_x=test_x,
        test_y=test_y,
        epochs=epochs,
    )


class TrainQsimovModelsParser(TrainModelsParser):
    def add_arguments(self):
        super().add_arguments()
        self.add_initial_layer_argument()

    def add_initial_layer_argument(self):
        self.add_argument(
            "--initial-layer",
            help="Layer of the model where path selection starts",
            type=int,
            default=-2,
        )


if __name__ == "__main__":
    main(TrainQsimovModelsParser().parse_args())
