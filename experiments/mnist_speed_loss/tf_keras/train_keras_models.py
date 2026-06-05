import pickle
import os
import argparse
from experiments.path_utils import (
    get_qsimov_dataset_dir,
    get_mnist_speed_loss_results_dir as get_results_dir,
)


# Possibly slow imports
def make_imports():
    global tf, keras, AccumulatedEpochTimeTracker, concat_batches, accuracy
    global get_train_test_data, make_dataset_portions, init_tensorflow
    global clone_model_with_weights, build_mnist

    import tensorflow as tf
    from tensorflow import keras
    from experiments.mnist_speed_loss.tf_keras.utils import (
        init_tensorflow,
        AccumulatedEpochTimeTracker,
        accuracy,
        clone_model_with_weights,
    )
    from experiments.mnist_speed_loss.tf_keras.model_factory import build_mnist

    from experiments.mnist_speed_loss.preprocess_data import (
        get_train_test_data,
        concat_batches,
    )


BATCH_SIZE = 32


###############################################################################
# Logic
###############################################################################


def build_models(image_shape):
    crossentropy_model = build_mnist(image_shape)
    crossentropy_models = [
        (crossentropy_model, "crossentropy_half"),
        (clone_model_with_weights(crossentropy_model), "crossentropy_quarter"),
    ]
    mse_model = build_mnist(image_shape, loss="mse")
    mse_models = [
        (mse_model, "mse_half"),
        (clone_model_with_weights(mse_model), "mse_quarter"),
    ]
    return [crossentropy_models, mse_models]


def save_results(name, model, history, times, output_dir):
    history.history["time(s)"] = times.times
    with open(f"{output_dir}/{name}_model_history.pkl", "wb") as f:
        pickle.dump(history.history, f)

    model.save(f"{output_dir}/{name}_model.tf", save_format="tf")

    with open(f"{output_dir}/model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))


def train_models(
    train_xs,
    train_ys,
    partial_x,
    partial_y,
    test_x,
    test_y,
    model,
    model_name,
    epochs,
    output_dir,
):
    # Separator for printing
    separator = "\n" + "-" * 80

    # make a copy for training the full model later
    full_model = clone_model_with_weights(model)

    # partial model
    print(f"Training {model_name}" + separator)
    times = AccumulatedEpochTimeTracker()  # callback to track time
    history = model.fit(
        partial_x,
        partial_y,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(test_x, test_y),
        callbacks=[times],
    )
    save_results(model_name, model, history, times, output_dir)

    # model training all the dataset
    print(f"Training {model_name} model with full dataset" + separator)
    times = AccumulatedEpochTimeTracker()  # callback to track time
    history = full_model.fit(
        concat_batches(train_xs),
        concat_batches(train_ys),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(test_x, test_y),
        callbacks=[times],
    )
    save_results(f"{model_name}_full", full_model, history, times, output_dir)


def execute_logic(data_dir, output_dir, epochs):
    # load dataset
    train_xs, train_ys, test_x, test_y = get_train_test_data(data_dir)

    # get image shape
    image_shape = test_x.shape[1:]

    # build models
    models = build_models(image_shape)

    # partial datasets
    partial_datas = make_dataset_portions(train_xs, train_ys)

    # loss functions to use
    loss_functions = ["categorical_crossentropy", "mse"]

    # separator for printing
    separator = "\n" + "-" * 50 + "\n"

    for partial_idx, (partial_x, partial_y, name) in enumerate(partial_datas):
        print(f"Training {name} models" + separator)
        for loss_idx, loss_function in enumerate(loss_functions):
            print(f"with {loss_function} loss function" + separator)
            current_models = models[loss_idx]
            model, model_name = current_models[partial_idx]

            # train partial_model
            train_models(
                train_xs,
                train_ys,
                partial_x,
                partial_y,
                test_x,
                test_y,
                model,
                model_name,
                epochs,
                output_dir,
            )


def make_dataset_portions(train_xs, train_ys):
    return [
        (concat_batches(train_xs[:2]), concat_batches(train_ys[:2]), "half"),
        (
            concat_batches(train_xs[:1]),
            concat_batches(train_ys[:1]),
            "quarter",
        ),
    ]


def main(args):
    # make slow imports (those that need to import tensorflow)
    make_imports()

    # arguments
    processor, epochs = args.processor, args.epochs

    # init tensorflow
    init_tensorflow(tf, processor)

    # directories to save the models
    data_dir = get_qsimov_dataset_dir("mnist")
    output_dir = get_results_dir("keras", processor)

    os.makedirs(output_dir, exist_ok=True)

    # train the different combinations of models and fractions of the dataset
    execute_logic(data_dir, output_dir, epochs)


###############################################################################
# CLI Parser
###############################################################################
class TrainModelsParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_arguments()

    def add_arguments(self):
        self.add_processor_argument()
        self.add_epochs_argument()

    def add_processor_argument(self):
        self.add_argument(
            "--processor",
            help="Whether to use 'cpu' or 'gpu' for training",
            required=False,
            choices=["cpu", "gpu"],
            default="cpu",
        )

    def add_epochs_argument(self):
        self.add_argument(
            "--epochs",
            help="Number of epochs to train the model",
            required=True,
            type=int,
            default=10,
        )


if __name__ == "__main__":
    args = TrainModelsParser().parse_args()
    main(args)
