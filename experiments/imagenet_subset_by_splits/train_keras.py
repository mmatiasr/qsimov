import pickle
import os
from experiments.path_utils import (
    get_imagenet_subset_by_splits_results_dir as get_results_dir,
)
import argparse
import multiprocessing as mp
from experiments.imagenet_subset_by_splits.preprocess_data import (
    load_dataset,
    make_split,
)


# Possibly slow imports
def make_imports():
    global tf, keras, AccumulatedEpochTimeTracker, init_tensorflow
    global load_model, get_optimizer

    import tensorflow as tf
    from tensorflow import keras
    from experiments.mnist_speed_loss.tf_keras.utils import (
        init_tensorflow,
        AccumulatedEpochTimeTracker,
    )
    from experiments.imagenet_subset_by_splits.keras_model_factory import (
        load_model,
        get_optimizer,
    )


BATCH_SIZE = 64

###############################################################################
# Logic
###############################################################################


def save_results(
    split_name, model_name, model_type, model, history, results_dir
):
    with open(
        f"{results_dir}/{split_name}{model_type}_{model_name}_history.pkl",
        "wb",
    ) as f:
        pickle.dump(history.history, f)

    model.save(
        f"{results_dir}/{split_name}{model_type}_{model_name}_model.tf",
        save_format="tf",
    )

    # save the model summary
    with open(
        f"{results_dir}/{model_name}{model_type}_model_summary.txt", "w"
    ) as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))


def train_model(model, train_x, train_y, test_x, test_y, args):
    time_tracker = AccumulatedEpochTimeTracker()
    history = model.fit(
        x=train_x,
        y=train_y,
        validation_data=(test_x, test_y),
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        callbacks=[time_tracker],
    )
    history.history["time(s)"] = time_tracker.times
    return history


def split_to_name(split):
    if split is not None:
        return f"{split}_split"
    else:
        return "full_dataset"


def execute_logic(results_dir, split, args):
    # load dataset
    train_x, train_y, test_x, test_y = load_dataset()

    # make the split if required
    if split is not None:
        train_x, train_y = make_split(train_x, train_y, split)

    model_type = ""
    if args.train_path_selector:
        model_type = "_path_selector"

    # define the model to train
    model = load_model(
        args.model_name, args.train_path_selector, weights="imagenet"
    )

    history = train_model(model, train_x, train_y, test_x, test_y, args)

    # save the results
    save_results(
        split_to_name(split),
        args.model_name,
        model_type,
        model,
        history,
        results_dir,
    )


def main(args):
    # one process for each split to avoid memory issues
    def main_subprocess(split):
        # make slow imports (those that need to import tensorflow)
        make_imports()

        # init tensorflow
        init_tensorflow(tf, args.processor)

        # train the different combinations of models and fractions of the data
        execute_logic(results_dir, split, args)

    # directories to save the models
    results_dir = get_results_dir("keras", args.processor)
    os.makedirs(results_dir, exist_ok=True)

    # add None split (full dataset) and train in a separate process per split
    for split in (
        args.splits + [None] if not args.train_path_selector else args.splits
    ):
        print(f"\n\nTraining model with split: {split_to_name(split)}\n")
        p = mp.Process(target=main_subprocess, args=(split,))
        p.start()
        p.join()


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
        self.add_splits_argument()
        self.add_model_name_argument()
        self.add_train_path_selector()

    def add_processor_argument(self):
        self.add_argument(
            "--processor",
            help="Whether to use 'cpu' or 'gpu' for training",
            required=False,
            choices=["cpu", "gpu"],
            default="gpu",
        )

    def add_model_name_argument(self):
        self.add_argument(
            "--model-name",
            help="Name of the model to train",
            required=False,
            choices=["vgg16"],
            default="vgg16",
        )

    def add_epochs_argument(self):
        self.add_argument(
            "--epochs",
            help="Number of epochs to train the model",
            required=True,
            type=int,
            default=10,
        )

    def add_splits_argument(self):
        self.add_argument(
            "--splits",
            help="Number of samples in each split of the dataset",
            nargs="+",
            type=int,
            required=True,
        )

    def add_train_path_selector(self):
        self.add_argument(
            "--train-path-selector",
            action="store_true",
            help="Train path selector model",
        )


if __name__ == "__main__":
    args = TrainModelsParser().parse_args()
    main(args)
