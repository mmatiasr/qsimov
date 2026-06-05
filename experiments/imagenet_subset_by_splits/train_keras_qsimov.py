import pickle
import os
from experiments.imagenet_subset_by_splits.train_keras import (
    TrainModelsParser,
    split_to_name,
    BATCH_SIZE,
)
from experiments.path_utils import (
    get_imagenet_subset_by_splits_results_dir as get_results_dir,
)
import numpy as np
import multiprocessing as mp
from experiments.imagenet_subset_by_splits.preprocess_data import (
    load_dataset,
)


# Possibly slow imports
def make_imports():
    global tf, keras, AccumulatedEpochTimeTracker, init_tensorflow
    global get_optimizer
    global KerasPathSelector, KerasQsimovGradient

    import tensorflow as tf
    from tensorflow import keras
    from qsimov.keras_path_selector import KerasPathSelector
    from qsimov.keras_qsimov_gradient import KerasQsimovGradient
    from experiments.mnist_speed_loss.tf_keras.utils import (
        init_tensorflow,
        AccumulatedEpochTimeTracker,
    )
    from experiments.imagenet_subset_by_splits.keras_model_factory import (
        get_optimizer,
    )


###############################################################################
# Logic
###############################################################################


def save_results(name, model_name, model, history, results_dir):
    name += "_qsimov"
    with open(f"{results_dir}/{name}_{model_name}_history.pkl", "wb") as f:
        pickle.dump(history.history, f)

    # compute total number of paths
    number_of_paths = (
        np.sum(model._path_selector.output_masks_, axis=1).max().astype(int)
    )

    # save the number of paths
    with open(f"{results_dir}/number_of_paths_{model_name}.txt", "w") as f:
        f.write(str(number_of_paths))


def train_model(model, train_x, train_y, test_x, test_y, args):
    time_tracker = AccumulatedEpochTimeTracker()
    history = model.fit(
        X=train_x,
        Y=train_y,
        validation_data=(test_x, test_y),
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        callbacks=[time_tracker],
    )
    history.history["time(s)"] = time_tracker.times
    return history


def load_base_model(results_dir, model_name, split):
    # model name corresponding to the split
    model_file = f"{split_to_name(split)}_path_selector_{model_name}_model.tf"
    return keras.models.load_model(os.path.join(results_dir, model_file))


def make_qsimov_model(results_dir, split, args):
    # build the qsimov gradient model
    path_selector = KerasPathSelector(
        load_base_model(results_dir, args.model_name, split),
        args.initial_layer,
    )
    qsimov_gradient = KerasQsimovGradient(path_selector)

    # compile the model
    qsimov_gradient.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=get_optimizer(model=args.model_name, is_qsimov=True),
        metrics=["accuracy"],
    )

    return qsimov_gradient


def execute_logic(results_dir, split, args):
    # load dataset
    train_x, train_y, test_x, test_y = load_dataset()

    # define the model to train
    model = make_qsimov_model(results_dir, split, args)

    # train the model
    history = train_model(model, train_x, train_y, test_x, test_y, args)

    # save the results
    save_results(
        split_to_name(split), args.model_name, model, history, results_dir
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

    # get results for each split in a different process
    for split in args.splits:
        print(f"\n\nTraining model with split: {split_to_name(split)}\n")
        p = mp.Process(target=main_subprocess, args=(split,))
        p.start()
        p.join()


###############################################################################
# CLI Parser
###############################################################################
class TrainQsimovModelsParser(TrainModelsParser):
    def add_arguments(self):
        self.add_processor_argument()
        self.add_epochs_argument()
        self.add_splits_argument()
        self.add_model_name_argument()
        self.add_initial_layer_argument()

    def add_initial_layer_argument(self):
        self.add_argument(
            "--initial-layer",
            type=int,
            default=-2,
            help="Initial layer of the qsimov model",
        )


if __name__ == "__main__":
    args = TrainQsimovModelsParser().parse_args()
    main(args)
