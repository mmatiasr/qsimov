import pickle
import os
from experiments.cifar10_gradient_by_splits.train_keras_models import (
    split_to_name,
    BATCH_SIZE,
)
from experiments.path_utils import (
    get_cifar10_gradient_by_splits_results_dir as get_results_dir,
)
from experiments.cifar10_gradient_by_splits.train_keras_qsimov_models import (
    TrainQsimovModelsParser,
)
import numpy as np
import multiprocessing as mp


# Possibly slow imports
def make_imports():
    global torch, nn, init_torch, fit, samples_to_channels_first
    global load_dataset, accuracy, get_optimizer
    global PytorchPathSelector, PytorchQsimovGradient, CustomCrossEntropyLoss

    import torch
    import torch.nn as nn
    from qsimov.pytorch_path_selector import PytorchPathSelector
    from qsimov.pytorch_qsimov_gradient import PytorchQsimovGradient
    from experiments.mnist_speed_loss.pytorch.utils import (
        init_torch,
        fit,
        accuracy,
        samples_to_channels_first,
        CustomCrossEntropyLoss,
    )
    from experiments.cifar10_gradient_by_splits.preprocess_data import (
        load_dataset,
    )
    from experiments.cifar10_gradient_by_splits.pytorch_model_factory import (
        get_optimizer,
    )


###############################################################################
# Logic
###############################################################################


def save_results(name, model_name, model, history, results_dir):
    name += "_qsimov"
    with open(f"{results_dir}/{name}_{model_name}_history.pkl", "wb") as f:
        pickle.dump(history, f)

    # compute total number of paths
    number_of_paths = (
        np.sum(model._path_selector.output_masks_, axis=1).max().astype(int)
    )

    # save the number of paths
    with open(f"{results_dir}/number_of_paths_{model_name}.txt", "w") as f:
        f.write(str(number_of_paths))


def train_model(model, train_x, train_y, test_x, test_y, device, args):
    history = model.fit(
        train_x,
        train_y,
        test_x,
        test_y,
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        device=device,
        optimizer=lambda params: get_optimizer(args.model_name, params),
        loss_function=CustomCrossEntropyLoss(),
        metrics=[accuracy],
    )
    return history


def load_base_model(results_dir, model_name, split):
    # model name corresponding to the split
    model_file = f"{split_to_name(split)}_path_selector_{model_name}_model.pt"

    return torch.load(os.path.join(results_dir, model_file), weights_only=False)


def make_qsimov_model(results_dir, split, input_shape, device, args):
    # build the qsimov gradient model
    path_selector = PytorchPathSelector(
        load_base_model(results_dir, args.model_name, split),
        input_shape,
        args.initial_layer,
        device=device,
    )
    return PytorchQsimovGradient(path_selector)


def execute_logic(results_dir, split, device, args):
    # load dataset
    train_x, train_y, test_x, test_y = load_dataset()

    # to channels first
    train_x = samples_to_channels_first(train_x)
    test_x = samples_to_channels_first(test_x)

    # get input shape
    input_shape = train_x.shape[1:]

    # define the model to train
    model = make_qsimov_model(results_dir, split, input_shape, device, args)

    # train the model
    history = train_model(
        model, train_x, train_y, test_x, test_y, device, args
    )

    # save the results
    save_results(
        split_to_name(split), args.model_name, model, history, results_dir
    )


# one process for each split to avoid memory issues
def main_subprocess(results_dir, split, args):
    # make slow imports (those that need to import tensorflow)
    make_imports()

    # init tensorflow
    device = init_torch(args.processor)

    # train the different combinations of models and fractions of the data
    execute_logic(results_dir, split, device, args)


def main(args):
    # directories to save the models
    results_dir = get_results_dir("pytorch", args.processor)
    os.makedirs(results_dir, exist_ok=True)

    # add None split (full dataset) and train in a separate process per split
    mp.set_start_method("spawn", force=True)  # prevent freeze in pytorch
    for split in args.splits:
        print(f"\n\nTraining model with split: {split_to_name(split)}\n")
        p = mp.Process(target=main_subprocess, args=(results_dir, split, args))
        p.start()
        p.join()


if __name__ == "__main__":
    args = TrainQsimovModelsParser().parse_args()
    main(args)
