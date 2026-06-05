import pickle
import os
from experiments.path_utils import (
    get_cifar10_gradient_by_splits_results_dir as get_results_dir,
)
import multiprocessing as mp
from experiments.cifar10_gradient_by_splits.preprocess_data import (
    load_dataset,
    make_split,
)
from experiments.cifar10_gradient_by_splits.train_keras_models import (
    BATCH_SIZE,
    split_to_name,
    TrainModelsParser,
)


# Possibly slow imports
def make_imports():
    global torch, nn, AccumulatedEpochTimeTracker, init_torch, fit
    global create_model, create_dataloader, samples_to_channels_first
    global get_optimizer, CustomCrossEntropyLoss

    import torch
    import torch.nn as nn
    from experiments.mnist_speed_loss.pytorch.utils import (
        init_torch,
        create_dataloader,
        fit,
        samples_to_channels_first,
        CustomCrossEntropyLoss,
    )
    from qsimov.pytorch_qsimov_gradient import AccumulatedEpochTimeTracker
    from experiments.cifar10_gradient_by_splits.pytorch_model_factory import (
        create_model,
        get_optimizer,
    )


###############################################################################
# Logic
###############################################################################


def save_results(name, model_name, model_type, model, history, results_dir):
    with open(
        f"{results_dir}/{name}{model_type}_{model_name}_history.pkl", "wb"
    ) as f:
        pickle.dump(history, f)

    torch.save(
        model, f"{results_dir}/{name}{model_type}_{model_name}_model.pt"
    )

    # save the model summary
    with open(
        f"{results_dir}/{model_name}{model_type}_model_summary.txt", "w"
    ) as f:
        f.write(str(model))


def train_model(model, train_x, train_y, test_x, test_y, device, args):
    history = fit(
        model,
        loss_function=CustomCrossEntropyLoss(),
        optimizer=get_optimizer(args.model_name, model.parameters()),
        train_dataloader=create_dataloader(
            train_x, train_y, batch_size=BATCH_SIZE
        ),
        test_dataloader=create_dataloader(
            test_x, test_y, batch_size=BATCH_SIZE
        ),
        epochs=args.epochs,
        device=device,
    )

    return history


def execute_logic(results_dir, split, device, args):
    # load dataset
    train_x, train_y, test_x, test_y = load_dataset()

    # to channels first
    train_x = samples_to_channels_first(train_x)
    test_x = samples_to_channels_first(test_x)

    # make the split
    if split is not None:
        split_train_x, split_train_y = make_split(train_x, train_y, split)
    else:  # use the full dataset
        split_train_x, split_train_y = train_x, train_y

    model_type = ""
    if args.train_path_selector:
        model_type = "_path_selector"

    # define the model to train
    model = create_model(args.model_name, model_type)

    history = train_model(
        model, split_train_x, split_train_y, test_x, test_y, device, args
    )

    # save the results
    save_results(
        split_to_name(split),
        args.model_name,
        model_type,
        model,
        history,
        results_dir,
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
    for split in (
        args.splits + [None] if not args.train_path_selector else args.splits
    ):
        print(f"\n\nTraining model with split: {split_to_name(split)}\n")
        p = mp.Process(target=main_subprocess, args=(results_dir, split, args))
        p.start()
        p.join()


if __name__ == "__main__":
    args = TrainModelsParser().parse_args()
    main(args)
