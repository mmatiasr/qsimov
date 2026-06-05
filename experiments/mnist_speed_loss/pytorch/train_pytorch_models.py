import os
import pickle
from experiments.mnist_speed_loss.tf_keras.train_keras_models import (
    TrainModelsParser,
)
from experiments.path_utils import (
    get_qsimov_dataset_dir,
    get_mnist_speed_loss_results_dir as get_results_dir,
)


# Possibly slow imports
def make_imports():
    global concat_batches, accuracy, create_dataloader, fit
    global torch, nn, CustomCrossEntropyLoss, get_train_test_data
    global samples_to_channels_first, init_torch, clone_model_with_weights
    global build_mnist, adam
    from experiments.mnist_speed_loss.pytorch.utils import (
        create_dataloader,
        samples_to_channels_first,
        adam,
        CustomCrossEntropyLoss,
        fit,
        init_torch,
        clone_model_with_weights,
    )
    from experiments.mnist_speed_loss.preprocess_data import (
        get_train_test_data,
        concat_batches,
    )
    from experiments.mnist_speed_loss.pytorch.model_factory import build_mnist
    import torch
    import torch.nn as nn


BATCH_SIZE = 32

###############################################################################
# Logic
###############################################################################


def build_models(in_channels):
    crossentropy_model = build_mnist(in_channels, "categorical_crossentropy")
    crossentropy_models = [
        (crossentropy_model, "crossentropy_half"),
        (clone_model_with_weights(crossentropy_model), "crossentropy_quarter"),
    ]
    mse_model = build_mnist(in_channels, "mse")
    mse_models = [
        (mse_model, "mse_half"),
        (clone_model_with_weights(mse_model), "mse_quarter"),
    ]
    return [crossentropy_models, mse_models]


def save_results(name, model, history, output_dir):
    with open(f"{output_dir}/{name}_model_history.pkl", "wb") as f:
        pickle.dump(history, f)

    torch.save(model, f"{output_dir}/{name}_model.pt")

    # print summary to file
    with open(f"{output_dir}/model_summary.txt", "w") as f:
        f.write(str(model))


def train_models(
    train_dataloader,
    partial_dataloader,
    test_dataloader,
    model,
    model_name,
    epochs,
    loss_function,
    optimizer,
    output_dir,
    device,
):
    # separator for printing
    separator = "\n" + "-" * 50 + "\n"

    # make a copy for training the full model later
    full_model = clone_model_with_weights(model)

    # partial model
    print(f"Training {model_name}" + separator)
    history = fit(
        model,
        loss_function,
        optimizer(model.parameters()),
        partial_dataloader,
        test_dataloader,
        epochs=epochs,
        device=device,
    )
    save_results(model_name, model, history, output_dir)

    # model training all the dataset
    print(f"Training {model_name} model with full dataset" + separator)

    history = fit(
        full_model,
        loss_function,
        optimizer(full_model.parameters()),
        train_dataloader,
        test_dataloader,
        epochs=epochs,
        device=device,
    )
    save_results(f"{model_name}_full", full_model, history, output_dir)


def make_dataloader_portions(train_xs, train_ys):
    return [
        (
            create_dataloader(
                concat_batches(train_xs[:2]),
                concat_batches(train_ys[:2]),
                batch_size=BATCH_SIZE,
            ),
            "half",
        ),
        (
            create_dataloader(
                concat_batches(train_xs[:1]),
                concat_batches(train_ys[:1]),
                batch_size=BATCH_SIZE,
            ),
            "quarter",
        ),
    ]


def execute_logic(data_dir, output_dir, epochs, device):
    # load dataset
    train_xs, train_ys, test_x, test_y = get_train_test_data(data_dir)

    # convert to channels first
    for i in range(len(train_xs)):
        train_xs[i] = samples_to_channels_first(train_xs[i])
    test_x = samples_to_channels_first(test_x)

    # build models
    models = build_models(test_x.shape[1])

    # partial dataloader
    partial_datas = make_dataloader_portions(train_xs, train_ys)

    # datasets to dataloader and conversion to channels first
    train_dataloader = create_dataloader(
        concat_batches(train_xs),
        concat_batches(train_ys),
        batch_size=BATCH_SIZE,
    )
    test_dataloader = create_dataloader(test_x, test_y, batch_size=512)

    # loss functions to use
    loss_functions = [
        (CustomCrossEntropyLoss(), "categorical_crossentropy"),
        (nn.MSELoss(), "mse"),
    ]

    # separator for printing
    separator = "\n" + "-" * 50 + "\n"

    for partial_idx, (partial_dataloader, name) in enumerate(partial_datas):
        print(f"Training {name} models" + separator)
        for loss_idx, (loss_function, loss_function_name) in enumerate(
            loss_functions
        ):
            print(f"with {loss_function_name} loss function" + separator)
            current_models = models[loss_idx]
            model, model_name = current_models[partial_idx]

            # train partial_model
            train_models(
                train_dataloader,
                partial_dataloader,
                test_dataloader,
                model,
                model_name,
                epochs,
                loss_function,
                adam,
                output_dir,
                device,
            )


def main(args):
    # make imports here to not force them on other scripts that may
    # want to import util functions only
    make_imports()

    # arguments
    processor, epochs = args.processor, args.epochs

    # init torch
    device = init_torch(processor)

    # directories to save the models
    data_dir = get_qsimov_dataset_dir("mnist")
    output_dir = get_results_dir("pytorch", processor)
    os.makedirs(output_dir, exist_ok=True)

    # train the different combinations of models and fractions of the dataset
    execute_logic(data_dir, output_dir, epochs, device)


if __name__ == "__main__":
    main(TrainModelsParser().parse_args())
