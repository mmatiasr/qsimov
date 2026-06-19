import numpy as np
import pickle
import os
from experiments.path_utils import (
    get_qsimov_dataset_dir,
    get_mnist_learning_rate_results_dir,
)
from experiments.mnist_speed_loss.preprocess_data import (
    get_train_test_data,
    concat_batches,
)
from experiments.mnist_learning_rate.train_keras_qsimov_models import (
    BATCH_SIZE,
)


# possibly slow imports
def make_imports():
    global build_mnist, create_dataloader, accuracy, CustomCrossEntropyLoss
    global samples_to_channels_first, fit, init_torch, clone_model_with_weights
    global PytorchQsimovGradient, PytorchPathSelector, torch, Adam
    from experiments.mnist_speed_loss.pytorch.model_factory import build_mnist
    from experiments.mnist_speed_loss.pytorch.utils import (
        create_dataloader,
        accuracy,
        CustomCrossEntropyLoss,
        samples_to_channels_first,
        fit,
        init_torch,
        clone_model_with_weights,
    )

    from qsimov.pytorch_qsimov_gradient import PytorchQsimovGradient
    from qsimov.pytorch_path_selector import PytorchPathSelector
    import torch
    from torch.optim import Adam


BASE_MODEL = None


def load_base_model(in_channels, loss):
    global BASE_MODEL
    if BASE_MODEL is None:
        BASE_MODEL = build_mnist(
            in_channels=in_channels,
            loss=loss,
        )
    return clone_model_with_weights(BASE_MODEL)


def save_results(
    output_dir,
    pytorch_stats,
    qsimov_stats,
    number_paths,
    learning_rates,
    multiples,
    base_learning_rate,
):
    stats = [(qsimov_stats, "qsimov"), (pytorch_stats, "pytorch")]
    for _, (history, method) in enumerate(stats):
        with open(
            f"{output_dir}/learning_rate_{method}_stats.pkl",
            "wb",
        ) as f:
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
    input_shape,
    initial_layer,
    device,
    loss,
    train_x,
    train_y,
    test_x,
    test_y,
    epochs,
    learning_rate,
):
    # Load a model trained with half of the dataset
    partial_model = torch.load(f"{output_dir}/crossentropy_half_model.pt", weights_only=False)
    qsimov_gradient = PytorchQsimovGradient(
        PytorchPathSelector(
            neural_network=partial_model,
            input_shape=input_shape,
            initial_layer=initial_layer,
            verbose=1,
            device=device,
        )
    )

    # train the qsimov model
    stats = qsimov_gradient.fit(
        train_x,
        train_y,
        X_val=test_x,
        Y_val=test_y,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        loss_function=loss,
        optimizer=lambda params: Adam(params, lr=learning_rate),
        metrics=[accuracy],
        device=device,
    )
    # compute number of paths
    number_paths = np.sum(
        qsimov_gradient._path_selector.output_masks_, axis=1
    ).max()
    return stats, number_paths


def train_pytorch_model(
    input_shape,
    loss,
    train_dataloader,
    test_dataloader,
    device,
    epochs,
    learning_rate,
):
    # load the starting model, but with same initial random weights
    model = load_base_model(
        in_channels=input_shape[0], loss="categorical_crossentropy"
    )

    return fit(
        model,
        loss,
        Adam(model.parameters(), lr=learning_rate),
        train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        epochs=epochs,
    )


def train_models(
    output_dir,
    train_x,
    train_y,
    test_x,
    test_y,
    epochs,
    device,
    initial_layer,
):
    # create dataloaders
    test_dataloader = create_dataloader(test_x, test_y, batch_size=BATCH_SIZE)
    train_dataloader = create_dataloader(
        train_x, train_y, batch_size=BATCH_SIZE
    )

    # define differents learning rates and the corresponding multiples
    base_learning_rate = 0.001
    multiples = np.power(4.0, np.arange(-2, 4))
    learning_rates = base_learning_rate * multiples

    # initialize the stats
    pytorch_stats = []
    qsimov_stats = []

    # input shape of the model
    input_shape = test_x.shape[1:]

    # loss function
    loss = CustomCrossEntropyLoss()

    # separator for printing
    separator = "\n" + "-" * 50 + "\n"
    for idx, learning_rate in enumerate(learning_rates):
        print(
            f"Learning rate {1+idx}/{len(learning_rates)}: {learning_rate}"
            + separator
        )
        # train the pytorch model
        pytorch_stats.append(
            train_pytorch_model(
                input_shape,
                loss,
                train_dataloader,
                test_dataloader,
                device,
                epochs,
                learning_rate,
            )
        )
        # train the qsimov model
        stats, number_paths = train_qsimov_model(
            output_dir,
            input_shape,
            initial_layer,
            device,
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
        pytorch_stats,
        qsimov_stats,
        number_paths,
        learning_rates,
        multiples,
        base_learning_rate,
    )


def execute_logic(data_dir, output_dir, epochs, initial_layer, device):
    # load dataset
    train_xs, train_ys, test_x, test_y = get_train_test_data(data_dir)

    # convert to channels first and create full dataset
    train_x = samples_to_channels_first(concat_batches(train_xs))
    train_y = concat_batches(train_ys)
    test_x = samples_to_channels_first(test_x)

    # train the models
    train_models(
        output_dir,
        train_x,
        train_y,
        test_x,
        test_y,
        epochs,
        device,
        initial_layer,
    )


def main(args):
    make_imports()

    # extract arguments
    processor = args.processor
    epochs = args.epochs
    initial_layer = args.initial_layer

    # set device and seed
    device = init_torch(processor)

    # input and output directories
    data_dir = get_qsimov_dataset_dir("mnist")
    output_dir = get_mnist_learning_rate_results_dir("pytorch", processor)
    os.makedirs(output_dir, exist_ok=True)

    # train the different combinations of learning rates
    execute_logic(data_dir, output_dir, epochs, initial_layer, device)


if __name__ == "__main__":
    from experiments.mnist_speed_loss.tf_keras.train_qsimov_models import (
        TrainQsimovModelsParser,
    )

    main(TrainQsimovModelsParser().parse_args())
