import os


def join_normalize_paths(*args):
    return os.path.normpath(os.path.join(*args))


def get_qsimov_home():
    return os.environ.get(
        "QSIMOV_HOME", join_normalize_paths(os.path.dirname(__file__), "..")
    )


def as_relative_path(path):
    return path[len(get_qsimov_home()) :]


def get_qsimov_data_dir():
    return join_normalize_paths(get_qsimov_home(), "data")


def get_qsimov_dataset_dir(dataset_name):
    return join_normalize_paths(get_qsimov_data_dir(), dataset_name)


def get_qsimov_results_dir():
    # use of the environment variable QSIMOV_RESULTS_DIR useful for
    # running the same experiment at once, so that results are not
    # overwritten
    return os.environ.get(
        "QSIMOV_RESULTS_DIR",
        join_normalize_paths(get_qsimov_home(), "results"),
    )


def get_qsimov_experiments_dir():
    return join_normalize_paths(get_qsimov_home(), "experiments")


def get_mnist_speed_loss_results_dir(framework, processor):
    framework_name = "tf_keras" if framework != "pytorch" else "pytorch"

    return join_normalize_paths(
        get_qsimov_results_dir(),
        "mnist",
        "learning_rate_speed_loss",
        framework_name,
        processor,
    )


# shared with mnist_speed_loss
def get_mnist_learning_rate_results_dir(framework, processor):
    return get_mnist_speed_loss_results_dir(framework, processor)


def get_mnist_forgetting_results_dir(framework, processor):
    framework_name = "tf_keras" if framework != "pytorch" else "pytorch"

    return join_normalize_paths(
        get_qsimov_results_dir(),
        "mnist",
        "forgetting",
        framework_name,
        processor,
    )


def get_cifar10_forgetting_results_dir(framework, processor):
    framework_name = "tf_keras" if framework != "pytorch" else "pytorch"
    return join_normalize_paths(
        get_qsimov_results_dir(), "cifar10", "forgetting", framework_name, processor
    )


###############################################################################
# CIFAR10
###############################################################################


def get_cifar10_gradient_by_splits_results_dir(framework, processor):
    framework_name = "tf_keras" if framework != "pytorch" else "pytorch"

    return join_normalize_paths(
        get_qsimov_results_dir(),
        "cifar10",
        "gradient_by_splits",
        framework_name,
        processor,
    )


def get_cifar10_gradient_by_splits_results_initial_weights_dir():
    return join_normalize_paths(
        get_qsimov_results_dir(),
        "cifar10",
        "gradient_by_splits",
        "tf_keras",
        "initial_weights",
    )




###############################################################################
# ImageNet32
###############################################################################


def get_imagenet_subset_by_splits_results_dir(framework, processor):
    framework_name = "tf_keras" if framework != "pytorch" else "pytorch"

    return join_normalize_paths(
        get_qsimov_results_dir(),
        "imagenet_subset",
        "gradient_by_splits",
        framework_name,
        processor,
    )


###############################################################################
# ImageNet continual learning
###############################################################################


def get_imagenet_continual_learning_results_dir(processor, framework="keras"):
    fw = "pytorch" if framework == "pytorch" else "tf_keras"
    return join_normalize_paths(
        get_qsimov_results_dir(),
        "imagenet_subset",
        "continual_learning",
        fw,
        processor,
    )


###############################################################################
# ImageNet streaming incremental
###############################################################################


def get_imagenet_streaming_incremental_results_dir(processor, framework="keras"):
    fw = "pytorch" if framework == "pytorch" else "tf_keras"
    return join_normalize_paths(
        get_qsimov_results_dir(),
        "imagenet_subset",
        "streaming_incremental",
        fw,
        processor,
    )


###############################################################################
# Initial layer sweep
###############################################################################


def get_initial_layer_sweep_results_dir(processor, framework="keras"):
    fw = "pytorch" if framework == "pytorch" else "tf_keras"
    return join_normalize_paths(
        get_qsimov_results_dir(),
        "imagenet_subset",
        "initial_layer_sweep",
        fw,
        processor,
    )
