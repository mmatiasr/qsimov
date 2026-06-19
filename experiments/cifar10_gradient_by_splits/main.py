import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import experiments.git as exp_git
import experiments.mlflow as exp_mlflow
import mlflow
import os
import shutil
from experiments.path_utils import (
    as_relative_path,
    get_qsimov_dataset_dir,
    get_qsimov_experiments_dir,
    get_cifar10_gradient_by_splits_results_dir as get_script_results_dir,
    get_cifar10_gradient_by_splits_results_initial_weights_dir,
)
from experiments.cifar10_gradient_by_splits.train_keras_qsimov_models import (
    TrainModelsParser,
    TrainQsimovModelsParser,
    split_to_name,
)
from experiments.cifar10_gradient_by_splits.plot_gradient_by_splits import (
    PlotGradientBySplitsParser,
)

EXPERIMENT_DESCRIPTION = (
    "Runs classification experiments on cifar 10, applying keras/torch"
    " models and qsimov models. A comparison is made between different "
    "splits of the dataset of the accuracy by epoch and time."
)
EXPERIMENT_NAME = "cifar_10_gradient_by_splits"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = get_qsimov_experiments_dir()


class GradientBySplitsParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_arguments()

    def parse_args(self, experiment_name):
        args = argparse.ArgumentParser.parse_args(self)

        # used for path names
        if args.framework == "keras":
            args.framework_name = "tf_keras"
        elif args.framework == "pytorch":
            args.framework_name = "pytorch"

        args.run_name = args.run_name or "{}_{}_{}_{}_{}".format(
            args.framework,
            experiment_name,
            args.model_name,
            str(args.initial_layer),
            args.processor,
        )

        args.splits = [str(split) for split in args.splits]

        return args

    def add_arguments(self):
        self.add_skip_data_preparation_argument()
        self.add_skip_initialize_weights()
        self.add_skip_models_argument()
        self.add_skip_path_selector_models_argument()
        self.add_skip_qsimov_argument()
        TrainQsimovModelsParser.add_initial_layer_argument(self)
        TrainModelsParser.add_splits_argument(self)
        TrainModelsParser.add_epochs_argument(self)
        TrainModelsParser.add_model_name_argument(self)
        PlotGradientBySplitsParser.add_framework_argument(self)
        PlotGradientBySplitsParser.add_processor_argument(self)
        self.add_run_name_argument()

    def add_skip_data_preparation_argument(self):
        self.add_argument(
            "--skip-data-preparation",
            action="store_true",
            help="Skip data preparation step (preprocessing and splitting)."
            " Useful if you want to run the experiments multiple times",
        )

    def add_skip_initialize_weights(self):
        self.add_argument(
            "--skip-initialize-weights",
            action="store_true",
            help="Skip load weights model (create model and save weights)."
            " Useful if you want to run the experiments multiple times",
        )

    def add_skip_models_argument(self):
        self.add_argument(
            "--skip-models",
            action="store_true",
            help="Skip generation of the keras/pytorch models. Useful if "
            "you want to run the experiments multiple times",
        )

    def add_skip_path_selector_models_argument(self):
        self.add_argument(
            "--skip-path-selector-models",
            action="store_true",
            help="Skip generation of the keras/pytorch path selector models. "
            "Useful if you want to run the experiments multiple times",
        )

    def add_skip_qsimov_argument(self):
        self.add_argument(
            "--skip-qsimov",
            action="store_true",
            help="Skip generation of the qsimov models. This is useful if you"
            " want to run the experiments multiple times",
        )

    def add_run_name_argument(self):
        self.add_argument(
            "--run-name",
            type=str,
            required=False,
            default=None,
            help="Name of the run in mlflow",
        )


def parse_arguments():
    parser = GradientBySplitsParser(description=EXPERIMENT_DESCRIPTION)
    return parser.parse_args(EXPERIMENT_NAME)


def copy_to_experiment_results(args, src, dst=None):
    # if dst is not set, use the same name as src
    if dst is None:
        dst = src

    # where the script results are saved
    script_results_dir = get_script_results_dir(args.framework, args.processor)

    # where the experiment results are saved
    experiment_results_dir = exp_mlflow.get_or_make_export_directory(
        EXPERIMENT_DIR
    )
    shutil.copy(
        os.path.join(script_results_dir, src),
        os.path.join(experiment_results_dir, dst),
    )


def prepare_data(args):
    script = os.path.join(EXPERIMENT_DIR, "preprocess_data.py")

    # Run the script
    if not args.skip_data_preparation:
        script_args = ["python", script]
        exp_mlflow.run_script(script_args, EXPERIMENT_DIR)

    # Log the used script
    mlflow.log_param("data_preparation_script", as_relative_path(script))

    data_dir = get_qsimov_dataset_dir("cifar10")
    mlflow.log_param("generated_data_dir", as_relative_path(data_dir))


def load_weights(args):
    if args.framework == "pytorch":
        script = os.path.join(EXPERIMENT_DIR, "pytorch_model_factory.py")
    else:
        script = os.path.join(EXPERIMENT_DIR, "keras_model_factory.py")
    # where weights are saved in the script
    script_weights_dir = (
        get_cifar10_gradient_by_splits_results_initial_weights_dir()
    )

    # Run the script
    if not args.skip_initialize_weights:
        script_args = [
            "python",
            script,
            "--model_name",
            args.model_name,
        ]
        exp_mlflow.run_script(script_args, EXPERIMENT_DIR)

    # Log the used script
    mlflow.log_param("initialized_weights_script", as_relative_path(script))

    # Log the generated weights directory
    mlflow.log_param(
        "initialized_weights_dir", as_relative_path(script_weights_dir)
    )

    for model_type in ("", "_path_selector"):
        # Log the model
        mlflow.log_param(
            f"initial_weights{model_type}_model",
            os.path.join(
                script_weights_dir, f"{args.model_name}{model_type}_model.tf"
            ),
        )

        # Log the pickle
        mlflow.log_param(
            f"initial_weights{model_type}",
            os.path.join(
                script_weights_dir,
                f"{args.model_name}{model_type}_weights.pkl",
            ),
        )


def create_models(args):
    script = os.path.join(
        EXPERIMENT_DIR,
        f"train_{args.framework}_models.py",
    )
    # where models and histories are saved in the script
    script_results_dir = get_script_results_dir(args.framework, args.processor)

    # Run the script
    if not args.skip_models:
        script_args = [
            "python",
            script,
            "--processor",
            args.processor,
            "--model_name",
            args.model_name,
            "--epochs",
            str(args.epochs),
            "--splits",
            *args.splits,
        ]
        exp_mlflow.run_script(script_args, EXPERIMENT_DIR)

    # Log the used script
    mlflow.log_param(
        f"{args.framework}_models_script", as_relative_path(script)
    )
    framework_extension = ".pt" if args.framework == "pytorch" else ".tf"

    for is_model in (True, False):
        # choose between model or history
        end = "_model" + framework_extension if is_model else "_history.pkl"
        files = []
        for split in args.splits + [None]:
            files.append(f"{split_to_name(split)}_{args.model_name}{end}")

        mlflow.log_param(
            "{}_{}".format(
                args.framework, "models" if is_model else "histories"
            ),
            ", ".join(files),
        )

    copy_to_experiment_results(args, f"{args.model_name}_model_summary.txt")

    # log the file
    mlflow.log_param(
        "model_summary",
        os.path.join(
            script_results_dir, f"{args.model_name}_model_summary.txt"
        ),
    )


def create_path_selector_models(args):
    script = os.path.join(
        EXPERIMENT_DIR,
        f"train_{args.framework}_models.py",
    )
    # where models and histories are saved in the script
    script_results_dir = get_script_results_dir(args.framework, args.processor)

    # Run the script
    if not args.skip_path_selector_models:
        script_args = [
            "python",
            script,
            "--processor",
            args.processor,
            "--model_name",
            args.model_name,
            "--epochs",
            str(args.epochs),
            "--splits",
            *args.splits,
            "--train-path-selector",
        ]
        exp_mlflow.run_script(script_args, EXPERIMENT_DIR)

    # Log the used script
    mlflow.log_param(
        "path_selector_models_script",
        as_relative_path(script),
    )
    framework_extension = ".pt" if args.framework == "pytorch" else ".tf"

    for is_model in (True, False):
        # choose between model or history
        end = "_model" + framework_extension if is_model else "_history.pkl"
        files = []
        for split in args.splits + [None]:
            files.append(
                f"{split_to_name(split)}_path_selector_"
                f"{args.model_name}{end}"
            )

        mlflow.log_param(
            "path_selector_{}".format("models" if is_model else "histories"),
            ", ".join(files),
        )

    copy_to_experiment_results(
        args, f"{args.model_name}_path_selector_model_summary.txt"
    )

    # log the file
    mlflow.log_param(
        "path_selector_model_summary",
        os.path.join(
            script_results_dir,
            f"{args.model_name}_path_selector_model_summary.txt",
        ),
    )


def create_qsimov_models(args):
    script = os.path.join(
        EXPERIMENT_DIR, f"train_{args.framework}_qsimov_models.py"
    )
    # where models and histories are saved in the script
    script_results_dir = get_script_results_dir(args.framework, args.processor)

    # Run the script
    if not args.skip_qsimov:
        script_args = [
            "python",
            script,
            "--initial-layer",
            str(args.initial_layer),
            "--processor",
            args.processor,
            "--model_name",
            args.model_name,
            "--epochs",
            str(args.epochs),
            "--splits",
            *args.splits,
        ]
        exp_mlflow.run_script(script_args, EXPERIMENT_DIR)

    # Log the used script
    mlflow.log_param("qsimov_models_script", as_relative_path(script))

    # Log the generated histories
    files = []
    for split in args.splits:
        files.append(f"{split}_split_qsimov_{args.model_name}_history.pkl")
    mlflow.log_param("qsimov_histories", ", ".join(files))

    # Copy the file with the number of paths
    experiment_results_dir = exp_mlflow.get_or_make_export_directory(
        EXPERIMENT_DIR
    )
    shutil.copy(
        os.path.join(
            script_results_dir, f"number_of_paths_{args.model_name}.txt"
        ),
        os.path.join(
            experiment_results_dir, f"number_of_paths_{args.model_name}.txt"
        ),
    )

    # log the file
    mlflow.log_param(
        "number_of_paths",
        os.path.join(
            script_results_dir, f"number_of_paths_{args.model_name}.txt"
        ),
    )


def create_plots(args):
    script = os.path.join(EXPERIMENT_DIR, "plot_gradient_by_splits.py")
    # Run the script
    exp_mlflow.run_script(
        [
            "python",
            script,
            "--processor",
            args.processor,
            "--model_name",
            args.model_name,
            "--framework",
            args.framework,
            "--splits",
            *args.splits,
        ],
        EXPERIMENT_DIR,
    )

    # Log the used script
    mlflow.log_param("plots_script", as_relative_path(script))

    # Log the generated plots directory
    script_results_dir = get_script_results_dir(args.framework, args.processor)
    mlflow.log_param(
        "generated_plots_dir", as_relative_path(script_results_dir)
    )

    # Results directory for the experiment
    experiment_results_dir = exp_mlflow.get_or_make_export_directory(
        EXPERIMENT_DIR
    )

    # Copy the plots to the experiment results directory
    # with a name that includes the run start time
    run_start_time = exp_mlflow.get_run_start_time()
    file = f"cifar10_gradient_by_splits_{args.model_name}.html"
    new_file = "{}_{}_{}".format(run_start_time, args.run_name, file)
    shutil.copy(
        os.path.join(script_results_dir, file),
        os.path.join(experiment_results_dir, new_file),
    )

    # Log the plots
    mlflow.log_param("plots", new_file)

    # Copy the file with the number of paths
    shutil.copy(
        os.path.join(
            script_results_dir, f"number_of_paths_{args.model_name}.txt"
        ),
        os.path.join(
            experiment_results_dir, f"number_of_paths_{args.model_name}.txt"
        ),
    )


def run_experiment(args):
    # Prepare the data
    prepare_data(args)

    # Load model weights
    load_weights(args)

    # Run the scripts and retrieve the number of paths
    # that will be used for the qsimov models
    create_models(args)

    # Run the path selector scripts
    create_path_selector_models(args)

    # Run the qsimov scripts
    create_qsimov_models(args)

    # Run the plotting scripts
    create_plots(args)


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Check that the git repo is clean
    exp_git.check_git_clean()

    # Start a new MLflow run
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=args.run_name):
        # Log the command line arguments
        mlflow.log_param("experiment_execution", " ".join(sys.argv))

        # Run the experiment
        run_experiment(args)

        # Export the run to a JSON file
        exp_mlflow.export_active_run(EXPERIMENT_DIR)


if __name__ == "__main__":
    main()
