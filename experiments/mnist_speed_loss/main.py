import sys
import argparse
import experiments.git as exp_git
import experiments.mlflow as exp_mlflow
import mlflow
import os
from itertools import product
import shutil
from experiments.path_utils import (
    get_qsimov_dataset_dir,
    get_qsimov_experiments_dir,
    get_mnist_speed_loss_results_dir as get_results_dir,
    as_relative_path,
)
from experiments.mnist_speed_loss.tf_keras.train_qsimov_models import (
    TrainQsimovModelsParser,
)
from experiments.mnist_speed_loss.speed_loss_plots import SpeedLossPlotsParser

EXPERIMENT_DESCRIPTION = (
    "Runs classification experiments on mnist, applying keras/torch"
    " models and qsimov models. Both MSE and cross entropy are used"
    " so that QsimovLinearSystem and QsimovGradient can be compared."
    " The experiments generate plots that compare accuracy and loss across"
    " epochs and time."
)
EXPERIMENT_NAME = "mnist_speed_loss"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = get_qsimov_experiments_dir()


class MnistSpeedLossParser(TrainQsimovModelsParser, SpeedLossPlotsParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_arguments()

    @staticmethod
    def framework_name(framework):
        if framework == "keras":
            return "tf_keras"
        elif framework == "pytorch":
            return "pytorch"

    @staticmethod
    def default_run_name(framework, experiment_name, initial_layer, processor):
        return "{}_{}_{}_{}".format(
            framework, experiment_name, str(initial_layer), processor
        )

    def parse_args(self, experiment_name):
        args = argparse.ArgumentParser.parse_args(self)

        # used for path names
        args.framework_name = self.framework_name(args.framework)

        args.run_name = args.run_name or self.default_run_name(
            args.framework, experiment_name, args.initial_layer, args.processor
        )

        return args

    def add_arguments(self):
        self.add_skip_data_preparation_argument()
        self.add_skip_models_argument()
        self.add_skip_qsimov_argument()
        self.add_initial_layer_argument()
        self.add_epochs_argument()
        self.add_processor_argument()
        self.add_run_name_argument()
        self.add_framework_argument()

    def add_skip_data_preparation_argument(self):
        self.add_argument(
            "--skip-data-preparation",
            action="store_true",
            help="Skip data preparation step (preprocessing and splitting)."
            " Useful if you want to run the experiments multiple times",
        )

    def add_skip_models_argument(self):
        self.add_argument(
            "--skip-models",
            action="store_true",
            help="Skip generation of the models. This is useful if you"
            " want to run the experiments multiple times",
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
            help="Name of the run. If not set, the name will be generated"
            " automatically based on the initial layer and the processor",
            default=None,
        )


def parse_arguments():
    return MnistSpeedLossParser(description=EXPERIMENT_DESCRIPTION).parse_args(
        EXPERIMENT_NAME
    )


# copy file in script results to experiment results
def copy_to_experiment_results(
    args, src, dst=None, experiment_dir=EXPERIMENT_DIR
):
    # if dst is not set, use the same name as src
    if dst is None:
        dst = src

    # where the script results are saved
    script_results_dir = get_results_dir(args.framework, args.processor)

    # where the experiment results are saved
    experiment_results_dir = exp_mlflow.get_or_make_export_directory(
        experiment_dir
    )
    shutil.copy(
        os.path.join(script_results_dir, src),
        os.path.join(experiment_results_dir, dst),
    )


def prepare_data(args):
    script = os.path.join(EXPERIMENT_DIR, "preprocess_data.py")

    # Log the data generation script
    mlflow.log_param("data_generation_script", as_relative_path(script))

    if not args.skip_data_preparation:
        exp_mlflow.run_script(["python", script], EXPERIMENT_DIR)

    # Log the generated data files
    data_dir = get_qsimov_dataset_dir("mnist")
    mlflow.log_param("generated_data_dir", as_relative_path(data_dir))

    data_files = (
        ["test_x.npy", "test_y.npy"]
        + [f"train_x_{i}.npy" for i in range(4)]
        + [f"train_y_{i}.npy" for i in range(4)]
    )
    mlflow.log_param("generated_data_files", ", ".join(data_files))


def create_models(args):
    script = os.path.join(
        EXPERIMENT_DIR,
        args.framework_name,
        f"train_{args.framework}_models.py",
    )
    # Run the script
    if not args.skip_models:
        script_args = [
            "python",
            script,
            "--processor",
            args.processor,
            "--epochs",
            str(args.epochs),
        ]
        exp_mlflow.run_script(script_args, EXPERIMENT_DIR)

    # Log the used script
    mlflow.log_param(
        f"{args.framework}_models_script", as_relative_path(script)
    )

    framework_extension = "pt" if args.framework == "pytorch" else "tf"
    for is_model in (True, False):
        end = framework_extension if is_model else "history.pkl"
        files = []
        for loss in ("mse", "crossentropy"):
            for partial in ("half", "quarter"):
                files.extend(
                    [
                        # Model trained with partial of the data
                        f"{loss}_{partial}_model{end}",
                        # Model that is trained with the full data
                        f"{loss}_{partial}_full_model{end}",
                    ]
                )

        mlflow.log_param(
            "{}_{}".format(
                args.framework, "models" if is_model else "histories"
            ),
            ", ".join(files),
        )

    # Results directory for the experiment
    # Copy the file with the number of paths
    copy_to_experiment_results(args, "model_summary.txt")


def create_qsimov_models(args):
    script = os.path.join(
        EXPERIMENT_DIR, args.framework_name, "train_qsimov_models.py"
    )
    # Run the script
    if not args.skip_qsimov:
        script_args = [
            "python",
            script,
            "--initial-layer",
            str(args.initial_layer),
            "--processor",
            args.processor,
            "--epochs",
            str(args.epochs),
        ]
        exp_mlflow.run_script(script_args, EXPERIMENT_DIR)

    # Log the used script
    mlflow.log_param("qsimov_models_script", as_relative_path(script))

    # Log the generated histories
    files = []
    for loss in ("mse", "crossentropy"):
        for partial in ("half", "quarter"):
            files.append(
                "qsimov_gradient_{}_{}_history.pkl".format(loss, partial),
            )
            if loss == "mse":
                files.append(
                    "qsimov_linear_{}_{}_history.pkl".format(loss, partial),
                )
    mlflow.log_param("qsimov_histories", ", ".join(files))

    # Copy the file with the number of paths
    copy_to_experiment_results(args, "number_of_paths.txt")


def create_plots(args):
    script = os.path.join(EXPERIMENT_DIR, "speed_loss_plots.py")
    # Run the script
    exp_mlflow.run_script(
        [
            "python",
            script,
            "--processor",
            args.processor,
            "--framework",
            args.framework,
        ],
        EXPERIMENT_DIR,
    )

    # Log the used script
    mlflow.log_param("plots_script", as_relative_path(script))

    # Log the generated plots directory
    results_dir = get_results_dir(args.processor, args.framework)
    mlflow.log_param("generated_results_dir", as_relative_path(results_dir))

    files = []
    run_start_time = exp_mlflow.get_run_start_time()
    for loss, partial, val in product(
        ("mse", "crossentropy"), ("half", "quarter"), ("Train", "Test")
    ):
        file = f"{loss}_{partial}_gradient_comparison_{val}.html"

        # Add date time and run name to the file name
        new_file = "{}_{}_{}".format(run_start_time, args.run_name, file)
        # Move the file to the experiment results directory
        copy_to_experiment_results(args, file, new_file)

        files.append(new_file)
    mlflow.log_params({f"plot_{i}": file for i, file in enumerate(files)})


def run_experiment(args):
    # Prepare the data
    prepare_data(args)

    # Run the scripts that train the models
    create_models(args)

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
