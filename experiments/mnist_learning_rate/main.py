from itertools import product
import experiments.git as exp_git
import experiments.mlflow as exp_mlflow
import mlflow
import sys
import os
from experiments.path_utils import (
    get_mnist_learning_rate_results_dir,
    as_relative_path,
    get_qsimov_experiments_dir,
)
from experiments.mnist_speed_loss.main import (
    prepare_data,
    MnistSpeedLossParser,
    copy_to_experiment_results as copy_to_experiment_results_speed_loss,
)


EXPERIMENT_DESCRIPTION = (
    "Runs classification experiments on a dataset, applying"
    " pytorch/keras models and qsimov models, varying the learning rate."
    " Compares qsimov trained with path selector built with half of the"
    " dataset and, a pytorch/keras model trained with the whole dataset."
)
EXPERIMENT_NAME = "mnist_learning_rate"

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = get_qsimov_experiments_dir()


class MnistLearningRateParser(MnistSpeedLossParser):
    def add_arguments(self):
        self.add_skip_data_preparation_argument()
        self.add_skip_base_models_argument()
        self.add_skip_models_argument()
        self.add_framework_argument()
        self.add_processor_argument()
        self.add_run_name_argument()
        self.add_epochs_argument()
        self.add_initial_layer_argument()

    def add_skip_base_models_argument(self):
        self.add_argument(
            "--skip-base-models",
            action="store_true",
            help="Skip generation of the base models. This is useful if you"
            " run the experiments multiple times, or have already run the"
            " corresponding speed-loss experiments.",
        )


def parse_arguments():
    return MnistLearningRateParser().parse_args(EXPERIMENT_NAME)


# use the same function as in mnist_speed_loss changing experiment dir
def copy_to_experiment_results(*args, **kwargs):
    copy_to_experiment_results_speed_loss(
        *args, **kwargs, experiment_dir=EXPERIMENT_DIR
    )


# run the script of mnist_speed_loss that generates the base models, one
# of them is used for the qsimov model
def create_base_models(args):
    script = os.path.join(
        EXPERIMENTS_DIR,
        "mnist_speed_loss",
        args.framework_name,
        f"train_{args.framework}_models.py",
    )
    # Run the script
    if not args.skip_base_models:
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
        for loss, partial in product(
            ("mse", "crossentropy"), ("half", "quarter")
        ):
            files.extend(
                [
                    # Model trained with partial of the data
                    f"{loss}_{partial}_model_{end}",
                    # Model that is trained with the full data
                    f"{loss}_{partial}_full_model_{end}",
                ]
            )

        mlflow.log_param(
            "{}_{}".format(
                args.framework, "models" if is_model else "histories"
            ),
            ", ".join(files),
        )

    # copy the model summary to the experiment results
    copy_to_experiment_results(args, "model_summary.txt")


def create_models(args):
    script = os.path.join(
        EXPERIMENT_DIR, f"train_{args.framework}_qsimov_models.py"
    )
    # Run the script
    if not args.skip_models:
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
    mlflow.log_param("compute_metrics_script", as_relative_path(script))

    # Log the generated models
    generated_files = [
        f"learning_rate_{args.framework}_partial_stats.pkl",
        f"learning_rate_{args.framework}_stats.pkl",
        "learning_rate_qsimov_stats.pkl",
    ]
    mlflow.log_param("generated_stats", ", ".join(generated_files))

    # copy the number of paths to the experiment results
    copy_to_experiment_results(args, "number_of_paths.txt")


def create_plots(args):
    script = os.path.join(EXPERIMENT_DIR, "learning_rate_plots.py")
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

    # Log the generated plots
    results_dir = get_mnist_learning_rate_results_dir(
        args.framework, args.processor
    )
    mlflow.log_param("generated_results_dir", as_relative_path(results_dir))

    generated_files = [
        "learning_rate_plot_by_epoch_test.html",
        "learning_rate_plot_by_epoch_train.html",
        "learning_rate_plot_by_learning_rate.html",
    ]
    files = []
    run_start_time = exp_mlflow.get_run_start_time()

    for file in generated_files:
        # new name for the file
        new_file = "{}_{}_{}".format(run_start_time, args.run_name, file)

        # copy the file to the experiment results
        copy_to_experiment_results(args, src=file, dst=new_file)
        files.append(new_file)

    mlflow.log_param("generated_plots", ", ".join(files))


def run_experiment(args):
    # Prepare the data
    prepare_data(args)

    # Run the training scripts to generate base models
    create_base_models(args)

    # Run scripts that use base models to retrain with different learning rates
    create_models(args)

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
        exp_mlflow.export_active_run(
            os.path.dirname(os.path.abspath(__file__))
        )


if __name__ == "__main__":
    main()
