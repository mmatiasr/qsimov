"""MLflow orchestration for MNIST forgetting experiment.

Demonstrates catastrophic forgetting (standard fine-tuning) vs. no forgetting
(QsimovLinearSystem with equation accumulation) in class-incremental learning.

Phase 1: train base model on MNIST classes 0-4.
Phase 2: update with classes 5-9 using different methods.
Measure accuracy on old classes (0-4) and new classes (5-9).
"""

import sys
import argparse
import os
import shutil
import mlflow

import experiments.git as exp_git
import experiments.mlflow as exp_mlflow
from experiments.path_utils import (
    get_qsimov_experiments_dir,
    get_mnist_forgetting_results_dir as get_results_dir,
    as_relative_path,
)

EXPERIMENT_NAME = "mnist_forgetting"
EXPERIMENT_DESCRIPTION = (
    "Demonstrates catastrophic forgetting vs Qsimov on MNIST. "
    "Base model trained on classes 0-4 (phase 1). "
    "Phase 2 update with classes 5-9 using: QsimovLinearSystem (accumulated), "
    "QsimovGradient, standard fine-tuning, and cumulative oracle. "
    "Measures accuracy on old (0-4) and new (5-9) classes."
)
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))


class MnistForgettingParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_arguments()

    def add_arguments(self):
        self.add_argument("--processor", choices=["cpu", "gpu"], default="cpu")
        self.add_argument("--framework", choices=["keras", "pytorch"], default="keras")
        self.add_argument(
            "--base-epochs", type=int, default=10,
            help="Epochs to train base model on phase 1",
        )
        self.add_argument(
            "--epochs", type=int, default=5,
            help="Epochs for gradient, finetune, and cumulative methods",
        )
        self.add_argument("--skip-data-preparation", action="store_true")
        self.add_argument("--skip-base-model", action="store_true")
        self.add_argument("--skip-forgetting", action="store_true")
        self.add_argument("--run-name", default=None)


def prepare_data(args):
    script = os.path.join(EXPERIMENT_DIR, "preprocess_data.py")
    mlflow.log_param("data_preparation_script", as_relative_path(script))

    if not args.skip_data_preparation:
        exp_mlflow.run_script(["python", script], EXPERIMENT_DIR)


def train_base_model(args):
    if args.framework == "pytorch":
        script = os.path.join(EXPERIMENT_DIR, "pytorch_train_base_model.py")
    else:
        script = os.path.join(EXPERIMENT_DIR, "train_base_model.py")
    mlflow.log_param("base_model_script", as_relative_path(script))

    if not args.skip_base_model:
        exp_mlflow.run_script(
            ["python", script,
             "--processor", args.processor,
             "--epochs", str(args.base_epochs)],
            EXPERIMENT_DIR,
        )


def run_forgetting(args):
    if args.framework == "pytorch":
        script = os.path.join(EXPERIMENT_DIR, "pytorch_run_forgetting.py")
    else:
        script = os.path.join(EXPERIMENT_DIR, "run_forgetting.py")
    mlflow.log_param("forgetting_script", as_relative_path(script))

    if not args.skip_forgetting:
        exp_mlflow.run_script(
            ["python", script,
             "--processor", args.processor,
             "--epochs", str(args.epochs)],
            EXPERIMENT_DIR,
        )


def generate_plots(args, run_name):
    script = os.path.join(EXPERIMENT_DIR, "plot_forgetting.py")
    exp_mlflow.run_script(
        ["python", script,
         "--processor", args.processor,
         "--framework", args.framework],
        EXPERIMENT_DIR,
    )
    mlflow.log_param("plots_script", as_relative_path(script))

    results_dir = get_results_dir(args.framework, args.processor)
    export_dir = exp_mlflow.get_or_make_export_directory(EXPERIMENT_DIR)
    run_start = exp_mlflow.get_run_start_time()

    src = os.path.join(results_dir, "mnist_forgetting_plot.html")
    dst_name = f"{run_start}_{run_name}_mnist_forgetting_plot.html"
    if os.path.exists(src):
        shutil.copy(src, os.path.join(export_dir, dst_name))
        mlflow.log_param("plot_file", dst_name)


def main():
    parser = MnistForgettingParser(description=EXPERIMENT_DESCRIPTION)
    args = parser.parse_args()
    run_name = args.run_name or f"{args.framework}_{EXPERIMENT_NAME}_{args.processor}"

    exp_git.check_git_clean()

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("experiment_execution", " ".join(sys.argv))
        mlflow.log_param("framework", args.framework)
        mlflow.log_param("processor", args.processor)
        mlflow.log_param("base_epochs", args.base_epochs)
        mlflow.log_param("phase2_epochs", args.epochs)

        prepare_data(args)
        train_base_model(args)
        run_forgetting(args)
        generate_plots(args, run_name)

        exp_mlflow.export_active_run(EXPERIMENT_DIR)


if __name__ == "__main__":
    main()
