"""MLflow orchestration for CIFAR-10 forgetting experiment.

Replicates the MNIST forgetting experiment on a harder dataset (32×32 RGB,
natural images). Demonstrates that Qsimov's no-forgetting property holds
beyond the simple MNIST benchmark.
"""

import sys
import argparse
import os
import shutil
import mlflow

import experiments.git as exp_git
import experiments.mlflow as exp_mlflow
from experiments.path_utils import (
    get_cifar10_forgetting_results_dir as get_results_dir,
    as_relative_path,
)

EXPERIMENT_NAME = "cifar10_forgetting"
EXPERIMENT_DESCRIPTION = (
    "Replicates the MNIST forgetting experiment on CIFAR-10. "
    "Base LeNet trained on classes 0-4 (airplane, auto, bird, cat, deer). "
    "Phase 2 update with classes 5-9 (dog, frog, horse, ship, truck) "
    "using: QsimovLinearSystem (accumulated), QsimovGradient, standard "
    "fine-tuning, and cumulative oracle."
)
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))


class Cifar10ForgettingParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_arguments()

    def add_arguments(self):
        self.add_argument("--processor", choices=["cpu", "gpu"], default="cpu")
        self.add_argument("--base-epochs", type=int, default=20)
        self.add_argument("--epochs", type=int, default=5)
        self.add_argument("--skip-data-preparation", action="store_true")
        self.add_argument("--skip-base-model", action="store_true")
        self.add_argument("--skip-forgetting", action="store_true")
        self.add_argument("--run-name", default=None)


def prepare_data(args):
    script = os.path.join(EXPERIMENT_DIR, "preprocess_data.py")
    mlflow.log_param("data_script", as_relative_path(script))
    if not args.skip_data_preparation:
        exp_mlflow.run_script(["python", script], EXPERIMENT_DIR)


def train_base_model(args):
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
        ["python", script, "--processor", args.processor],
        EXPERIMENT_DIR,
    )
    mlflow.log_param("plots_script", as_relative_path(script))

    results_dir = get_results_dir("keras", args.processor)
    export_dir  = exp_mlflow.get_or_make_export_directory(EXPERIMENT_DIR)
    run_start   = exp_mlflow.get_run_start_time()

    src = os.path.join(results_dir, "cifar10_forgetting_plot.html")
    dst = os.path.join(export_dir, f"{run_start}_{run_name}_cifar10_forgetting_plot.html")
    if os.path.exists(src):
        shutil.copy(src, dst)
        mlflow.log_param("plot_file", dst)


def main():
    parser = Cifar10ForgettingParser(description=EXPERIMENT_DESCRIPTION)
    args = parser.parse_args()
    run_name = args.run_name or f"keras_{EXPERIMENT_NAME}_{args.processor}"

    exp_git.check_git_clean()

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("experiment_execution", " ".join(sys.argv))
        mlflow.log_param("processor", args.processor)
        mlflow.log_param("base_epochs", args.base_epochs)
        mlflow.log_param("phase2_epochs", args.epochs)
        mlflow.log_param("initial_layer", -1)
        mlflow.log_param("architecture", "LeNet (MSE, linear output)")

        prepare_data(args)
        train_base_model(args)
        run_forgetting(args)
        generate_plots(args, run_name)

        exp_mlflow.export_active_run(EXPERIMENT_DIR)


if __name__ == "__main__":
    main()
