"""MLflow orchestrator for the imagenet_continual_learning experiment.

Pipeline:
  1. prepare_data  — verify the ImageNet subset dataset exists (preprocess if needed).
  2. train_keras   — train path-selector and standard-baseline models on round 1.
  3. train_qsimov  — run all three Qsimov continual-learning variants across rounds.
  4. train_standard — run standard fine-tuning and cumulative-retrain baselines.
  5. plots         — generate overall-accuracy, forgetting-curves, and timing HTML plots.

Key results
-----------
  * overall_accuracy.html    — test accuracy per method after each round.
  * forgetting_curves.html   — per-round val accuracy, reveals catastrophic forgetting.
  * training_time.html       — cumulative wall-clock time per method.

Usage example
-------------
  python experiments/imagenet_continual_learning/main.py \\
      --processor gpu --epochs 10 --initial-layer -1
"""

import sys
import os
import argparse
import shutil
import experiments.git as exp_git
import experiments.mlflow as exp_mlflow
import mlflow
from experiments.path_utils import (
    as_relative_path,
    get_qsimov_dataset_dir,
    get_imagenet_continual_learning_results_dir as get_results_dir,
)
from experiments.imagenet_continual_learning.preprocess_data import N_ROUNDS

EXPERIMENT_DESCRIPTION = (
    "Continual learning on ImageNet subset (100 labels, all 4 train dirs). "
    "Compares QsimovLinearSystem (accumulative, no forgetting) against "
    "QsimovLinearSystem (reset per round), QsimovGradient, "
    "standard sequential fine-tuning, and a cumulative-retrain oracle."
)
EXPERIMENT_NAME = "imagenet_continual_learning"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))


###############################################################################
# Step helpers
###############################################################################


def prepare_data(args):
    script = os.path.join(
        os.path.dirname(EXPERIMENT_DIR),
        "imagenet_subset_by_splits",
        "preprocess_data.py",
    )
    mlflow.log_param("data_preparation_script", as_relative_path(script))

    if not args.skip_data_preparation:
        exp_mlflow.run_script(["python", script], EXPERIMENT_DIR)

    data_dir = get_qsimov_dataset_dir("imagenet_subset")
    mlflow.log_param("data_dir", as_relative_path(data_dir))


def train_initial_models(args):
    script = os.path.join(EXPERIMENT_DIR, "train_keras.py")
    mlflow.log_param("initial_models_script", as_relative_path(script))

    if not args.skip_initial_models:
        exp_mlflow.run_script(
            ["python", script, "--processor", args.processor, "--epochs", str(args.epochs)],
            EXPERIMENT_DIR,
        )

    results_dir = get_results_dir(args.processor)
    for tag in ("path_selector_softmax", "path_selector_linear", "standard"):
        mlflow.log_param(
            f"vgg16_{tag}_model",
            os.path.join(results_dir, f"vgg16_{tag}_model.tf"),
        )


def train_qsimov_models(args):
    script = os.path.join(EXPERIMENT_DIR, "train_keras_qsimov.py")
    mlflow.log_param("qsimov_script", as_relative_path(script))

    if not args.skip_qsimov:
        exp_mlflow.run_script(
            [
                "python",
                script,
                "--processor",
                args.processor,
                "--initial-layer",
                str(args.initial_layer),
            ],
            EXPERIMENT_DIR,
        )

    results_dir = get_results_dir(args.processor)
    for method in ("qsimov_linear_accum", "qsimov_linear_reset", "qsimov_gradient"):
        mlflow.log_param(
            f"{method}_results",
            os.path.join(results_dir, f"{method}_results.pkl"),
        )


def train_standard_models(args):
    script = os.path.join(EXPERIMENT_DIR, "train_keras_standard.py")
    mlflow.log_param("standard_script", as_relative_path(script))

    if not args.skip_standard:
        exp_mlflow.run_script(
            ["python", script, "--processor", args.processor],
            EXPERIMENT_DIR,
        )

    results_dir = get_results_dir(args.processor)
    for method in ("standard_finetune", "standard_cumulative"):
        mlflow.log_param(
            f"{method}_results",
            os.path.join(results_dir, f"{method}_results.pkl"),
        )


def create_plots(args):
    script = os.path.join(EXPERIMENT_DIR, "plot_continual_learning.py")
    mlflow.log_param("plots_script", as_relative_path(script))

    exp_mlflow.run_script(
        ["python", script, "--processor", args.processor],
        EXPERIMENT_DIR,
    )

    results_dir = get_results_dir(args.processor)
    export_dir = exp_mlflow.get_or_make_export_directory(EXPERIMENT_DIR)
    run_start = exp_mlflow.get_run_start_time()
    run_name = args.run_name

    for plot_file in (
        "overall_accuracy.html",
        "forgetting_curves.html",
        "training_time.html",
        "metrics_bwt_aa.html",
        "per_round_update_time.html",
    ):
        src = os.path.join(results_dir, plot_file)
        if os.path.exists(src):
            dst_name = f"{run_start}_{run_name}_{plot_file}"
            shutil.copy(src, os.path.join(export_dir, dst_name))
            mlflow.log_param(f"plot_{plot_file}", dst_name)

    n_paths_file = os.path.join(results_dir, "number_of_paths_vgg16.txt")
    if os.path.exists(n_paths_file):
        shutil.copy(n_paths_file, os.path.join(export_dir, "number_of_paths_vgg16.txt"))
        with open(n_paths_file) as f:
            mlflow.log_param("number_of_paths", f.read().strip())


###############################################################################
# Main
###############################################################################


def run_experiment(args):
    prepare_data(args)
    train_initial_models(args)
    train_qsimov_models(args)
    train_standard_models(args)
    create_plots(args)


def main():
    args = parse_arguments()
    exp_git.check_git_clean()

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("experiment_execution", " ".join(sys.argv))
        mlflow.log_param("n_rounds", N_ROUNDS)
        mlflow.log_param("initial_layer", args.initial_layer)

        run_experiment(args)

        exp_mlflow.export_active_run(EXPERIMENT_DIR)


###############################################################################
# CLI Parser
###############################################################################


class ContinualLearningParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(description=EXPERIMENT_DESCRIPTION, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="gpu")
        self.add_argument("--epochs", type=int, required=True,
                          help="Epochs for initial model training (round 1)")
        self.add_argument("--initial-layer", type=int, default=-1,
                          help="Qsimov path selector initial layer (default -1: last Dense)")
        self.add_argument("--run-name", default=None)
        self.add_argument("--skip-data-preparation", action="store_true")
        self.add_argument("--skip-initial-models", action="store_true")
        self.add_argument("--skip-qsimov", action="store_true")
        self.add_argument("--skip-standard", action="store_true")

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        if parsed.run_name is None:
            parsed.run_name = f"imagenet_cl_{parsed.processor}_layer{parsed.initial_layer}"
        return parsed


def parse_arguments():
    return ContinualLearningParser().parse_args()


if __name__ == "__main__":
    main()
