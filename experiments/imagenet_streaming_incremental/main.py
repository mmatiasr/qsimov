"""MLflow orchestrator for the imagenet_streaming_incremental experiment.

Pipeline:
  1. train   — run all 4 streaming methods on the class-incremental data stream.
  2. plots   — generate accuracy, update-time, and efficiency HTML plots.

Key results
-----------
  * accuracy_vs_batch.html          — cumulative seen-class accuracy per batch.
  * update_time_vs_batch.html       — per-batch update time (constant vs. growing).
  * accuracy_vs_cumulative_time.html — accuracy vs. total compute (efficiency frontier).

This experiment directly demonstrates the README's headline claim:
  "extremely fast re-training since incremental re-training does not suffer
   from catastrophic forgetting."

  - qsimov_linear_accum: constant update cost AND no forgetting.
  - standard_finetune:   constant cost BUT forgets old classes.
  - standard_cumulative: no forgetting BUT cost grows linearly.

Prerequisites
-------------
  The imagenet_continual_learning experiment must be run first to produce
  the pre-trained base models:

    python experiments/imagenet_continual_learning/main.py \\
        --epochs 20 --processor gpu --skip-qsimov --skip-standard

Usage example
-------------
  python experiments/imagenet_streaming_incremental/main.py \\
      --processor gpu --run-name streaming_v1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import subprocess
import argparse
import shutil
import experiments.git as exp_git
import experiments.mlflow as exp_mlflow
import mlflow
from experiments.path_utils import (
    as_relative_path,
    get_imagenet_streaming_incremental_results_dir as get_results_dir,
    get_imagenet_continual_learning_results_dir,
)
from experiments.imagenet_streaming_incremental.preprocess_data import (
    N_BATCHES,
    CLASSES_PER_BATCH,
)

EXPERIMENT_DESCRIPTION = (
    "Class-incremental streaming experiment on ImageNet subset. "
    "Compares QsimovLinearSystem (accumulate), QsimovGradient, standard fine-tuning "
    "and cumulative retraining across 20 sequential class batches. "
    "Demonstrates fast constant-time updates with no catastrophic forgetting."
)
EXPERIMENT_NAME = "imagenet_streaming_incremental"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))


def train_all_methods(args):
    if args.framework == "pytorch":
        if not args.skip_training:
            script = os.path.join(EXPERIMENT_DIR, "train_pytorch_streaming.py")
            subprocess.run(
                ["python3", script, "--processor", args.processor],
                check=True,
            )
        return

    script = os.path.join(EXPERIMENT_DIR, "train_keras_streaming.py")
    mlflow.log_param("train_script", as_relative_path(script))

    if not args.skip_training:
        exp_mlflow.run_script(
            ["python", script, "--processor", args.processor],
            EXPERIMENT_DIR,
        )

    results_dir = get_results_dir(args.processor)
    for method in ("qsimov_linear_accum", "qsimov_gradient",
                   "standard_finetune", "standard_cumulative"):
        mlflow.log_param(
            f"{method}_results",
            os.path.join(results_dir, f"{method}_results.pkl"),
        )


def create_plots(args):
    script = os.path.join(EXPERIMENT_DIR, "plot_streaming.py")
    plot_args = ["--processor", args.processor]
    if args.framework == "pytorch":
        plot_args += ["--framework", "pytorch"]
        subprocess.run(["python3", script] + plot_args, check=True)
        return

    mlflow.log_param("plots_script", as_relative_path(script))
    exp_mlflow.run_script(["python", script] + plot_args, EXPERIMENT_DIR)

    results_dir = get_results_dir(args.processor)
    export_dir = exp_mlflow.get_or_make_export_directory(EXPERIMENT_DIR)
    run_start = exp_mlflow.get_run_start_time()
    run_name = args.run_name

    for plot_file in (
        "accuracy_vs_batch.html",
        "update_time_vs_batch.html",
        "accuracy_vs_cumulative_time.html",
    ):
        src = os.path.join(results_dir, plot_file)
        if os.path.exists(src):
            dst_name = f"{run_start}_{run_name}_{plot_file}"
            shutil.copy(src, os.path.join(export_dir, dst_name))
            mlflow.log_param(f"plot_{plot_file}", dst_name)


def run_experiment(args):
    train_all_methods(args)
    create_plots(args)


def main():
    args = parse_arguments()
    exp_git.check_git_clean()

    if args.framework == "pytorch":
        run_experiment(args)
        return

    cl_results_dir = get_imagenet_continual_learning_results_dir(args.processor)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("experiment_execution", " ".join(sys.argv))
        mlflow.log_param("n_batches", N_BATCHES)
        mlflow.log_param("classes_per_batch", CLASSES_PER_BATCH)
        mlflow.log_param("continual_learning_results_dir", cl_results_dir)

        run_experiment(args)

        exp_mlflow.export_active_run(EXPERIMENT_DIR)


###############################################################################
# CLI Parser
###############################################################################


class StreamingParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(description=EXPERIMENT_DESCRIPTION, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="gpu")
        self.add_argument("--framework", choices=["keras", "pytorch"], default="keras")
        self.add_argument("--run-name", default=None)
        self.add_argument("--skip-training", action="store_true")

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        if parsed.run_name is None:
            parsed.run_name = f"streaming_{parsed.framework}_{parsed.processor}"
        return parsed


def parse_arguments():
    return StreamingParser().parse_args()


if __name__ == "__main__":
    main()
