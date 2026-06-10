"""MLflow orchestrator for the initial_layer_sweep experiment.

Pipeline:
  1. sweep  — run linear and gradient methods for each initial_layer value.
  2. plots  — generate n_paths, build-time, accuracy, and training-time plots.

Key results
-----------
  * n_paths_vs_initial_layer.html       — path explosion with depth.
  * build_time_vs_initial_layer.html    — PathSelector construction cost.
  * accuracy_vs_initial_layer.html      — accuracy trade-off per depth.
  * training_time_vs_initial_layer.html — training cost per depth.

This experiment demonstrates the scalability limits of Qsimov:
the number of paths grows exponentially with the number of layers in φ_R,
so there is a practical upper bound on useful initial_layer depth.

Prerequisites
-------------
  The imagenet_continual_learning experiment must be run first:

    python experiments/imagenet_continual_learning/main.py \\
        --epochs 20 --processor gpu --skip-qsimov --skip-standard

Usage example
-------------
  python experiments/initial_layer_sweep/main.py \\
      --processor gpu --run-name sweep_v1
"""

import sys
import os
import subprocess
import argparse
import shutil
import experiments.git as exp_git
import experiments.mlflow as exp_mlflow
import mlflow
from experiments.path_utils import (
    as_relative_path,
    get_initial_layer_sweep_results_dir as get_results_dir,
    get_imagenet_continual_learning_results_dir,
)
from experiments.initial_layer_sweep.train_keras_sweep import INITIAL_LAYERS, MAX_PATHS

EXPERIMENT_DESCRIPTION = (
    "Sweep of initial_layer parameter for Qsimov PathSelector on ImageNet. "
    "Measures n_paths, build time, accuracy, and training time for "
    f"initial_layer ∈ {INITIAL_LAYERS}. "
    "Documents the exponential path growth that limits φ_R depth in practice."
)
EXPERIMENT_NAME = "initial_layer_sweep"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_sweep(args):
    if args.framework == "pytorch":
        if not args.skip_sweep:
            script = os.path.join(EXPERIMENT_DIR, "train_pytorch_sweep.py")
            subprocess.run(
                ["python3", script, "--processor", args.processor],
                check=True,
            )
        return

    script = os.path.join(EXPERIMENT_DIR, "train_keras_sweep.py")
    mlflow.log_param("sweep_script", as_relative_path(script))

    if not args.skip_sweep:
        exp_mlflow.run_script(
            ["python", script, "--processor", args.processor],
            EXPERIMENT_DIR,
        )

    results_dir = get_results_dir(args.processor)
    mlflow.log_param("sweep_results", os.path.join(results_dir, "sweep_results.pkl"))


def create_plots(args):
    script = os.path.join(EXPERIMENT_DIR, "plot_sweep.py")
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
        "n_paths_vs_initial_layer.html",
        "build_time_vs_initial_layer.html",
        "accuracy_vs_initial_layer.html",
        "training_time_vs_initial_layer.html",
    ):
        src = os.path.join(results_dir, plot_file)
        if os.path.exists(src):
            dst_name = f"{run_start}_{run_name}_{plot_file}"
            shutil.copy(src, os.path.join(export_dir, dst_name))
            mlflow.log_param(f"plot_{plot_file}", dst_name)


def run_experiment(args):
    run_sweep(args)
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
        mlflow.log_param("initial_layers_tested", str(INITIAL_LAYERS))
        mlflow.log_param("max_paths_threshold", MAX_PATHS)
        mlflow.log_param("continual_learning_results_dir", cl_results_dir)

        run_experiment(args)

        exp_mlflow.export_active_run(EXPERIMENT_DIR)


###############################################################################
# CLI Parser
###############################################################################


class SweepMainParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(description=EXPERIMENT_DESCRIPTION, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="gpu")
        self.add_argument("--framework", choices=["keras", "pytorch"], default="keras")
        self.add_argument("--run-name", default=None)
        self.add_argument("--skip-sweep", action="store_true")

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        if parsed.run_name is None:
            parsed.run_name = f"sweep_{parsed.framework}_{parsed.processor}"
        return parsed


def parse_arguments():
    return SweepMainParser().parse_args()


if __name__ == "__main__":
    main()
