import shutil
import subprocess
import sys
import mlflow
import os
import json
from datetime import datetime
import experiments.git as git_exp


def get_run_start_time():
    """Get the start time of the run.

    Returns
    -------
    datetime.datetime
        The start time of the run.
    """
    run = mlflow.active_run()
    return datetime.fromtimestamp(run.info.start_time / 1000).strftime(
        "%Y%m%d_%H%M%S"
    )


def get_or_make_export_directory(directory):
    """Create a directory for the experiment.

    Parameters
    ----------
    directory : str
        Path to the directory of the experiment.

    Returns
    -------
    str
        Path to the directory of the experiment results.
    """
    active_run = mlflow.active_run()

    export_path = os.path.join(
        directory,
        "run_exports",
        "{}_{}".format(get_run_start_time(), active_run.info.run_name),
    )
    os.makedirs(export_path, exist_ok=True)
    return export_path


def export_active_run(directory):
    """Export the active mlflow run to a JSON file.

    Parameters
    ----------
    directory : str
        Path to the directory of the experiment.
    """
    active_run = mlflow.active_run()

    # Generate using date, time and run name
    export_path = os.path.join(
        get_or_make_export_directory(directory),
        "report.json",
    )
    run_export = mlflow.tracking.MlflowClient().get_run(active_run.info.run_id)
    run_export = run_export.to_dictionary()

    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    with open(os.path.join(export_path), "w") as f:
        run_export["info"]["user_id"] = git_exp.get_git_username()
        run_export["info"]["git_commit"] = run_export["data"]["tags"][
            "mlflow.source.git.commit"
        ]
        json.dump(run_export, f, indent=4)

    return export_path


def run_script(command_as_list, experiment_dir):
    if command_as_list[0] == "python":
        command_as_list = [sys.executable] + command_as_list[1:]
    print("\n\nRunning {}\n\n".format(" ".join(command_as_list)), flush=True)
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(command_as_list, env=env)
    if result.returncode != 0:
        # delete the experiment directory
        experiment_results_dir = get_or_make_export_directory(experiment_dir)
        shutil.rmtree(experiment_results_dir, ignore_errors=True)
        raise RuntimeError(
            f"Error while running {' '.join(command_as_list)}."
            f" Return code: {result.returncode}"
        )
