"""Generate interactive plots for the initial_layer sweep experiment.

Produces four HTML files:

n_paths_vs_initial_layer.html
    Bar chart of total number of paths vs. initial_layer value.
    Illustrates the exponential growth that makes deep initial_layer infeasible.

build_time_vs_initial_layer.html
    Wall-clock time to construct the PathSelector vs. initial_layer.

accuracy_vs_initial_layer.html
    Test accuracy for Linear and Gradient methods vs. initial_layer.
    Shows whether deeper φ_R actually improves predictive quality.
    Infeasible entries are omitted.

training_time_vs_initial_layer.html
    Training time for Linear and Gradient methods vs. initial_layer.
    Paired with accuracy to show the accuracy/cost trade-off.
"""

import os
import pickle
import argparse
import plotly.graph_objects as go
from experiments.path_utils import get_initial_layer_sweep_results_dir

PLOTLY_CONFIG = {"editable": True}
COLORS = {"linear": "#1f77b4", "gradient": "#ff7f0e"}


def load_results(results_dir):
    fpath = os.path.join(results_dir, "sweep_results.pkl")
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"No sweep results found at {fpath}. "
            "Run train_keras_sweep.py first."
        )
    with open(fpath, "rb") as f:
        return pickle.load(f)


def _x_labels(results):
    return [str(r["initial_layer"]) for r in results]


def plot_n_paths(results, results_dir):
    x = _x_labels(results)
    y = [r["n_paths"] for r in results]

    colors = [
        "#d62728" if (r.get("linear", {}).get("status") == "infeasible") else "#1f77b4"
        for r in results
    ]

    fig = go.Figure(go.Bar(
        x=x, y=y,
        marker_color=colors,
        text=[f"{v:,}" for v in y],
        textposition="outside",
    ))
    fig.update_layout(
        title="Number of Paths vs. initial_layer",
        xaxis_title="initial_layer",
        yaxis_title="Number of Paths",
        yaxis_type="log",
        annotations=[
            dict(
                x=xi, y=yi,
                text="infeasible" if r.get("linear", {}).get("status") == "infeasible" else "",
                showarrow=False, yshift=20, font=dict(color="#d62728"),
            )
            for xi, yi, r in zip(x, y, results)
        ],
    )
    out = os.path.join(results_dir, "n_paths_vs_initial_layer.html")
    fig.write_html(out, config=PLOTLY_CONFIG)
    print(f"Saved: {out}")


def plot_build_time(results, results_dir):
    x = _x_labels(results)
    y_linear = [r.get("build_time_linear", None) for r in results]
    y_gradient = [r.get("build_time_gradient", None) for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Linear PathSelector", x=x, y=y_linear,
        marker_color=COLORS["linear"],
    ))
    if any(v is not None for v in y_gradient):
        fig.add_trace(go.Bar(
            name="Gradient PathSelector", x=x, y=y_gradient,
            marker_color=COLORS["gradient"],
        ))

    fig.update_layout(
        title="PathSelector Build Time vs. initial_layer",
        xaxis_title="initial_layer",
        yaxis_title="Build Time (seconds)",
        barmode="group",
    )
    out = os.path.join(results_dir, "build_time_vs_initial_layer.html")
    fig.write_html(out, config=PLOTLY_CONFIG)
    print(f"Saved: {out}")


def plot_accuracy(results, results_dir):
    x_linear, y_linear = [], []
    x_gradient, y_gradient = [], []

    for r in results:
        label = str(r["initial_layer"])
        if r.get("linear", {}).get("status") == "ok":
            x_linear.append(label)
            y_linear.append(r["linear"]["test_accuracy"])
        if r.get("gradient", {}).get("status") == "ok":
            x_gradient.append(label)
            y_gradient.append(r["gradient"]["test_accuracy"])

    fig = go.Figure()
    if x_linear:
        fig.add_trace(go.Bar(
            name="Qsimov Linear", x=x_linear, y=y_linear,
            marker_color=COLORS["linear"],
        ))
    if x_gradient:
        fig.add_trace(go.Bar(
            name="Qsimov Gradient", x=x_gradient, y=y_gradient,
            marker_color=COLORS["gradient"],
        ))

    fig.update_layout(
        title="Test Accuracy vs. initial_layer",
        xaxis_title="initial_layer",
        yaxis_title="Top-1 Accuracy",
        yaxis=dict(tickformat=".2%"),
        barmode="group",
    )
    out = os.path.join(results_dir, "accuracy_vs_initial_layer.html")
    fig.write_html(out, config=PLOTLY_CONFIG)
    print(f"Saved: {out}")


def plot_training_time(results, results_dir):
    x_linear, y_linear = [], []
    x_gradient, y_gradient = [], []

    for r in results:
        label = str(r["initial_layer"])
        if r.get("linear", {}).get("status") == "ok":
            x_linear.append(label)
            y_linear.append(r["linear"]["train_time"])
        if r.get("gradient", {}).get("status") == "ok":
            x_gradient.append(label)
            y_gradient.append(r["gradient"]["train_time"])

    fig = go.Figure()
    if x_linear:
        fig.add_trace(go.Bar(
            name="Qsimov Linear", x=x_linear, y=y_linear,
            marker_color=COLORS["linear"],
        ))
    if x_gradient:
        fig.add_trace(go.Bar(
            name="Qsimov Gradient", x=x_gradient, y=y_gradient,
            marker_color=COLORS["gradient"],
        ))

    fig.update_layout(
        title="Training Time vs. initial_layer",
        xaxis_title="initial_layer",
        yaxis_title="Training Time (seconds)",
        barmode="group",
    )
    out = os.path.join(results_dir, "training_time_vs_initial_layer.html")
    fig.write_html(out, config=PLOTLY_CONFIG)
    print(f"Saved: {out}")


def main(args):
    results_dir = get_initial_layer_sweep_results_dir(args.processor)
    results = load_results(results_dir)

    plot_n_paths(results, results_dir)
    plot_build_time(results, results_dir)
    plot_accuracy(results, results_dir)
    plot_training_time(results, results_dir)


###############################################################################
# CLI Parser
###############################################################################


class PlotParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="gpu")


if __name__ == "__main__":
    main(PlotParser().parse_args())
