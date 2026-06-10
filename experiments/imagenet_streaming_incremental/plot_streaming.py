"""Generate interactive plots for the streaming incremental experiment.

Produces three HTML files:

accuracy_vs_batch.html
    Test accuracy on the cumulative seen-class test set after each batch.
    Shows whether each method retains knowledge of previously-seen classes.
    Methods that forget show a slower growth or even a dip compared to the
    qsimov_linear_accum no-forgetting baseline.

update_time_vs_batch.html
    Wall-clock seconds needed to process ONLY the current batch (not cumulative).
    The key plot for the "extremely fast re-training" claim:
      - qsimov_linear_accum: roughly constant (only processes new batch equations)
      - standard_cumulative: grows linearly (re-trains on all seen data)
      - qsimov_gradient / standard_finetune: roughly constant (fixed epochs)

accuracy_vs_cumulative_time.html
    Accuracy on seen classes vs. total elapsed time.
    Efficiency frontier: qsimov_linear_accum should achieve the best
    accuracy per unit of compute spent.
"""

import os
import pickle
import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from experiments.path_utils import get_imagenet_streaming_incremental_results_dir

METHOD_LABELS = {
    "qsimov_linear_accum": "Qsimov Linear (accum)",
    "qsimov_gradient":     "Qsimov Gradient",
    "standard_finetune":   "Standard Fine-tuning",
    "standard_cumulative": "Standard Cumulative (oracle)",
}
COLORS = {
    "qsimov_linear_accum": "#1f77b4",
    "qsimov_gradient":     "#ff7f0e",
    "standard_finetune":   "#d62728",
    "standard_cumulative": "#2ca02c",
}
PLOTLY_CONFIG = {"editable": True}


def load_results(results_dir):
    methods = list(METHOD_LABELS.keys())
    data = {}
    for method in methods:
        fpath = os.path.join(results_dir, f"{method}_results.pkl")
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                data[method] = pickle.load(f)
        else:
            print(f"  Warning: {fpath} not found — skipping {method}")
    return data


def _batch_series(data, method, key):
    """Extract a list field from batches list."""
    return [b[key] for b in data[method]["batches"]]


def plot_accuracy_vs_batch(data, results_dir):
    fig = go.Figure()

    for method, label in METHOD_LABELS.items():
        if method not in data:
            continue
        batches = _batch_series(data, method, "batch")
        acc = _batch_series(data, method, "accuracy_on_seen")

        fig.add_trace(go.Scatter(
            x=batches, y=acc,
            mode="lines+markers",
            name=label,
            line=dict(color=COLORS[method], width=2),
            marker=dict(size=6),
        ))

    fig.update_layout(
        title="Accuracy on Cumulative Seen-Class Test Set vs. Streaming Batch",
        xaxis_title="Streaming Batch Number",
        yaxis_title="Top-1 Accuracy",
        yaxis=dict(tickformat=".2%"),
        legend=dict(x=0.01, y=0.01),
        hovermode="x unified",
    )
    out = os.path.join(results_dir, "accuracy_vs_batch.html")
    fig.write_html(out, config=PLOTLY_CONFIG)
    print(f"Saved: {out}")


def plot_update_time_vs_batch(data, results_dir):
    fig = go.Figure()

    for method, label in METHOD_LABELS.items():
        if method not in data:
            continue
        batches = _batch_series(data, method, "batch")
        times = _batch_series(data, method, "update_time")

        fig.add_trace(go.Scatter(
            x=batches, y=times,
            mode="lines+markers",
            name=label,
            line=dict(color=COLORS[method], width=2),
            marker=dict(size=6),
        ))

    fig.update_layout(
        title="Update Time per Batch (not cumulative)",
        xaxis_title="Streaming Batch Number",
        yaxis_title="Update Time (seconds)",
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
    )
    out = os.path.join(results_dir, "update_time_vs_batch.html")
    fig.write_html(out, config=PLOTLY_CONFIG)
    print(f"Saved: {out}")


def plot_accuracy_vs_cumulative_time(data, results_dir):
    fig = go.Figure()

    for method, label in METHOD_LABELS.items():
        if method not in data:
            continue
        times = _batch_series(data, method, "update_time")
        acc = _batch_series(data, method, "accuracy_on_seen")
        batches = _batch_series(data, method, "batch")
        cum_times = np.cumsum(times).tolist()

        fig.add_trace(go.Scatter(
            x=cum_times, y=acc,
            mode="lines+markers",
            name=label,
            line=dict(color=COLORS[method], width=2),
            marker=dict(size=6),
            text=[f"batch {b}" for b in batches],
            hovertemplate="%{text}<br>cumulative time: %{x:.0f}s<br>acc: %{y:.2%}",
        ))

    fig.update_layout(
        title="Accuracy vs. Total Training Time (Efficiency Frontier)",
        xaxis_title="Cumulative Update Time (seconds)",
        yaxis_title="Top-1 Accuracy on Seen Classes",
        yaxis=dict(tickformat=".2%"),
        legend=dict(x=0.01, y=0.01),
    )
    out = os.path.join(results_dir, "accuracy_vs_cumulative_time.html")
    fig.write_html(out, config=PLOTLY_CONFIG)
    print(f"Saved: {out}")


def main(args):
    framework = getattr(args, "framework", "keras")
    results_dir = get_imagenet_streaming_incremental_results_dir(args.processor, framework=framework)
    data = load_results(results_dir)

    if not data:
        print("No result files found. Run train_keras_streaming.py first.")
        return

    plot_accuracy_vs_batch(data, results_dir)
    plot_update_time_vs_batch(data, results_dir)
    plot_accuracy_vs_cumulative_time(data, results_dir)


###############################################################################
# CLI Parser
###############################################################################


class PlotParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="gpu")
        self.add_argument("--framework", choices=["keras", "pytorch"], default="keras")


if __name__ == "__main__":
    main(PlotParser().parse_args())
