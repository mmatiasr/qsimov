"""Generate plots for the continual learning experiment.

Produces five HTML interactive figures (Plotly):

  1. overall_accuracy.html  — test-set accuracy after each training round.
  2. forgetting_curves.html — per-round val accuracy revealing catastrophic
     forgetting (accuracy drops after a method trains on new rounds).
  3. training_time.html     — cumulative wall-clock time per method.
  4. metrics_bwt_aa.html    — Backward Transfer (BWT) and Average Accuracy (AA)
     after each round, the standard continual-learning evaluation metrics.
  5. per_round_update_time.html — wall-clock seconds for EACH round's update
     (not cumulative). Shows whether update cost grows with rounds.

Metrics definitions
-------------------
Average Accuracy (AA) after round T
    Mean val accuracy across all rounds seen so far:
      AA(T) = (1/T) * Σ_{i=1}^{T} acc(i, T)
    where acc(i, T) is the val accuracy on round i's held-out set,
    evaluated AFTER training on round T.

Backward Transfer (BWT) after round T (T > 1)
    Mean decrease in accuracy on previously-seen rounds:
      BWT(T) = (1/(T-1)) * Σ_{i=1}^{T-1} [acc(i, T) - acc(i, i)]
    A value near 0 means no forgetting; negative means catastrophic forgetting.
    qsimov_linear_accum is expected to have BWT ≈ 0.
"""

import os
import pickle
import argparse
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from experiments.path_utils import get_imagenet_continual_learning_results_dir
from experiments.imagenet_continual_learning.preprocess_data import N_ROUNDS

ALL_METHODS = [
    "qsimov_linear_accum",
    "qsimov_linear_reset",
    "qsimov_gradient",
    "standard_finetune",
    "standard_cumulative",
]

METHOD_LABELS = {
    "qsimov_linear_accum": "Qsimov linear (accumulative — no forgetting)",
    "qsimov_linear_reset": "Qsimov linear (reset per round)",
    "qsimov_gradient": "Qsimov gradient (no replay)",
    "standard_finetune": "Standard fine-tuning (sequential)",
    "standard_cumulative": "Standard cumulative (oracle upper bound)",
}

METHOD_COLORS = {
    "qsimov_linear_accum": "green",
    "qsimov_linear_reset": "orange",
    "qsimov_gradient": "blue",
    "standard_finetune": "red",
    "standard_cumulative": "black",
}

METHOD_DASH = {
    "qsimov_linear_accum": None,
    "qsimov_linear_reset": "dash",
    "qsimov_gradient": "dot",
    "standard_finetune": "dashdot",
    "standard_cumulative": "longdash",
}

AXES_STYLE = dict(
    showgrid=False,
    linecolor="black",
    linewidth=1.5,
    ticks="inside",
    tickwidth=2,
)

ROUND_LABELS = [f"Round {k}" for k in range(1, N_ROUNDS + 1)]


def load_results(results_dir):
    results = {}
    for method in ALL_METHODS:
        path = os.path.join(results_dir, f"{method}_results.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                results[method] = pickle.load(f)
        else:
            print(f"Warning: results not found for {method} at {path}")
    return results


###############################################################################
# Plot 1: overall accuracy by round
###############################################################################


def make_overall_accuracy_plot(results, results_dir):
    fig = go.Figure()

    for method, data in results.items():
        x_rounds, y_acc, y_time = [], [], []
        for k in range(1, N_ROUNDS + 1):
            key = f"after_round_{k}"
            if key not in data:
                continue
            x_rounds.append(k)
            y_acc.append(data[key]["overall"]["accuracy"])
            y_time.append(data[key]["time(s)"])

        fig.add_trace(
            go.Scatter(
                x=x_rounds,
                y=y_acc,
                name=METHOD_LABELS[method],
                line=dict(color=METHOD_COLORS[method], dash=METHOD_DASH[method]),
                mode="lines+markers",
            )
        )

    fig.update_layout(
        height=600,
        width=900,
        title_text="Overall test-set accuracy after each training round",
        title_x=0.5,
        xaxis_title="Training round",
        yaxis_title="Test accuracy",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(1, N_ROUNDS + 1)),
            ticktext=ROUND_LABELS,
            **AXES_STYLE,
        ),
        yaxis=AXES_STYLE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    path = os.path.join(results_dir, "overall_accuracy.html")
    fig.write_html(path, config={"editable": True})
    print(f"Saved: {path}")
    return fig


###############################################################################
# Plot 2: forgetting curves (per-round validation accuracy)
###############################################################################


def make_forgetting_curves_plot(results, results_dir):
    """One subplot per past round, showing accuracy on that round's val set."""
    fig = make_subplots(
        rows=1,
        cols=N_ROUNDS,
        subplot_titles=[f"Val accuracy on {lbl}" for lbl in ROUND_LABELS],
        shared_yaxes=True,
    )

    for col_idx, val_round in enumerate(range(1, N_ROUNDS + 1), 1):
        for method, data in results.items():
            x_rounds, y_acc = [], []
            for train_round in range(val_round, N_ROUNDS + 1):
                key = f"after_round_{train_round}"
                if key not in data:
                    continue
                val_key = f"round_{val_round}"
                if val_key not in data[key].get("per_round_val", {}):
                    continue
                x_rounds.append(train_round)
                y_acc.append(data[key]["per_round_val"][val_key]["accuracy"])

            fig.add_trace(
                go.Scatter(
                    x=x_rounds,
                    y=y_acc,
                    name=METHOD_LABELS[method],
                    legendgroup=METHOD_LABELS[method],
                    showlegend=(col_idx == 1),
                    line=dict(
                        color=METHOD_COLORS[method], dash=METHOD_DASH[method]
                    ),
                    mode="lines+markers",
                ),
                row=1,
                col=col_idx,
            )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, N_ROUNDS + 1)),
        ticktext=ROUND_LABELS,
        title_text="Training round",
        **AXES_STYLE,
    )
    fig.update_yaxes(title_text="Val accuracy", col=1, **AXES_STYLE)
    fig.update_yaxes(**AXES_STYLE)

    fig.update_layout(
        height=500,
        width=350 * N_ROUNDS,
        title_text=(
            "Forgetting curves: accuracy on each round's held-out val set "
            "as training continues"
        ),
        title_x=0.5,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    path = os.path.join(results_dir, "forgetting_curves.html")
    fig.write_html(path, config={"editable": True})
    print(f"Saved: {path}")
    return fig


###############################################################################
# Plot 3: training time by round
###############################################################################


def make_time_plot(results, results_dir):
    fig = go.Figure()

    for method, data in results.items():
        x_rounds, y_time = [], []
        for k in range(1, N_ROUNDS + 1):
            key = f"after_round_{k}"
            if key not in data:
                continue
            x_rounds.append(k)
            y_time.append(data[key]["time(s)"])

        fig.add_trace(
            go.Scatter(
                x=x_rounds,
                y=y_time,
                name=METHOD_LABELS[method],
                line=dict(color=METHOD_COLORS[method], dash=METHOD_DASH[method]),
                mode="lines+markers",
            )
        )

    fig.update_layout(
        height=600,
        width=900,
        title_text="Cumulative training time after each round",
        title_x=0.5,
        xaxis_title="Training round",
        yaxis_title="Cumulative time (s)",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(1, N_ROUNDS + 1)),
            ticktext=ROUND_LABELS,
            **AXES_STYLE,
        ),
        yaxis=AXES_STYLE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    path = os.path.join(results_dir, "training_time.html")
    fig.write_html(path, config={"editable": True})
    print(f"Saved: {path}")
    return fig


###############################################################################
# Metrics: Average Accuracy (AA) and Backward Transfer (BWT)
###############################################################################


def _round_val_acc(data, train_round, val_round):
    """Return accuracy on val_round's held-out set after training through train_round."""
    key = f"after_round_{train_round}"
    val_key = f"round_{val_round}"
    try:
        return data[key]["per_round_val"][val_key]["accuracy"]
    except KeyError:
        return None


def compute_aa_bwt(data, n_rounds):
    """Compute AA and BWT for each round T.

    Returns
    -------
    aa : list of (round, aa_value) pairs
    bwt : list of (round, bwt_value) pairs (starts from round 2)
    """
    aa_values, bwt_values = [], []

    for T in range(1, n_rounds + 1):
        # Average Accuracy: mean accuracy across all val rounds seen so far
        accs = [_round_val_acc(data, T, i) for i in range(1, T + 1)]
        accs = [a for a in accs if a is not None]
        if accs:
            aa_values.append((T, float(np.mean(accs))))

        # BWT: only defined from round 2 onwards
        if T > 1:
            diffs = []
            for i in range(1, T):
                acc_iT = _round_val_acc(data, T, i)  # acc on round i after training on T
                acc_ii = _round_val_acc(data, i, i)  # acc on round i right after training on i
                if acc_iT is not None and acc_ii is not None:
                    diffs.append(acc_iT - acc_ii)
            if diffs:
                bwt_values.append((T, float(np.mean(diffs))))

    return aa_values, bwt_values


def make_bwt_aa_plot(results, results_dir):
    """Plot AA and BWT metrics after each round for every method."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Average Accuracy (AA)", "Backward Transfer (BWT)"],
    )

    for method, data in results.items():
        aa_values, bwt_values = compute_aa_bwt(data, N_ROUNDS)

        if aa_values:
            x_aa, y_aa = zip(*aa_values)
            fig.add_trace(
                go.Scatter(
                    x=list(x_aa), y=list(y_aa),
                    name=METHOD_LABELS[method],
                    legendgroup=METHOD_LABELS[method],
                    showlegend=True,
                    line=dict(color=METHOD_COLORS[method], dash=METHOD_DASH[method]),
                    mode="lines+markers",
                ),
                row=1, col=1,
            )

        if bwt_values:
            x_bwt, y_bwt = zip(*bwt_values)
            fig.add_trace(
                go.Scatter(
                    x=list(x_bwt), y=list(y_bwt),
                    name=METHOD_LABELS[method],
                    legendgroup=METHOD_LABELS[method],
                    showlegend=False,
                    line=dict(color=METHOD_COLORS[method], dash=METHOD_DASH[method]),
                    mode="lines+markers",
                ),
                row=1, col=2,
            )

    # Add BWT=0 reference line (no forgetting)
    fig.add_hline(y=0, row=1, col=2, line=dict(color="gray", dash="dot", width=1))

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, N_ROUNDS + 1)),
        ticktext=ROUND_LABELS,
        title_text="Training round",
        **AXES_STYLE,
    )
    fig.update_yaxes(title_text="AA", col=1, **AXES_STYLE)
    fig.update_yaxes(title_text="BWT", col=2, **AXES_STYLE)

    fig.update_layout(
        height=500,
        width=1200,
        title_text=(
            "Continual learning metrics: Average Accuracy (AA) and "
            "Backward Transfer (BWT) per round"
        ),
        title_x=0.5,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    path = os.path.join(results_dir, "metrics_bwt_aa.html")
    fig.write_html(path, config={"editable": True})
    print(f"Saved: {path}")
    return fig


###############################################################################
# Plot 5: per-round update time (not cumulative)
###############################################################################


def make_per_round_time_plot(results, results_dir):
    """Plot wall-clock seconds needed to process EACH round (not cumulative).

    The key distinction from training_time.html (which shows cumulative time):
    - standard_cumulative grows each round because it re-trains on more data
    - Qsimov linear should remain roughly constant (only processes new batch)
    """
    fig = go.Figure()

    for method, data in results.items():
        x_rounds, y_update_time = [], []
        prev_cumulative = 0.0
        for k in range(1, N_ROUNDS + 1):
            key = f"after_round_{k}"
            if key not in data:
                continue
            cumulative = data[key]["time(s)"]
            x_rounds.append(k)
            y_update_time.append(cumulative - prev_cumulative)
            prev_cumulative = cumulative

        fig.add_trace(
            go.Scatter(
                x=x_rounds,
                y=y_update_time,
                name=METHOD_LABELS[method],
                line=dict(color=METHOD_COLORS[method], dash=METHOD_DASH[method]),
                mode="lines+markers",
            )
        )

    fig.update_layout(
        height=600,
        width=900,
        title_text="Per-round update time (not cumulative)",
        title_x=0.5,
        xaxis_title="Training round",
        yaxis_title="Update time (s)",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(1, N_ROUNDS + 1)),
            ticktext=ROUND_LABELS,
            **AXES_STYLE,
        ),
        yaxis=AXES_STYLE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    path = os.path.join(results_dir, "per_round_update_time.html")
    fig.write_html(path, config={"editable": True})
    print(f"Saved: {path}")
    return fig


def main(args):
    framework = getattr(args, "framework", "keras")
    results_dir = get_imagenet_continual_learning_results_dir(args.processor, framework=framework)
    results = load_results(results_dir)

    if not results:
        raise RuntimeError(f"No results found in {results_dir}")

    make_overall_accuracy_plot(results, results_dir)
    make_forgetting_curves_plot(results, results_dir)
    make_time_plot(results, results_dir)
    make_bwt_aa_plot(results, results_dir)
    make_per_round_time_plot(results, results_dir)


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
