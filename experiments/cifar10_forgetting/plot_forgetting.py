"""Generate forgetting experiment plots for CIFAR-10 class-incremental learning."""

import os
import pickle
import argparse

from experiments.path_utils import get_cifar10_forgetting_results_dir

METHODS = [
    "base",
    "linear_accum",
    "linear_new_only",
    "gradient_new_only",
    "finetune_new_only",
    "cumulative",
]

METHOD_LABELS = {
    "base":              "Base model<br>(phase 1 only)",
    "linear_accum":      "Qsimov Linear<br>(accumulated) ★",
    "linear_new_only":   "Qsimov Linear<br>(phase 2 only)",
    "gradient_new_only": "Qsimov Gradient<br>(phase 2 only)",
    "finetune_new_only": "Standard<br>Fine-tune",
    "cumulative":        "Cumulative<br>(oracle)",
}


def load_results(results_dir):
    results = {}
    for method in METHODS:
        path = os.path.join(results_dir, f"{method}_results.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                results[method] = pickle.load(f)
    return results


def create_accuracy_plot(results, output_path):
    import plotly.graph_objects as go

    present = [m for m in METHODS if m in results]
    labels  = [METHOD_LABELS[m] for m in present]

    acc_old = [results[m]["acc_old"] * 100 for m in present]
    acc_new = [results[m]["acc_new"] * 100 for m in present]
    acc_all = [results[m]["acc_all"] * 100 for m in present]

    fig = go.Figure(data=[
        go.Bar(
            name="Old classes (0-4)<br>airplane/auto/bird/cat/deer",
            x=labels, y=acc_old,
            marker_color="#3182bd",
            text=[f"{v:.1f}%" for v in acc_old],
            textposition="outside",
        ),
        go.Bar(
            name="New classes (5-9)<br>dog/frog/horse/ship/truck",
            x=labels, y=acc_new,
            marker_color="#e6550d",
            text=[f"{v:.1f}%" for v in acc_new],
            textposition="outside",
        ),
        go.Bar(
            name="All classes",
            x=labels, y=acc_all,
            marker_color="#31a354",
            text=[f"{v:.1f}%" for v in acc_all],
            textposition="outside",
        ),
    ])

    fig.update_layout(
        title={
            "text": (
                "CIFAR-10 Class-Incremental: Catastrophic Forgetting vs. Qsimov<br>"
                "<sub>Phase 1: train on classes 0–4 (airplane…deer) | "
                "Phase 2: update with classes 5–9 (dog…truck) | "
                "★ = no-forgetting method</sub>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Method",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 118]),
        barmode="group",
        legend=dict(x=0.60, y=1.0),
        template="plotly_white",
        width=1050,
        height=580,
    )

    fig.write_html(output_path)
    print(f"Plot saved to {output_path}")


def main(args):
    results_dir = get_cifar10_forgetting_results_dir(args.framework, args.processor)
    results = load_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}. Run run_forgetting.py first.")
        return

    print(f"\nResults ({len(results)} methods):")
    for method, m in results.items():
        print(
            f"  {method:<22} acc_old={m['acc_old']:.4f}  "
            f"acc_new={m['acc_new']:.4f}  acc_all={m['acc_all']:.4f}  "
            f"time={m['time']:.1f}s"
        )

    output_path = os.path.join(results_dir, "cifar10_forgetting_plot.html")
    create_accuracy_plot(results, output_path)


class PlotForgettingParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="cpu")
        self.add_argument("--framework", choices=["keras", "pytorch"], default="keras")


if __name__ == "__main__":
    main(PlotForgettingParser().parse_args())
