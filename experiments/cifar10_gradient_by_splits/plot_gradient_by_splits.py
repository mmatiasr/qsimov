import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pickle
from experiments.path_utils import (
    get_cifar10_gradient_by_splits_results_dir as get_results_dir,
)
from experiments.cifar10_gradient_by_splits.train_keras_models import (
    TrainModelsParser,
    split_to_name,
)
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

# Color assignment for linear system markers (same palette, dotted lines)
LINEAR_SYSTEM_COLORS = ["darkred", "darkblue", "darkgreen", "darkorchid"]


def load_histories(results_dir, args):
    histories = {}

    for model_type in ("", "_path_selector", "_qsimov"):
        # for qsimov or for keras
        histories[model_type] = {}

        # base model is trained with all data, but not qsimov
        if model_type == "":
            splits = args.splits + [None]
        else:
            splits = args.splits

        for split in splits:
            # name corresponding to split and model type
            name = split_to_name(split) + model_type

            # Load history
            path = "{}/{}_{}_history.pkl".format(
                results_dir, name, args.model_name
            )
            with open(path, "rb") as f:
                history = pickle.load(f)

            # start time at 0
            history["time(s)"] = (
                np.array(history["time(s)"]) - history["time(s)"][0]
            )

            # Add epochs
            history["epochs"] = np.arange(len(history["time(s)"])) + 1

            # Add to histories
            histories[model_type][split] = history

    return histories


def init_figure():
    return make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Training accuracy by epoch",
            "Test accuracy by epoch",
            "Training accuracy by time",
            "Test accuracy by time",
        ],
        vertical_spacing=0.07,
        shared_yaxes=True,
    )


def make_traces_colors(args):
    dark_colors = ["red", "blue", "green", "purple"]
    light_colors = ["tomato", "cornflowerblue", "limegreen", "violet"]

    traces_colors = {}

    for model_type in ("", "_path_selector", "_qsimov"):
        traces_colors[model_type] = {}

        if model_type == "":
            colors = dark_colors
            # associate black to the base model trained with all data
            traces_colors[model_type][None] = "black"
        elif model_type == "_path_selector":
            colors = dark_colors
        else:
            colors = light_colors

        # assign a color to each split
        for split, color in zip(args.splits, colors):
            traces_colors[model_type][split] = color

    return traces_colors


def get_trace_description(model_type, split, color, history, args):
    trace = {}

    # name corresponding to split and model type
    if model_type == "":
        model_type_name = args.framework
    elif model_type == "_path_selector":
        model_type_name = "path selector"
    else:
        model_type_name = "qsimov"

    # capitalize first letter
    model_type_name = model_type_name.capitalize()

    # base model is trained with all data, but not qsimov
    split_name = "all" if split is None else split

    trace["name"] = "{} model with {} samples".format(
        model_type_name, split_name
    )

    trace["color"] = color

    # Set dash lines for path selector
    trace["dash"] = "longdash" if model_type == "_path_selector" else None

    # fill info for each subplot
    trace["subplots"] = {}

    train_accuracy = history["accuracy"]
    test_accuracy = history["val_accuracy"]
    times = history["time(s)"]
    epochs = history["epochs"]

    # training accuracy by epoch
    trace["subplots"][(1, 1)] = {"x": epochs, "y": train_accuracy}

    # test accuracy by epoch
    trace["subplots"][(1, 2)] = {"x": epochs, "y": test_accuracy}

    # training accuracy by time
    trace["subplots"][(2, 1)] = {"x": times, "y": train_accuracy}

    # test accuracy by time
    trace["subplots"][(2, 2)] = {"x": times, "y": test_accuracy}

    return trace


def get_traces_description(histories, args):
    # two traces per split
    traces = {}

    # get colors for each trace
    colors = make_traces_colors(args)

    for model_type in ("", "_path_selector", "_qsimov"):
        # one trace per model type
        traces[model_type] = {}

        # base model is trained with all data, but not qsimov
        if model_type == "":
            splits = args.splits + [None]
        else:
            splits = args.splits

        # one trace per split
        for split in splits:
            color = colors[model_type][split]
            traces[model_type][split] = get_trace_description(
                model_type, split, color, histories[model_type][split], args
            )

    return traces


def add_traces(fig, traces_description):
    for model_traces in traces_description.values():  # by model type
        for model_trace in model_traces.values():  # by split
            # by subplot
            for (row, col), trace in model_trace["subplots"].items():
                fig.add_trace(
                    go.Scatter(
                        x=trace["x"],
                        y=trace["y"],
                        name=model_trace["name"],
                        legendgroup=model_trace["name"],
                        showlegend=(row == 1 and col == 1),
                        line=dict(
                            color=model_trace["color"],
                            dash=model_trace["dash"],
                        ),
                    ),
                    row=row,
                    col=col,
                )


def configure_axes(fig):
    # general style
    axes_args = dict(
        showgrid=False,
        linecolor="black",
        linewidth=1.5,
        ticks="inside",
        tickwidth=2,
    )
    fig.update_xaxes(**axes_args)
    fig.update_yaxes(**axes_args)

    # axis labels
    fig.update_xaxes(title_text="Epoch", row=1)  # shared x-axis
    fig.update_xaxes(title_text="Time (s)", row=2)  # shared x-axis
    fig.update_yaxes(title_text="Accuracy", col=1)  # shared y-axis


def configure_layout_and_save(fig, results_dir, args):
    framework = args.framework.capitalize()
    # Configure the layout
    fig.update_layout(
        height=1200,
        width=1400,
        title_text=f"Comparison of qsimov and {framework} models on CIFAR-10 "
        f"using {args.model_name} with different data splits",
        title_x=0.5,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # Save the figure
    fig.write_html(
        f"{results_dir}/cifar10_gradient_by_splits_{args.model_name}.html",
        config={"editable": False},
    )


def plot_gradient_by_splits(histories, linear_results, results_dir, args):
    # initialize base figure
    fig = init_figure()

    # dictionary with information for each trace
    traces_description = get_traces_description(histories, args)

    # add traces to figure
    add_traces(fig, traces_description)

    # add linear system reference lines and markers
    add_linear_system_markers(fig, linear_results, args)

    # configure axes
    configure_axes(fig)

    # configure layout and save
    configure_layout_and_save(fig, results_dir, args)


def load_linear_results(results_dir, args):
    """Load QsimovLinearSystem results if available."""
    linear = {}
    for split in args.splits:
        path = "{}/{}_qsimov_linear_{}_results.pkl".format(
            results_dir, split_to_name(split), args.model_name
        )
        if os.path.exists(path):
            with open(path, "rb") as f:
                linear[split] = pickle.load(f)
    return linear


def add_linear_system_markers(fig, linear_results, args):
    """Add QsimovLinearSystem as horizontal reference lines + time markers.

    The linear system is a one-shot solver, so it has no training curves.
    It is shown as:
      - Horizontal dashed lines at the final test accuracy (accuracy-by-epoch plots)
      - Single marker on the time-based accuracy plots
    """
    if not linear_results:
        return

    for i, (split, res) in enumerate(linear_results.items()):
        color = LINEAR_SYSTEM_COLORS[i % len(LINEAR_SYSTEM_COLORS)]
        label = f"Qsimov Linear (system) — {split} samples"
        acc = res["test_accuracy"]
        t = res["train_time"]

        # Horizontal reference on accuracy-by-epoch subplot (test accuracy, col 2)
        fig.add_hline(
            y=acc, row=1, col=2,
            line=dict(color=color, dash="dot", width=1.5),
            annotation_text=label,
            annotation_position="bottom right",
            annotation_font_size=10,
        )

        # Single marker on accuracy-by-time subplot (test accuracy by time, row 2 col 2)
        fig.add_trace(
            go.Scatter(
                x=[t], y=[acc],
                mode="markers",
                name=label,
                legendgroup=label,
                showlegend=True,
                marker=dict(color=color, size=12, symbol="star"),
            ),
            row=2, col=2,
        )


def main(args):
    # where plots histories and models are saved
    results_dir = get_results_dir(args.framework, args.processor)

    # load results
    histories = load_histories(results_dir, args)
    linear_results = load_linear_results(results_dir, args)

    # make plot
    plot_gradient_by_splits(histories, linear_results, results_dir, args)


###############################################################################
# CLI Parser
###############################################################################
class PlotGradientBySplitsParser(TrainModelsParser):
    def add_arguments(self):
        self.add_processor_argument()
        self.add_splits_argument()
        self.add_framework_argument()
        self.add_model_name_argument()

    def add_framework_argument(self):
        self.add_argument(
            "--framework",
            type=str,
            default="keras",
            choices=["keras", "pytorch"],
        )


if __name__ == "__main__":
    main(PlotGradientBySplitsParser().parse_args())
