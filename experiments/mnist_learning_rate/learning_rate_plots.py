import matplotlib
import pickle
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from itertools import product
import os
from experiments.path_utils import get_mnist_learning_rate_results_dir
from experiments.mnist_speed_loss.speed_loss_plots import (
    SpeedLossPlotsParser,
)


def multiples_as_fractions(multiples):
    # Convert multiples to fractions (0.25 -> 1/4)
    fractions = []
    for number in multiples:
        fraction = number.as_integer_ratio()
        if fraction[1] == 1:
            fraction = str(fraction[0])
        else:
            fraction = f"{fraction[0]}/{fraction[1]}"
        fractions.append(fraction)
    return fractions


def generate_color_gradient(n):
    cmap = matplotlib.colormaps["coolwarm"]
    colors = np.linspace(0, 1, n)
    return [matplotlib.colors.to_hex(cmap(c)) for c in colors]


def load_history(path):
    with open(path, "rb") as f:
        return pickle.load(f)


###############################################################################
# Plot by learning rate
###############################################################################


def get_get_plot_by_learning_rate_axes():
    x_axes = ["Learning rate multiple"] * 4
    y_axes = ["Accuracy"] * 2 + ["Binary cross-entropy"] * 2
    return x_axes, y_axes


def add_plot_by_learning_rate_layout_and_save(
    fig, stats_and_histories, output_dir, framework
):
    # Configure the axes

    # general style
    axes_args = dict(
        showgrid=False,
        linecolor="black",
        linewidth=1.5,
        ticks="inside",
        tickwidth=2,
    )

    # we must set the tickvals and ticktext manually for x-axis
    fig.update_xaxes(
        **axes_args,
        type="log",
        tickformat=".4",
        tickmode="array",
        # set the tick values to the original x values
        tickvals=stats_and_histories["learning_rates"],
        # set the tick text to the custom xtick labels
        ticktext=multiples_as_fractions(stats_and_histories["multiples"]),
        tickfont=dict(size=14),  # customize the font size of the tick labels)
    )

    # Axis labels
    x_axes, y_axes = get_get_plot_by_learning_rate_axes()
    fig.update_xaxes(title_text=x_axes[0])  # shared x-axis
    fig.update_yaxes(title_text=y_axes[0], row=1, col=1)  # y-axis for row 1
    fig.update_yaxes(title_text=y_axes[2], row=2, col=1)  # y-axis for row 2
    fig.update_yaxes(**axes_args)

    # Configure the layout
    fig.update_layout(
        height=1200,
        width=1400,
        title_text="Effect of learning rate on Qsimov Gradient "
        + f"and vanilla {framework.capitalize()}.",
        title_x=0.5,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # Save the figure
    fig.write_html(
        f"{output_dir}/learning_rate_plot_by_learning_rate.html",
        config={"editable": False},
    )


def get_plot_by_learning_rate_fig():
    return make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Train accuracy by learning rate",
            "Test accuracy by learning rate",
            "Train binary cross-entropy by learning rate",
            "Test binary cross-entropy by learning rate",
        ],
        vertical_spacing=0.07,
        shared_yaxes=True,
    )


def add_plot_by_learning_rate_subplot(
    history_key, stats_and_histories, fig, row, col, framework
):
    traces = []
    trace_names = []
    trace_colors = ["red", "green", "blue", "tomato", "palegreen", "cyan"]

    # Compute trace names and data for Qsimov and the keras/pytorch model
    for name, stats in [
        ("Qsimov", stats_and_histories["qsimov"]),
        (framework.capitalize(), stats_and_histories["model"]),
    ]:
        # Get the min, max, and mean of the statistic for each learning rate
        traces.append([min(stat[history_key]) for stat in stats])
        traces.append([max(stat[history_key]) for stat in stats])
        traces.append([np.mean(stat[history_key]) for stat in stats])

        # Add the trace names
        trace_names.append(
            f"{name} minimum in {len(stats[0][history_key])} epochs"
        )
        trace_names.append(
            f"{name} maximum in {len(stats[0][history_key])} epochs"
        )
        trace_names.append(
            f"{name} mean in {len(stats[0][history_key])} epochs"
        )

    # Add the traces to the figure
    for trace_idx, trace_data in enumerate(traces):
        fig.add_trace(
            go.Scatter(
                x=stats_and_histories["learning_rates"],
                y=trace_data,
                name=trace_names[trace_idx],
                legendgroup=trace_names[trace_idx],
                showlegend=(row == 0 and col == 0),
                line=dict(color=trace_colors[trace_idx]),
            ),
            row=row + 1,
            col=col + 1,
        )


def make_plot_by_learning_rate(output_dir, stats_and_histories, framework):
    fig = get_plot_by_learning_rate_fig()

    # keys to index into histories, one for each subplot
    stats_keys = ["accuracy", "val_accuracy", "loss", "val_loss"]

    for subplot_idx, (row, col) in enumerate(product(range(2), range(2))):
        # get the statistics key for this subplot
        subplot_stat_key = stats_keys[subplot_idx]

        add_plot_by_learning_rate_subplot(
            subplot_stat_key, stats_and_histories, fig, row, col, framework
        )

    add_plot_by_learning_rate_layout_and_save(
        fig, stats_and_histories, output_dir, framework
    )


###############################################################################
# Plot learning rate by epoch
###############################################################################


def get_learning_rate_by_epoch_fig(framework):
    framework = framework.capitalize()
    return make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"Accuracy of {framework}",
            "Accuracy of Qsimov",
            f"Cross-entropy of {framework}",
            "Cross-entropy of Qsimov",
        ],
        vertical_spacing=0.07,
        shared_yaxes=True,
    )


def get_learning_rate_by_epoch_axes():
    x_axes = ["Epoch"] * 4
    y_axes = ["Accuracy"] * 2 + ["Binary cross-entropy"] * 2
    return x_axes, y_axes


def add_learning_rate_by_epoch_layout_and_save(
    fig, data_source, stats_and_histories, output_dir
):
    # Get axes names
    x_axes, y_axes = get_learning_rate_by_epoch_axes()

    # Configure axes

    # General axes args
    axes_args = dict(
        showgrid=False,
        linecolor="black",
        linewidth=1.5,
        ticks="inside",
        tickwidth=2,
    )
    # all x axis names are shared
    fig.update_xaxes(tickformat=".4", title_text=x_axes[0], **axes_args)

    # Set y axes (they are different for each row)
    fig.update_yaxes(**axes_args)
    fig.update_yaxes(title_text=y_axes[0], row=1, col=1)
    fig.update_yaxes(title_text=y_axes[2], row=2, col=1)

    # Configure figure
    data_source_verbose = data_source.capitalize()
    base_learning_rate = stats_and_histories["base_learning_rate"]
    fig.update_layout(
        height=1200,
        width=1400,
        title_text="Model performance by learning rate "
        + f"({data_source_verbose}).",
        legend_title="Multiple of original learning rate "
        + f"({base_learning_rate}):\n<br>",
        title_x=0.5,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # Save figure
    fig.write_html(
        f"{output_dir}/learning_rate_plot_by_epoch_{data_source}.html",
        config={"editable": False},
    )
    return fig


def plot_learning_rate_by_epoch(
    output_dir, stats_and_histories, framework, data_source
):
    # Create figure
    fig = get_learning_rate_by_epoch_fig(framework)

    # Get keys to index histories
    data_source_suffix = "val_" if data_source == "test" else ""
    stats_keys = [f"{data_source_suffix}{key}" for key in ["accuracy", "loss"]]

    # one trace for each learning rate multiple
    learning_rates = stats_and_histories["learning_rates"]
    multiples = stats_and_histories["multiples"]

    # trace names and colors
    trace_names = [
        f"{fraction} times the learning rate"
        for fraction in multiples_as_fractions(multiples)
    ]
    trace_colors = generate_color_gradient(len(multiples))

    for subplot_idx, (row, col) in enumerate(product(range(2), range(2))):
        key = stats_keys[row]  # accuracy or loss
        qsimov_stats = stats_and_histories["qsimov"]  # trace data on qsimov
        model_stats = stats_and_histories["model"]  # trace data on model
        stats = (model_stats, qsimov_stats)[col]  # select model or qsimov

        # get traces for each learning rate multiple on qsimov or model
        traces = [stats[idx][key] for idx in range(learning_rates.shape[0])]

        # currently we are not using the last learning rate multiple here
        traces = traces[:-1]

        # add traces to figure
        for trace_idx, trace_data in enumerate(traces):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(qsimov_stats[0][key])) + 1,  # epochs
                    y=trace_data,
                    name=trace_names[trace_idx],
                    legendgroup=trace_names[trace_idx],
                    showlegend=(subplot_idx == 0),
                    line=dict(color=trace_colors[trace_idx]),
                ),
                row=row + 1,
                col=col + 1,
            )

    # Configure axes and layout and save
    add_learning_rate_by_epoch_layout_and_save(
        fig, data_source, stats_and_histories, output_dir
    )


# retrieve previously generated files with statistics and histories
def load_stats_and_histories(output_dir, framework):
    # qsimov history for each learning rate multiple
    qsimov_stats = load_history(f"{output_dir}/learning_rate_qsimov_stats.pkl")

    # keras/pytorch model history for each learning rate multiple
    model_stats = load_history(
        f"{output_dir}/learning_rate_{framework}_stats.pkl"
    )

    # wich learning rates were used
    learning_rates = np.load(os.path.join(output_dir, "learning_rates.npy"))

    # what multiple of the original learning rate are they
    multiples = np.load(
        os.path.join(output_dir, "learning_rate_multiples.npy")
    )

    # what was the original learning rate
    base_learning_rate = np.load(
        os.path.join(output_dir, "base_learning_rate.npy")
    )
    return {
        "qsimov": qsimov_stats,
        "model": model_stats,
        "learning_rates": learning_rates,
        "multiples": multiples,
        "base_learning_rate": base_learning_rate,
    }


def main(args):
    # retrieve arguments
    processor = args.processor
    framework = args.framework

    # output directory
    output_dir = get_mnist_learning_rate_results_dir(framework, processor)
    os.makedirs(output_dir, exist_ok=True)

    # load stats and histories
    stats_and_histories = load_stats_and_histories(output_dir, framework)

    # Plot with learning rate on x axis
    make_plot_by_learning_rate(output_dir, stats_and_histories, framework)

    # Plot with epoch on x axis
    plot_learning_rate_by_epoch(
        output_dir, stats_and_histories, framework, "train"
    )
    plot_learning_rate_by_epoch(
        output_dir, stats_and_histories, framework, "test"
    )


if __name__ == "__main__":
    main(SpeedLossPlotsParser().parse_args())  # same arguments as speed_loss
