from experiments.mnist_speed_loss.tf_keras.train_keras_models import (
    TrainModelsParser,
)
import pickle
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from itertools import product
from experiments.path_utils import (
    get_mnist_speed_loss_results_dir as get_results_dir,
)


# Load a history file and add the epochs and time columns
def load_history_with_times_epochs(path):
    with open(path, "rb") as f:
        history = pickle.load(f)
        history["epochs"] = np.arange(len(history["time(s)"]), dtype=int)
        return history


# Base figure for the plots
def make_figure(data_source, loss_function):
    return make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            data_source + " accuracy by epoch",
            data_source + " accuracy by time",
            data_source + " " + loss_function + " by epoch",
            data_source + " " + loss_function + " by time",
        ],
        vertical_spacing=0.07,
    )


def get_keys(data_source):
    x_keys = ["epochs", "time(s)", "epochs", "time(s)"]
    y_keys = ["accuracy", "accuracy", "loss", "loss"]
    if data_source == "Test":
        y_keys = [f"val_{key}" for key in y_keys]
    return x_keys, y_keys


def update_general_layout_and_save(
    fig, output_dir, data_source, model_name, framework, loss_function
):
    fig.update_layout(
        height=1200,
        width=1400,
        title_text=f"Qsimov vs vanilla {framework.capitalize()} "
        + f"models using {loss_function}.",
        title_x=0.5,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    axes_args = dict(
        linecolor="black", linewidth=1.5, ticks="inside", tickwidth=2
    )
    fig.update_xaxes(**axes_args)
    fig.update_yaxes(**axes_args)

    fig.write_html(
        f"{output_dir}/{model_name}_gradient_comparison_"
        + f"{data_source}.html",
        config={"editable": True},
    )


def get_axis_titles(data_source, loss_function):
    x_axes = ["Epoch", "Time (s)", "Epoch", "Time (s)"]
    y_axes = [
        data_source + " accuracy",
        data_source + " accuracy",
        data_source + " " + loss_function,
        data_source + " " + loss_function,
    ]
    return x_axes, y_axes


def make_traces_info(
    framework, model_name, partial, loss_function, dict_models
):
    traces = [
        # Base model trained on partial data
        {
            "color": "red",
            "name": f"{framework} trained with a {partial}".capitalize(),
            "data": dict_models[f"{model_name}_model_history"],
        },
        # Model trained on full data
        {
            "color": "blue",
            "name": f"{framework} trained with full data".capitalize(),
            "data": dict_models[f"{model_name}_full_model_history"],
        },
        # Qsimov gradient model
        {
            "color": "black",
            "name": f"Qsimov gradient (path selector fit on {partial}), "
            "fit on full data",
            "data": dict_models[f"qsimov_gradient_{model_name}_history"],
        },
    ]

    if loss_function != "mse":
        return traces

    traces.append(
        {
            "color": "orange",
            "name": "Qsimov linear system (path selector fit "
            f"on {partial}), fit on full data",
            "data": dict_models[f"qsimov_linear_{model_name}_history"],
        }
    )
    return traces


def add_trace(trace_info, x_key, y_key, fig, subplot_idx, row, col):
    x = trace_info["data"][x_key]
    if x_key == "epochs":  # Shift epochs by 1
        x = np.array(x) + 1
    fig.add_trace(
        go.Scatter(
            x=x,
            y=trace_info["data"][y_key],
            name=trace_info["name"],
            legendgroup=trace_info["name"],
            showlegend=(subplot_idx == 0),
            line=dict(color=trace_info["color"]),
        ),
        row=row + 1,
        col=col + 1,
    )


def add_subplot(
    x_keys, y_keys, x_axes, y_axes, traces, fig, subplot_idx, row, col
):
    # retrieve the corresponding key that indexes the history object
    x_key, y_key = x_keys[subplot_idx], y_keys[subplot_idx]
    for trace_info in traces:
        add_trace(trace_info, x_key, y_key, fig, subplot_idx, row, col)

    fig["layout"][f"xaxis{subplot_idx+1}"]["title"] = x_axes[subplot_idx]
    fig["layout"][f"yaxis{subplot_idx+1}"]["title"] = y_axes[subplot_idx]


def make_plots(
    data_source,  # "Train" or "Test"
    partial,  # "half" or "quarter"
    loss_function,
    model_name,
    framework,
    dict_models,
    output_dir,
):
    # Make the figure
    fig = make_figure(data_source, loss_function)

    # Get the keys to index the history objects
    x_keys, y_keys = get_keys(data_source)

    # Get the axis titles
    x_axes, y_axes = get_axis_titles(data_source, loss_function)

    # Get data, color and name for each trace
    traces = make_traces_info(
        framework, model_name, partial, loss_function, dict_models
    )

    # Add the traces to the figure for each subplot
    for subplot_idx, (row, col) in enumerate(product(range(2), range(2))):
        add_subplot(
            x_keys, y_keys, x_axes, y_axes, traces, fig, subplot_idx, row, col
        )

    update_general_layout_and_save(
        fig, output_dir, data_source, model_name, framework, loss_function
    )
    return fig


# Fill an entry of the dict_models dictionary with the history objects
# corresponding to the model_name (e.g. mse_half) and the loss function
# (e.g. mse)
def fill_dict_models(dict_models, model_name, output_dir, loss_function):
    def dup(a):  # Turn a list of one element into a list of two elements
        return [float(a[0]), float(a[0])]

    # Load the history objects
    dict_models[
        f"{model_name}_model_history"
    ] = load_history_with_times_epochs(
        f"{output_dir}/{model_name}_model_history.pkl"
    )
    dict_models[
        f"{model_name}_full_model_history"
    ] = load_history_with_times_epochs(
        f"{output_dir}/{model_name}_full_model_history.pkl"
    )
    dict_models[
        f"qsimov_gradient_{model_name}_history"
    ] = load_history_with_times_epochs(
        f"{output_dir}/qsimov_gradient_" f"{model_name}_history.pkl"
    )

    # Modify Qsimov linear history to match the other models
    if loss_function == "mse":
        # Load the history object
        hist_key = f"qsimov_linear_{model_name}_history"
        dict_models[hist_key] = load_history_with_times_epochs(
            f"{output_dir}/qsimov_linear_" f"{model_name}_history.pkl"
        )

        # Make it so that there are two epochs
        dict_models[hist_key]["epochs"] = [
            0,
            len(dict_models[f"{model_name}_model_history"]["epochs"]) - 1,
        ]

        # Make it so that there are two points in the history
        for metric in ["accuracy", "loss", "val_accuracy", "val_loss"]:
            dict_models[hist_key][metric] = dup(dict_models[hist_key][metric])

        # Make it so that there are two times
        dict_models[hist_key]["time(s)"] = [
            dict_models[hist_key]["time(s)"][0],
            dict_models[hist_key]["time(s)"][0] + 1,
        ]
        # The last one is the maximum time of the other models
        dict_models[hist_key]["time(s)"][-1] = max(
            (
                hist["time(s)"][-1]
                for hist in [
                    dict_models[f"{model_name}_model_history"],
                    dict_models[f"{model_name}_full_model_history"],
                    dict_models[f"qsimov_gradient_{model_name}_history"],
                ]
            ),
        )


# Retrieve the history objects for each base model and each loss function
def retrieve_dict_models(partial_datas, loss_functions, models, output_dir):
    # separator for printing
    separator = "\n" + "-" * 50 + "\n"

    dict_models = {}

    for partial_idx, name in enumerate(partial_datas):
        print(f"Loading {name} models" + separator)
        for loss_idx, loss_function in enumerate(loss_functions):
            print(f"with {loss_function} loss function" + separator)
            current_models = models[loss_idx]
            model_name = current_models[partial_idx]
            fill_dict_models(
                dict_models,
                model_name,
                output_dir,
                loss_function,
            )

    return dict_models


def main(args):
    # Retrieve the arguments
    processor = args.processor
    framework = args.framework

    # Result directory
    output_dir = get_results_dir(framework, processor)

    # dataset portions used to train the models
    partial_datas = ["half", "quarter"]

    # models to train
    loss_functions = ["categorical_crossentropy", "mse"]
    crossentropy_models = ["crossentropy_half", "crossentropy_quarter"]
    mse_models = ["mse_half", "mse_quarter"]
    models = [crossentropy_models, mse_models]
    data_source = ["Train", "Test"]

    # load and prepare history objects
    dict_models = retrieve_dict_models(
        partial_datas, loss_functions, models, output_dir
    )

    # make plots
    for partial_idx, partial_name in enumerate(partial_datas):
        for loss_idx, loss_function in enumerate(loss_functions):
            current_models = models[loss_idx]
            model_name = current_models[partial_idx]
            for source in data_source:
                # train partial_model
                make_plots(
                    source,  # "Train" or "Test"
                    partial_name,  # "half" or "quarter"
                    loss_function,
                    model_name,
                    framework,
                    dict_models,
                    output_dir,
                )


###############################################################################
# Parser
###############################################################################


class SpeedLossPlotsParser(TrainModelsParser):
    def add_arguments(self):
        self.add_processor_argument()
        self.add_framework_argument()

    def add_framework_argument(self):
        self.add_argument(
            "--framework",
            help="framework used in the execution",
            default="keras",
            choices=["keras", "pytorch"],
        )


if __name__ == "__main__":
    args = SpeedLossPlotsParser().parse_args()
    main(args)
