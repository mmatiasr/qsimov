from experiments.imagenet_subset_by_splits.train_keras import TrainModelsParser
from experiments.path_utils import (
    get_imagenet_subset_by_splits_results_dir as get_results_dir,
)
from experiments.cifar10_gradient_by_splits.plot_gradient_by_splits import (
    load_histories,
    init_figure,
    get_traces_description,
    add_traces,
    configure_axes,
)
from experiments.imagenet_subset_by_splits.preprocess_data import NUM_LABELS


def configure_layout_and_save(fig, results_dir, args):
    framework = args.framework.capitalize()
    # Configure the layout
    fig.update_layout(
        height=1200,
        width=1400,
        title_text=f"Comparison of qsimov and {framework} models on "
        f"ImageNet with {NUM_LABELS} labels "
        f"using {args.model_name} with different data splits",
        title_x=0.5,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # Save the figure
    fig.write_html(
        f"{results_dir}/imagenet_subset_by_splits_{args.model_name}.html",
        config={"editable": False},
    )


def plot_gradient_by_splits(histories, results_dir, args):
    # initialize base figure
    fig = init_figure()

    # dictionary with information for each trace
    traces_description = get_traces_description(histories, args)

    # add traces to figure
    add_traces(fig, traces_description)

    # configure axes
    configure_axes(fig)

    # configure layout and save
    configure_layout_and_save(fig, results_dir, args)


def main(args):
    # where plots histories and models are saved
    results_dir = get_results_dir(args.framework, args.processor)

    # load results
    histories = load_histories(results_dir, args)

    # make plot
    plot_gradient_by_splits(histories, results_dir, args)


###############################################################################
# CLI Parser
###############################################################################
class PlotGradientBySplitsParser(TrainModelsParser):
    def add_arguments(self):
        self.add_processor_argument()
        self.add_splits_argument()
        self.add_model_name_argument()
        self.add_framework_argument()

    def add_framework_argument(self):
        self.add_argument(
            "--framework",
            type=str,
            default="keras",
            choices=["keras"],
        )


if __name__ == "__main__":
    main(PlotGradientBySplitsParser().parse_args())
