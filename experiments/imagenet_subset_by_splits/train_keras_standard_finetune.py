"""Standard fine-tuning baseline for the imagenet_subset_by_splits experiment.

This provides a fair comparison for QsimovGradient:

  QsimovGradient   trains a custom masked dense layer on a SUBSET of data
  StandardFinetune fine-tunes the same base model's last layers on the SAME SUBSET

Both start from identical base models (path_selector VGG16 pre-trained on the
FULL dataset), train on the same subset, and are evaluated on the same test set.

Previously, the imagenet_subset_by_splits experiment compared Qsimov on a data
SUBSET against a full model trained from scratch on ALL data — an unfair baseline
that inflated the apparent cost of standard training.  This script provides the
correct apples-to-apples comparison: same model, same data, different training
algorithm.

Results are saved as:
    {split_name}_standard_finetune_{model_name}_history.pkl
    {split_name}_standard_finetune_{model_name}_model.tf
"""

import pickle
import os
import argparse
import multiprocessing as mp
import numpy as np
from experiments.path_utils import (
    get_imagenet_subset_by_splits_results_dir as get_results_dir,
)
from experiments.imagenet_subset_by_splits.train_keras import (
    TrainModelsParser,
    split_to_name,
    BATCH_SIZE,
)
from experiments.imagenet_subset_by_splits.preprocess_data import (
    load_dataset,
    make_split,
)


def make_imports():
    global tf, keras, AccumulatedEpochTimeTracker, init_tensorflow
    global get_optimizer

    import tensorflow as tf
    from tensorflow import keras
    from experiments.mnist_speed_loss.tf_keras.utils import (
        init_tensorflow,
        AccumulatedEpochTimeTracker,
    )
    from experiments.imagenet_subset_by_splits.keras_model_factory import get_optimizer


def load_path_selector_model(results_dir, model_name, split):
    """Load the pre-trained path_selector model for this split."""
    model_file = f"{split_to_name(split)}_path_selector_{model_name}_model.tf"
    model_path = os.path.join(results_dir, model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Path selector model not found: {model_path}\n"
            "Run train_keras.py --train-path-selector first."
        )
    return keras.models.load_model(model_path)


def train_model(model, train_x, train_y, test_x, test_y, args):
    time_tracker = AccumulatedEpochTimeTracker()
    history = model.fit(
        x=train_x,
        y=train_y,
        validation_data=(test_x, test_y),
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        callbacks=[time_tracker],
    )
    history.history["time(s)"] = time_tracker.times
    return history


def execute_logic(results_dir, split, args):
    train_x, train_y, test_x, test_y = load_dataset()

    if split is not None:
        train_x, train_y = make_split(train_x, train_y, split)

    # Load the same base model used by Qsimov (path_selector VGG16, full-data pretrained)
    model = load_path_selector_model(results_dir, args.model_name, split)

    # Freeze conv base, fine-tune only top layers — same as Qsimov setup
    for layer in model.layers[:-3]:
        layer.trainable = False

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=get_optimizer(args.model_name, is_qsimov=False),
        metrics=["accuracy"],
    )

    history = train_model(model, train_x, train_y, test_x, test_y, args)

    split_name = split_to_name(split)
    with open(
        f"{results_dir}/{split_name}_standard_finetune_{args.model_name}_history.pkl",
        "wb",
    ) as f:
        pickle.dump(history.history, f)

    model.save(
        f"{results_dir}/{split_name}_standard_finetune_{args.model_name}_model.tf",
        save_format="tf",
    )
    print(f"\nSaved standard_finetune results for split={split_name}")


def main(args):
    results_dir = get_results_dir("keras", args.processor)
    os.makedirs(results_dir, exist_ok=True)

    def main_subprocess(split):
        make_imports()
        init_tensorflow(tf, args.processor)
        execute_logic(results_dir, split, args)

    for split in args.splits:
        print(f"\n\nStandard fine-tuning (fair baseline) split={split_to_name(split)}\n")
        p = mp.Process(target=main_subprocess, args=(split,))
        p.start()
        p.join()


###############################################################################
# CLI Parser
###############################################################################


class StandardFinetuneParser(TrainModelsParser):
    def add_arguments(self):
        self.add_processor_argument()
        self.add_epochs_argument()
        self.add_splits_argument()
        self.add_model_name_argument()


if __name__ == "__main__":
    main(StandardFinetuneParser().parse_args())
