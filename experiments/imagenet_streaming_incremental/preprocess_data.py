"""Split ImageNet subset training data into class-incremental streaming batches.

N_BATCHES sequential batches, each covering CLASSES_PER_BATCH new classes.
With NUM_LABELS=100 and N_BATCHES=20, each batch introduces 5 new classes.

The non-stationary ordering creates genuine distribution shift between batches:
  Batch  1 → trains on classes  0-4    Batch 11 → trains on classes 50-54
  Batch  2 → trains on classes  5-9    ...
  ...                                  Batch 20 → trains on classes 95-99

This allows meaningful measurement of catastrophic forgetting:
after training on batch k, accuracy on cumulative test data (classes 0..k*5-1)
drops for methods that forget and stays high for qsimov_linear_accum.

Cumulative evaluation
---------------------
After batch k, every method is evaluated on the test samples belonging to
ALL classes seen so far (0..(k+1)*CLASSES_PER_BATCH - 1).  A method that
forgets sees lower cumulative accuracy because old-class predictions degrade.

Prerequisites
-------------
Requires the ImageNet subset numpy files at:
    {QSIMOV_HOME}/data/imagenet_subset/x_train.npy  (N, 224, 224, 3)
    {QSIMOV_HOME}/data/imagenet_subset/y_train.npy  (N,) int labels 0..99
    {QSIMOV_HOME}/data/imagenet_subset/x_test.npy
    {QSIMOV_HOME}/data/imagenet_subset/y_test.npy
These are created by the imagenet_subset_by_splits preprocessing scripts.
"""

import numpy as np
import os
from experiments.path_utils import get_qsimov_dataset_dir
from experiments.imagenet_subset_by_splits.preprocess_data import NUM_LABELS

SEED = 42
N_BATCHES = 20
VAL_FRACTION = 0.15

assert NUM_LABELS % N_BATCHES == 0, (
    f"NUM_LABELS ({NUM_LABELS}) must be divisible by N_BATCHES ({N_BATCHES})"
)
CLASSES_PER_BATCH = NUM_LABELS // N_BATCHES  # 5 new classes per batch


def load_dataset():
    data_dir = get_qsimov_dataset_dir("imagenet_subset")
    return (
        np.load(os.path.join(data_dir, "x_train.npy")),
        np.load(os.path.join(data_dir, "y_train.npy")),
        np.load(os.path.join(data_dir, "x_test.npy")),
        np.load(os.path.join(data_dir, "y_test.npy")),
    )


def make_streaming_batches(x_train, y_train, x_test, y_test,
                           n_batches=N_BATCHES, val_fraction=VAL_FRACTION):
    """Class-incremental split into sequential streaming batches.

    Parameters
    ----------
    x_train, y_train : full training arrays (all 100 classes)
    x_test, y_test   : full test arrays (all 100 classes)
    n_batches        : number of streaming batches
    val_fraction     : fraction of each class's training samples held out

    Returns
    -------
    batches : list of dicts, one per batch, with keys:
        train_x, train_y  — training samples for THIS batch's classes only
        val_x, val_y      — held-out val for THIS batch's classes
        cum_test_x, cum_test_y — test samples for ALL classes seen so far

    The cumulative test set grows with each batch, enabling measurement of
    forgetting on previously-seen classes.
    """
    rng = np.random.RandomState(SEED)
    classes_per_batch = NUM_LABELS // n_batches

    batches = []
    for b in range(n_batches):
        batch_classes = np.arange(b * classes_per_batch, (b + 1) * classes_per_batch)

        train_x_b, train_y_b = [], []
        val_x_b, val_y_b = [], []

        for cls in batch_classes:
            mask = y_train == cls
            cx, cy = x_train[mask], y_train[mask]

            perm = rng.permutation(len(cx))
            cx, cy = cx[perm], cy[perm]

            n_val = max(1, int(len(cx) * val_fraction))
            val_x_b.append(cx[:n_val])
            val_y_b.append(cy[:n_val])
            train_x_b.append(cx[n_val:])
            train_y_b.append(cy[n_val:])

        tx = np.concatenate(train_x_b)
        ty = np.concatenate(train_y_b)
        vx = np.concatenate(val_x_b)
        vy = np.concatenate(val_y_b)

        perm = rng.permutation(len(tx))
        tx, ty = tx[perm], ty[perm]

        # Cumulative test set: all classes seen up to and including batch b
        seen_classes = np.arange((b + 1) * classes_per_batch)
        test_mask = np.isin(y_test, seen_classes)

        batches.append({
            "train_x": tx,
            "train_y": ty,
            "val_x": vx,
            "val_y": vy,
            "cum_test_x": x_test[test_mask],
            "cum_test_y": y_test[test_mask],
            "classes": batch_classes,
        })

    return batches
