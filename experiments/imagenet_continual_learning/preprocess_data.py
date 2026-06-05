"""Load and split the ImageNet subset for continual learning experiments.

Uses all 100 labels (4 train dirs).

Class-incremental split
-----------------------
Round k contains ONLY the classes in group k:
  - Round 1 → classes   0..24
  - Round 2 → classes  25..49
  - Round 3 → classes  50..74
  - Round 4 → classes  75..99

This is the standard class-incremental setting used in the continual-learning
literature.  It guarantees a distribution shift between rounds: a method that
forgets will lose accuracy on round-k val data after being trained on round k+1.
An i.i.d. split (all classes every round) would show no forgetting because each
round re-confirms the same task, so results are uninterpretable as forgetting.

Forgetting measurement
----------------------
For each round, VAL_FRACTION of each class's samples is withheld as a
dedicated validation set.  After training on round K, every method is evaluated
on the val sets of ALL past rounds 1..K.  A drop in val accuracy on round j < K
after training on round K quantifies catastrophic forgetting.
"""

import numpy as np
import os
from experiments.path_utils import get_qsimov_dataset_dir
from experiments.imagenet_subset_by_splits.preprocess_data import (
    NUM_LABELS,
    TRAIN_IMAGES_PER_LABEL,
)

SEED = 42
N_ROUNDS = 4
VAL_FRACTION = 0.15  # fraction of each class's samples withheld per round

assert NUM_LABELS % N_ROUNDS == 0, (
    f"NUM_LABELS ({NUM_LABELS}) must be divisible by N_ROUNDS ({N_ROUNDS})"
)
CLASSES_PER_ROUND = NUM_LABELS // N_ROUNDS  # 25 per round


def load_dataset():
    data_dir = get_qsimov_dataset_dir("imagenet_subset")
    return (
        np.load(os.path.join(data_dir, "x_train.npy")),
        np.load(os.path.join(data_dir, "y_train.npy")),
        np.load(os.path.join(data_dir, "x_test.npy")),
        np.load(os.path.join(data_dir, "y_test.npy")),
    )


def get_class_groups(n_rounds=N_ROUNDS):
    """Return a list of class-index arrays, one per round.

    Example with N_ROUNDS=4, NUM_LABELS=100:
        [array([ 0..24]), array([25..49]), array([50..74]), array([75..99])]
    """
    classes_per_round = NUM_LABELS // n_rounds
    return [
        np.arange(k * classes_per_round, (k + 1) * classes_per_round)
        for k in range(n_rounds)
    ]


def make_round_splits(x_train, y_train, n_rounds=N_ROUNDS, val_fraction=VAL_FRACTION):
    """Class-incremental split: round k contains ONLY classes from group k.

    Parameters
    ----------
    x_train : np.ndarray  shape (N, H, W, C)
    y_train : np.ndarray  shape (N,), integer labels 0..NUM_LABELS-1
    n_rounds : int
    val_fraction : float  fraction of each class's samples withheld for evaluation

    Returns
    -------
    list of (train_x, train_y, val_x, val_y) — one tuple per round.
    val sets are never seen during training and are used to measure forgetting.
    """
    rng = np.random.RandomState(SEED)
    class_groups = get_class_groups(n_rounds)

    rounds = []
    for group_classes in class_groups:
        group_train_x, group_train_y = [], []
        group_val_x, group_val_y = [], []

        for cls in group_classes:
            mask = y_train == cls
            cls_x = x_train[mask]
            cls_y = y_train[mask]

            # shuffle within class
            perm = rng.permutation(len(cls_x))
            cls_x, cls_y = cls_x[perm], cls_y[perm]

            n_val = max(1, int(len(cls_x) * val_fraction))
            group_val_x.append(cls_x[:n_val])
            group_val_y.append(cls_y[:n_val])
            group_train_x.append(cls_x[n_val:])
            group_train_y.append(cls_y[n_val:])

        tx = np.concatenate(group_train_x)
        ty = np.concatenate(group_train_y)
        vx = np.concatenate(group_val_x)
        vy = np.concatenate(group_val_y)

        # shuffle training portion
        perm = rng.permutation(len(tx))
        tx, ty = tx[perm], ty[perm]

        rounds.append((tx, ty, vx, vy))

    return rounds
