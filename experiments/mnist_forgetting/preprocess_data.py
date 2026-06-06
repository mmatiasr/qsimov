"""Load MNIST and split into phase 1 (classes 0-4) and phase 2 (classes 5-9).

Saves numpy arrays to data/mnist_forgetting/ with one-hot encoded labels.
"""

import numpy as np
import os
from experiments.path_utils import get_qsimov_dataset_dir

PHASE1_CLASSES = [0, 1, 2, 3, 4]
PHASE2_CLASSES = [5, 6, 7, 8, 9]
N_CLASSES = 10
DATA_DIR_NAME = "mnist_forgetting"
SEED = 42


def get_data_dir():
    return get_qsimov_dataset_dir(DATA_DIR_NAME)


def filter_by_classes(x, y_int, classes):
    mask = np.isin(y_int, classes)
    return x[mask], y_int[mask]


def to_onehot(y_int, n_classes=N_CLASSES):
    return np.eye(n_classes, dtype=np.float32)[y_int.astype(int)]


def load_data(data_dir=None):
    if data_dir is None:
        data_dir = get_data_dir()
    return {
        "phase1_train_x": np.load(os.path.join(data_dir, "phase1_train_x.npy")),
        "phase1_train_y": np.load(os.path.join(data_dir, "phase1_train_y.npy")).astype(np.float32),
        "phase2_train_x": np.load(os.path.join(data_dir, "phase2_train_x.npy")),
        "phase2_train_y": np.load(os.path.join(data_dir, "phase2_train_y.npy")).astype(np.float32),
        "test_phase1_x": np.load(os.path.join(data_dir, "test_phase1_x.npy")),
        "test_phase1_y": np.load(os.path.join(data_dir, "test_phase1_y.npy")).astype(np.float32),
        "test_phase2_x": np.load(os.path.join(data_dir, "test_phase2_x.npy")),
        "test_phase2_y": np.load(os.path.join(data_dir, "test_phase2_y.npy")).astype(np.float32),
        "test_x": np.load(os.path.join(data_dir, "test_x.npy")),
        "test_y": np.load(os.path.join(data_dir, "test_y.npy")).astype(np.float32),
    }


if __name__ == "__main__":
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)

    data_dir = get_data_dir()
    os.makedirs(data_dir, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(x_train))
    x_train, y_train = x_train[idx], y_train[idx]

    # Split by phase
    p1_x, p1_y = filter_by_classes(x_train, y_train, PHASE1_CLASSES)
    p2_x, p2_y = filter_by_classes(x_train, y_train, PHASE2_CLASSES)
    t1_x, t1_y = filter_by_classes(x_test, y_test, PHASE1_CLASSES)
    t2_x, t2_y = filter_by_classes(x_test, y_test, PHASE2_CLASSES)

    # Shuffle test splits
    idx1 = rng.permutation(len(t1_x))
    t1_x, t1_y = t1_x[idx1], t1_y[idx1]
    idx2 = rng.permutation(len(t2_x))
    t2_x, t2_y = t2_x[idx2], t2_y[idx2]
    idx_all = rng.permutation(len(x_test))
    x_test, y_test = x_test[idx_all], y_test[idx_all]

    # Convert labels to one-hot
    np.save(os.path.join(data_dir, "phase1_train_x.npy"), p1_x)
    np.save(os.path.join(data_dir, "phase1_train_y.npy"), to_onehot(p1_y))
    np.save(os.path.join(data_dir, "phase2_train_x.npy"), p2_x)
    np.save(os.path.join(data_dir, "phase2_train_y.npy"), to_onehot(p2_y))
    np.save(os.path.join(data_dir, "test_phase1_x.npy"), t1_x)
    np.save(os.path.join(data_dir, "test_phase1_y.npy"), to_onehot(t1_y))
    np.save(os.path.join(data_dir, "test_phase2_x.npy"), t2_x)
    np.save(os.path.join(data_dir, "test_phase2_y.npy"), to_onehot(t2_y))
    np.save(os.path.join(data_dir, "test_x.npy"), x_test)
    np.save(os.path.join(data_dir, "test_y.npy"), to_onehot(y_test))

    print(f"Phase 1 (classes 0-4): {len(p1_x)} train, {len(t1_x)} test")
    print(f"Phase 2 (classes 5-9): {len(p2_x)} train, {len(t2_x)} test")
    print(f"All test: {len(x_test)}")
