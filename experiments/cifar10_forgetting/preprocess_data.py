"""Load CIFAR-10 and split into phase 1 (classes 0-4) and phase 2 (classes 5-9).

Class names:
  0=airplane  1=automobile  2=bird  3=cat  4=deer
  5=dog  6=frog  7=horse  8=ship  9=truck
"""

import numpy as np
import os
from experiments.path_utils import get_qsimov_dataset_dir

PHASE1_CLASSES = [0, 1, 2, 3, 4]
PHASE2_CLASSES = [5, 6, 7, 8, 9]
N_CLASSES = 10
DATA_DIR_NAME = "cifar10_forgetting"
SEED = 42


def get_data_dir():
    return get_qsimov_dataset_dir(DATA_DIR_NAME)


def load_data(data_dir=None):
    if data_dir is None:
        data_dir = get_data_dir()
    return {
        "phase1_train_x": np.load(os.path.join(data_dir, "phase1_train_x.npy")),
        "phase1_train_y": np.load(os.path.join(data_dir, "phase1_train_y.npy")).astype(np.float32),
        "phase2_train_x": np.load(os.path.join(data_dir, "phase2_train_x.npy")),
        "phase2_train_y": np.load(os.path.join(data_dir, "phase2_train_y.npy")).astype(np.float32),
        "test_phase1_x":  np.load(os.path.join(data_dir, "test_phase1_x.npy")),
        "test_phase1_y":  np.load(os.path.join(data_dir, "test_phase1_y.npy")).astype(np.float32),
        "test_phase2_x":  np.load(os.path.join(data_dir, "test_phase2_x.npy")),
        "test_phase2_y":  np.load(os.path.join(data_dir, "test_phase2_y.npy")).astype(np.float32),
        "test_x":         np.load(os.path.join(data_dir, "test_x.npy")),
        "test_y":         np.load(os.path.join(data_dir, "test_y.npy")).astype(np.float32),
    }


if __name__ == "__main__":
    import torchvision

    data_dir = get_data_dir()
    os.makedirs(data_dir, exist_ok=True)

    _cache = os.path.join(data_dir, "_torchvision_cache")
    _train = torchvision.datasets.CIFAR10(root=_cache, train=True,  download=True)
    _test  = torchvision.datasets.CIFAR10(root=_cache, train=False, download=True)

    x_train = np.array(_train.data, dtype=np.float32) / 255.0
    y_train = np.array(_train.targets)
    x_test  = np.array(_test.data,  dtype=np.float32) / 255.0
    y_test  = np.array(_test.targets)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(x_train))
    x_train, y_train = x_train[idx], y_train[idx]

    def split_by_classes(x, y, classes):
        mask = np.isin(y, classes)
        return x[mask], np.eye(N_CLASSES, dtype=np.float32)[y[mask]]

    p1_x, p1_y = split_by_classes(x_train, y_train, PHASE1_CLASSES)
    p2_x, p2_y = split_by_classes(x_train, y_train, PHASE2_CLASSES)
    t1_x, t1_y = split_by_classes(x_test, y_test, PHASE1_CLASSES)
    t2_x, t2_y = split_by_classes(x_test, y_test, PHASE2_CLASSES)

    y_test_onehot = np.eye(N_CLASSES, dtype=np.float32)[y_test]
    idx_t = rng.permutation(len(x_test))
    x_test, y_test_onehot = x_test[idx_t], y_test_onehot[idx_t]

    np.save(os.path.join(data_dir, "phase1_train_x.npy"), p1_x)
    np.save(os.path.join(data_dir, "phase1_train_y.npy"), p1_y)
    np.save(os.path.join(data_dir, "phase2_train_x.npy"), p2_x)
    np.save(os.path.join(data_dir, "phase2_train_y.npy"), p2_y)
    np.save(os.path.join(data_dir, "test_phase1_x.npy"),  t1_x)
    np.save(os.path.join(data_dir, "test_phase1_y.npy"),  t1_y)
    np.save(os.path.join(data_dir, "test_phase2_x.npy"),  t2_x)
    np.save(os.path.join(data_dir, "test_phase2_y.npy"),  t2_y)
    np.save(os.path.join(data_dir, "test_x.npy"),         x_test)
    np.save(os.path.join(data_dir, "test_y.npy"),         y_test_onehot)

    print(f"Phase 1 (classes 0-4): {len(p1_x)} train, {len(t1_x)} test")
    print(f"Phase 2 (classes 5-9): {len(p2_x)} train, {len(t2_x)} test")
    print(f"All test: {len(x_test)}")
