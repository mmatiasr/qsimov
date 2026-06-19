import numpy as np
import os
from experiments.path_utils import get_qsimov_dataset_dir

SEED = 42


def make_split(x_train, y_train, split):
    x_by_label, y_by_label = [], []
    for i in range(10):
        selected = y_train[:, i] == 1
        x_by_label.append(x_train[selected])
        y_by_label.append(y_train[selected])

    split_x, split_y = [], []
    for label_idx in range(10):
        n = split // 10 + (1 if label_idx < split % 10 else 0)
        split_x.append(x_by_label[label_idx][:n])
        split_y.append(y_by_label[label_idx][:n])

    split_x = np.concatenate(split_x)
    split_y = np.concatenate(split_y)

    idx = np.arange(len(split_x))
    np.random.shuffle(idx)
    return split_x[idx], split_y[idx]


def load_dataset():
    data_dir = get_qsimov_dataset_dir("cifar10")
    return (
        np.load(os.path.join(data_dir, "x_train.npy")),
        np.load(os.path.join(data_dir, "y_train.npy")),
        np.load(os.path.join(data_dir, "x_test.npy")),
        np.load(os.path.join(data_dir, "y_test.npy")),
    )


def main():
    import torchvision

    np.random.seed(SEED)

    data_dir = get_qsimov_dataset_dir("cifar10")
    os.makedirs(data_dir, exist_ok=True)

    cache = os.path.join(data_dir, "_torchvision_cache")
    train_ds = torchvision.datasets.CIFAR10(root=cache, train=True,  download=True)
    test_ds  = torchvision.datasets.CIFAR10(root=cache, train=False, download=True)

    x_train = np.array(train_ds.data,  dtype=np.float32) / 255.0
    y_train = np.eye(10, dtype=np.float32)[np.array(train_ds.targets)]
    x_test  = np.array(test_ds.data,   dtype=np.float32) / 255.0
    y_test  = np.eye(10, dtype=np.float32)[np.array(test_ds.targets)]

    idx = np.arange(len(x_train))
    np.random.shuffle(idx)
    x_train, y_train = x_train[idx], y_train[idx]

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    np.save(os.path.join(data_dir, "x_train.npy"), x_train)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "x_test.npy"),  x_test)
    np.save(os.path.join(data_dir, "y_test.npy"),  y_test)


if __name__ == "__main__":
    main()
