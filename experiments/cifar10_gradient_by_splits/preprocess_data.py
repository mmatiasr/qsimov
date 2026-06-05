# Import libraries

import numpy as np
import os
from experiments.path_utils import get_qsimov_dataset_dir


SEED = 42


# slow imports
def make_imports():
    global kr, to_categorical
    from tensorflow import keras as kr

    to_categorical = kr.utils.to_categorical


def get_train_test_data():
    # Load mnist dataset from tensorflow
    (x_train, y_train), (x_test, y_test) = kr.datasets.cifar10.load_data()

    # Preprocess labels to use one hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Normalize values of inputs to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Set the data type of the arrays to float32
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # Shuffle the data
    train_idxs = np.arange(len(x_train))
    np.random.shuffle(train_idxs)
    x_train = x_train[train_idxs]
    y_train = y_train[train_idxs]

    # Print the shape of the data
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    return x_train, y_train, x_test, y_test


def make_split(x_train_augmented, y_train_augmented, split):
    # Split by label to ensure that each split has the same number of samples
    # of each label
    x_train_augmented_by_label = []
    y_train_augmented_by_label = []
    for i in range(10):
        selected = y_train_augmented[:, i] == 1
        x_train_augmented_by_label.append(x_train_augmented[selected])
        y_train_augmented_by_label.append(y_train_augmented[selected])

    left_split_x, left_split_y = [], []
    for label_idx in range(10):
        split_by_label = split // 10
        if label_idx < split % 10:  # Add one more sample to the split
            split_by_label += 1

        left_split_x.append(
            x_train_augmented_by_label[label_idx][:split_by_label]
        )
        left_split_y.append(
            y_train_augmented_by_label[label_idx][:split_by_label]
        )

    # Concatenate the splits
    left_split_x = np.concatenate(left_split_x)
    left_split_y = np.concatenate(left_split_y)

    # Shuffle the splits
    train_idxs = np.arange(len(left_split_x))
    np.random.shuffle(train_idxs)
    left_split_x = left_split_x[train_idxs]
    left_split_y = left_split_y[train_idxs]

    return left_split_x, left_split_y


def load_dataset():
    data_dir = get_qsimov_dataset_dir("cifar10")
    return (
        np.load(os.path.join(data_dir, "x_train.npy")),
        np.load(os.path.join(data_dir, "y_train.npy")),
        np.load(os.path.join(data_dir, "x_test.npy")),
        np.load(os.path.join(data_dir, "y_test.npy")),
    )


def main():
    # Import libraries
    make_imports()

    # Set seed
    kr.utils.set_random_seed(SEED)

    # Define the directory of the files
    data_dir = get_qsimov_dataset_dir("cifar10")
    os.makedirs(data_dir, exist_ok=True)

    # Load the data
    x_train, y_train, x_test, y_test = get_train_test_data()

    # Save the train data
    np.save(os.path.join(data_dir, "x_train.npy"), x_train)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)

    # Save the test data
    np.save(os.path.join(data_dir, "x_test.npy"), x_test)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)


if __name__ == "__main__":
    main()
