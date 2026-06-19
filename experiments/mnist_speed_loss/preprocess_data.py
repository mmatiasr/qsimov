# Import libraries
import numpy as np
import os
from experiments.path_utils import get_qsimov_home

# Library functions


def save_numpy_arrays(data_dir, train_x, train_y, test_x, test_y):
    # Split the train data into 4 equal parts and
    # save each part to a different file in npy format for train
    for i in range(4):
        lower = int((len(train_x) / 4) * i)
        upper = lower + len(train_x) // 4
        np.save(
            os.path.join(data_dir, f"train_x_{i}.npy"), train_x[lower:upper]
        )
        np.save(
            os.path.join(data_dir, f"train_y_{i}.npy"), train_y[lower:upper]
        )

    # For the test data no need to split
    np.save(os.path.join(data_dir, "test_x.npy"), test_x)
    np.save(os.path.join(data_dir, "test_y.npy"), test_y)


# Define a function that takes a prefix as an argument and
# returns a list of numpy arrays
def load_numpy_arrays(prefix):
    # Initialize an empty list to store the arrays
    arrays = []
    # Loop through the numbers 0 to 3
    for i in range(4):
        # Construct the file name using the prefix and the number
        file_name = f"{prefix}_{i}.npy"
        # Load the file using np.load
        array = np.load(file_name)
        # Append the array to the list
        arrays.append(array.astype(np.float32))
    # Return the list of arrays
    return arrays


def get_train_test_data(data_dir):
    train_xs = load_numpy_arrays(os.path.join(data_dir, "train_x"))
    train_ys = load_numpy_arrays(os.path.join(data_dir, "train_y"))
    test_x = np.load(os.path.join(data_dir, "test_x.npy")).astype(np.float32)
    test_y = np.load(os.path.join(data_dir, "test_y.npy")).astype(np.float32)
    return train_xs, train_ys, test_x, test_y


# concatenate batches on the first axis
def concat_batches(list_batches):
    return np.concatenate(list_batches, axis=0)


if __name__ == "__main__":
    # set random seed
    SEED = 42
    np.random.seed(SEED)

    # Define the directory of the files
    QSIMOV_HOME = get_qsimov_home()
    DATA_DIR = os.path.join(QSIMOV_HOME, "data/mnist")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load mnist dataset from torchvision (no TensorFlow required)
    import torchvision
    train_ds = torchvision.datasets.MNIST(root=DATA_DIR, train=True,  download=True)
    test_ds  = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True)

    x_train = np.array(train_ds.data,   dtype=np.float32)
    y_train = np.array(train_ds.targets, dtype=np.int64)
    x_test  = np.array(test_ds.data,    dtype=np.float32)
    y_test  = np.array(test_ds.targets,  dtype=np.int64)

    x_train = x_train.reshape(list(x_train.shape) + [1]).astype(np.float32)
    x_test  = x_test.reshape(list(x_test.shape)   + [1]).astype(np.float32)

    # One-hot encoding with numpy
    def one_hot(y, depth=10):
        out = np.zeros((len(y), depth), dtype=np.float32)
        out[np.arange(len(y)), y] = 1.0
        return out

    y_train = one_hot(y_train)
    y_test  = one_hot(y_test)

    # Normalize values of inputs to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Shuffle the data
    train_idxs = np.arange(len(x_train))
    np.random.shuffle(train_idxs)
    x_train = x_train[train_idxs]
    y_train = y_train[train_idxs]
    test_idxs = np.arange(len(x_test))
    np.random.shuffle(test_idxs)
    x_test = x_test[test_idxs]
    y_test = y_test[test_idxs]

    # Save arrays to differents files
    save_numpy_arrays(DATA_DIR, x_train, y_train, x_test, y_test)
