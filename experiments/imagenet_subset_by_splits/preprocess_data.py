# Import libraries

import numpy as np
import os
from experiments.path_utils import get_qsimov_dataset_dir
import tqdm
import tempfile


SEED = 42

# Each directory in the train.X* directories has 25 labels and 1300 images per
# label. The val.X directory has 50 images per label.
TRAIN_DIRS = ["train.X1", "train.X2"]  # , "train.X3", "train.X4"]
NUM_LABELS = 25 * len(TRAIN_DIRS)

TRAIN_IMAGES_PER_LABEL = 1300
VAL_IMAGES_PER_LABEL = 50
NUM_TRAIN_SAMPLES = TRAIN_IMAGES_PER_LABEL * NUM_LABELS
NUM_VAL_SAMPLES = VAL_IMAGES_PER_LABEL * NUM_LABELS
IMAGE_SIZE = (224, 224)


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def sizeof_numpy_array(arr):
    return sizeof_fmt(arr.size * arr.itemsize)


# slow imports
def make_imports():
    global kr, tf
    from tensorflow import keras as kr
    import tensorflow as tf


def get_train_test_data(data_dir):
    # make a temporary directory with soft links to training images that
    # belong to the labels in the train_dirs
    tmp_dir = tempfile.mkdtemp()
    os.makedirs(tmp_dir, exist_ok=True)
    used_labels = set()  # set of labels in the training set

    # create soft links
    for train_dir in TRAIN_DIRS:
        for label in os.listdir(os.path.join(data_dir, train_dir)):
            used_labels.add(label)
            os.symlink(
                os.path.join(data_dir, train_dir, label),
                os.path.join(tmp_dir, label),
            )

    # Create the train dataset using the temporary directory with soft links
    dataset_train = kr.utils.image_dataset_from_directory(
        tmp_dir, image_size=IMAGE_SIZE, batch_size=32
    ).as_numpy_iterator()

    # Create the train arrays
    x_train = np.empty((NUM_TRAIN_SAMPLES, *IMAGE_SIZE, 3), dtype=np.uint8)
    y_train = np.zeros((NUM_TRAIN_SAMPLES,), dtype=np.int16)

    print("Loading the data...")
    # Fill the arrays
    idx = 0
    for images, labels in tqdm.tqdm(dataset_train):
        batch_size = images.shape[0]
        x_train[idx : idx + batch_size] = images
        y_train[idx : idx + batch_size] = labels
        idx += batch_size

    # Create the test arrays
    y_test = np.zeros((NUM_VAL_SAMPLES,), dtype=np.int16)
    x_test = np.empty((NUM_VAL_SAMPLES, *IMAGE_SIZE, 3), dtype=np.uint8)

    # make a temporary directory with soft links to validation images that
    # belong to the labels in the training set
    tmp_dir = tempfile.mkdtemp()
    test_dir = os.path.join(data_dir, "val.X")
    os.makedirs(tmp_dir, exist_ok=True)

    # create soft links
    for label in used_labels:
        os.symlink(os.path.join(test_dir, label), os.path.join(tmp_dir, label))

    dataset_test = kr.utils.image_dataset_from_directory(
        tmp_dir, image_size=(224, 224), batch_size=None
    ).as_numpy_iterator()

    idx = 0
    for image, label in tqdm.tqdm(dataset_test):
        x_test[idx] = image
        y_test[idx] = label
        idx += 1

    # Print the shape of the data
    print("x_train shape:", x_train.shape, sizeof_numpy_array(x_train))
    print("y_train shape:", y_train.shape, sizeof_numpy_array(y_train))
    print("x_test shape:", x_test.shape, sizeof_numpy_array(x_test))
    print("y_test shape:", y_test.shape, sizeof_numpy_array(y_test))

    return x_train, y_train, x_test, y_test


def make_split_indices(y_train, split):
    # Split by label to ensure that each split has the same number of samples
    # of each label
    indices_by_label = []
    for i in range(NUM_LABELS):
        indices_by_label.append(np.where(y_train == i)[0])

    left_split_indices = []
    for label_idx in range(NUM_LABELS):
        split_by_label = split // NUM_LABELS
        if label_idx < split % NUM_LABELS:  # Add one more sample to the split
            split_by_label += 1

        left_split_indices.append(indices_by_label[label_idx][:split_by_label])

    # Concatenate the splits
    left_split_indices = np.concatenate(left_split_indices)

    # Shuffle the splits
    np.random.shuffle(left_split_indices)

    return left_split_indices


def make_split(x_train, y_train, split):
    indices = make_split_indices(y_train, split)
    return x_train[indices], y_train[indices]


def load_dataset():
    data_dir = get_qsimov_dataset_dir("imagenet_subset")
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
    data_dir = get_qsimov_dataset_dir("imagenet_subset")
    os.makedirs(data_dir, exist_ok=True)

    # Load the data
    x_train, y_train, x_test, y_test = get_train_test_data(data_dir)

    # Save the train data
    np.save(os.path.join(data_dir, "x_train.npy"), x_train)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)

    # Save the test data
    np.save(os.path.join(data_dir, "x_test.npy"), x_test)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)


if __name__ == "__main__":
    main()
