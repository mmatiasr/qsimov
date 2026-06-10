# Import libraries

import numpy as np
import os
from experiments.path_utils import get_qsimov_dataset_dir
import tqdm


SEED = 42

# Each directory in the train.X* directories has 25 labels and 1300 images per
# label. The val.X directory has 50 images per label.
# All 4 dirs are needed so that NUM_LABELS=100 divides by N_ROUNDS=4 and N_BATCHES=20.
TRAIN_DIRS = ["train.X1", "train.X2", "train.X3", "train.X4"]
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


def get_train_test_data(imagenet_dir):
    """Load and resize imagenet images from raw directory to numpy arrays.

    Parameters
    ----------
    imagenet_dir : str
        Directory containing train.X1, train.X2, ..., val.X subdirectories.
    """
    from PIL import Image

    # Collect sorted label names from all train dirs
    used_labels = set()
    for train_dir in TRAIN_DIRS:
        d = os.path.join(imagenet_dir, train_dir)
        if os.path.isdir(d):
            used_labels.update(os.listdir(d))
    sorted_labels = sorted(used_labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(sorted_labels)}

    # Load train images
    x_list, y_list = [], []
    print("Loading train data...")
    for train_dir in TRAIN_DIRS:
        for label in tqdm.tqdm(sorted_labels, desc=train_dir):
            cls_dir = os.path.join(imagenet_dir, train_dir, label)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in sorted(os.listdir(cls_dir)):
                try:
                    img = Image.open(os.path.join(cls_dir, img_name)).convert("RGB")
                    img = img.resize(IMAGE_SIZE)
                    x_list.append(np.array(img, dtype=np.uint8))
                    y_list.append(label_to_idx[label])
                except Exception:
                    pass

    x_train = np.array(x_list, dtype=np.uint8)
    y_train = np.array(y_list, dtype=np.int16)

    # Load validation images
    xv_list, yv_list = [], []
    print("Loading val data...")
    val_dir = os.path.join(imagenet_dir, "val.X")
    for label in tqdm.tqdm(sorted_labels, desc="val"):
        cls_dir = os.path.join(val_dir, label)
        if not os.path.isdir(cls_dir):
            continue
        for img_name in sorted(os.listdir(cls_dir)):
            try:
                img = Image.open(os.path.join(cls_dir, img_name)).convert("RGB")
                img = img.resize(IMAGE_SIZE)
                xv_list.append(np.array(img, dtype=np.uint8))
                yv_list.append(label_to_idx[label])
            except Exception:
                pass

    x_test = np.array(xv_list, dtype=np.uint8)
    y_test = np.array(yv_list, dtype=np.int16)

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imagenet-dir",
        default=get_qsimov_dataset_dir("imagenet_subset"),
        help=(
            "Path to the raw ImageNet directory containing train.X1, "
            "train.X2, train.X3, train.X4, and val.X subdirectories. "
            "Defaults to data/imagenet_subset/ inside QSIMOV_HOME."
        ),
    )
    args = parser.parse_args()

    np.random.seed(SEED)

    data_dir = get_qsimov_dataset_dir("imagenet_subset")
    os.makedirs(data_dir, exist_ok=True)

    x_train, y_train, x_test, y_test = get_train_test_data(args.imagenet_dir)

    np.save(os.path.join(data_dir, "x_train.npy"), x_train)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "x_test.npy"), x_test)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)


if __name__ == "__main__":
    main()
