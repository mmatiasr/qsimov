# Import libraries

import numpy as np
import os
from experiments.path_utils import get_qsimov_dataset_dir
import tqdm


SEED = 42

# 100 classes: NUM_LABELS must divide N_ROUNDS=4 and N_BATCHES=20.
# Legacy split format: 25 labels per train.X* dir × 4 dirs = 100.
# Flat ILSVRC format: first 100 synsets alphabetically from train/.
TRAIN_DIRS = ["train.X1", "train.X2", "train.X3", "train.X4"]
NUM_LABELS = 25 * len(TRAIN_DIRS)  # 100

TRAIN_IMAGES_PER_LABEL = 1300
VAL_IMAGES_PER_LABEL = 50
NUM_TRAIN_SAMPLES = TRAIN_IMAGES_PER_LABEL * NUM_LABELS
NUM_VAL_SAMPLES = VAL_IMAGES_PER_LABEL * NUM_LABELS

# 64×64: used for both npz loading AND PIL resizing so the stored arrays
# always have the same spatial size regardless of loading mode.
IMAGE_SIZE = (64, 64)

# Convenience constant imported by PyTorch training scripts (NCHW).
INPUT_SHAPE_NCHW = (3, IMAGE_SIZE[0], IMAGE_SIZE[1])


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def sizeof_numpy_array(arr):
    return sizeof_fmt(arr.size * arr.itemsize)


# ---------------------------------------------------------------------------
# NPZ format (downsampled 64×64 ImageNet from image-net.org)
# ---------------------------------------------------------------------------

def _load_npz_array(path):
    """Load one npz file, returning (x_nhwc_uint8, y_int0indexed)."""
    d = np.load(path, allow_pickle=False)

    # --- locate image array (try common key names) ---
    for key in ("data", "x", "images"):
        if key in d:
            x_raw = d[key]
            break
    else:
        raise KeyError(f"Cannot find image array in {path}. Keys: {list(d.keys())}")

    # --- locate label array ---
    for key in ("labels", "y", "fine_labels"):
        if key in d:
            y_raw = d[key].astype(np.int32).ravel()
            break
    else:
        raise KeyError(f"Cannot find label array in {path}. Keys: {list(d.keys())}")

    # --- reshape image array to NHWC uint8 ---
    n = len(y_raw)
    if x_raw.ndim == 2:
        # Flat (N, C*H*W) — most common for downsampled ImageNet
        chw = x_raw.shape[1]
        h = w = int(round((chw / 3) ** 0.5))
        x_raw = x_raw.reshape(n, 3, h, w)          # → NCHW
    if x_raw.ndim == 4 and x_raw.shape[1] == 3:
        x_raw = x_raw.transpose(0, 2, 3, 1)        # NCHW → NHWC

    x_uint8 = x_raw.astype(np.uint8)

    # Labels may be 1-indexed (ILSVRC convention) or 0-indexed
    if y_raw.min() >= 1:
        y_raw = y_raw - 1  # → 0-indexed

    return x_uint8, y_raw


def get_train_test_data_npz(npz_dir):
    """Load 64×64 downsampled ImageNet from npz files.

    Expected files in npz_dir (any naming containing 'train'/'val'):
        Imagenet64_train_part1.npz  (~6 GB)
        Imagenet64_train_part2.npz  (~6 GB)
        Imagenet64_val.npz          (~500 MB)

    Selects the first NUM_LABELS=100 classes (indices 0-99).
    """
    files = sorted(os.listdir(npz_dir))
    train_files = [f for f in files if f.endswith(".npz") and "train" in f.lower()]
    val_files   = [f for f in files if f.endswith(".npz") and "val"   in f.lower()]

    if not train_files:
        raise FileNotFoundError(
            f"No npz train files found in '{npz_dir}'. "
            "Expected files containing 'train' in the name."
        )
    if not val_files:
        raise FileNotFoundError(
            f"No npz val files found in '{npz_dir}'. "
            "Expected a file containing 'val' in the name."
        )

    print(f"Train npz files: {train_files}")
    print(f"Val   npz files: {val_files}")

    # Load and concatenate training data
    train_x_parts, train_y_parts = [], []
    for fname in train_files:
        print(f"Loading {fname}…")
        x, y = _load_npz_array(os.path.join(npz_dir, fname))
        train_x_parts.append(x)
        train_y_parts.append(y)
    x_all = np.concatenate(train_x_parts)
    y_all = np.concatenate(train_y_parts)

    # Load validation
    print(f"Loading {val_files[0]}…")
    x_val_all, y_val_all = _load_npz_array(os.path.join(npz_dir, val_files[0]))

    # Subset to first NUM_LABELS classes
    train_mask = y_all     < NUM_LABELS
    val_mask   = y_val_all < NUM_LABELS
    x_train = x_all[train_mask]
    y_train = y_all[train_mask].astype(np.int16)
    x_test  = x_val_all[val_mask]
    y_test  = y_val_all[val_mask].astype(np.int16)

    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _detect_format(imagenet_dir):
    """Return 'split' (train.X1..X4 / val.X) or 'flat' (train/ / val/)."""
    if os.path.isdir(os.path.join(imagenet_dir, "train.X1")):
        return "split"
    if os.path.isdir(os.path.join(imagenet_dir, "train")):
        return "flat"
    raise FileNotFoundError(
        f"No ImageNet data found in '{imagenet_dir}'.\n"
        "Expected either:\n"
        "  train.X1/, train.X2/, train.X3/, train.X4/, val.X/  (legacy format)\n"
        "or:\n"
        "  train/<synset>/, val/<synset>/                        (ILSVRC flat format)"
    )


# ---------------------------------------------------------------------------
# Common image loader
# ---------------------------------------------------------------------------

def _load_images_from_dir(cls_dir, label_idx, img_list, lbl_list):
    from PIL import Image
    for img_name in sorted(os.listdir(cls_dir)):
        img_path = os.path.join(cls_dir, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMAGE_SIZE)
            img_list.append(np.array(img, dtype=np.uint8))
            lbl_list.append(label_idx)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Legacy split format (train.X1 / val.X)
# ---------------------------------------------------------------------------

def _get_train_test_data_split(imagenet_dir):
    from PIL import Image

    used_labels = set()
    for train_dir in TRAIN_DIRS:
        d = os.path.join(imagenet_dir, train_dir)
        if os.path.isdir(d):
            used_labels.update(os.listdir(d))
    sorted_labels = sorted(used_labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(sorted_labels)}

    x_list, y_list = [], []
    print("Loading train data (split format)...")
    for train_dir in TRAIN_DIRS:
        for label in tqdm.tqdm(sorted_labels, desc=train_dir):
            cls_dir = os.path.join(imagenet_dir, train_dir, label)
            if not os.path.isdir(cls_dir):
                continue
            _load_images_from_dir(cls_dir, label_to_idx[label], x_list, y_list)

    x_train = np.array(x_list, dtype=np.uint8)
    y_train = np.array(y_list, dtype=np.int16)

    xv_list, yv_list = [], []
    print("Loading val data (split format)...")
    val_dir = os.path.join(imagenet_dir, "val.X")
    for label in tqdm.tqdm(sorted_labels, desc="val"):
        cls_dir = os.path.join(val_dir, label)
        if not os.path.isdir(cls_dir):
            continue
        _load_images_from_dir(cls_dir, label_to_idx[label], xv_list, yv_list)

    x_test = np.array(xv_list, dtype=np.uint8)
    y_test = np.array(yv_list, dtype=np.int16)
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Flat ILSVRC format (train/<synset>/ + val/<synset>/)
# ---------------------------------------------------------------------------

def _get_train_test_data_flat(imagenet_dir):
    """Load from standard ILSVRC flat format.

    Expects:
        <imagenet_dir>/train/<synset>/*.JPEG   (extracted per-class tars)
        <imagenet_dir>/val/<synset>/*.JPEG     (reorganised by synset)

    If val/ contains flat numbered images instead of synset subdirectories,
    run with --reorganize-val first.

    Uses the first NUM_LABELS=100 synsets alphabetically from train/.
    """
    train_root = os.path.join(imagenet_dir, "train")
    val_root   = os.path.join(imagenet_dir, "val")

    # Collect synsets from train/
    all_synsets = sorted([
        d for d in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, d))
    ])
    if len(all_synsets) < NUM_LABELS:
        raise ValueError(
            f"Only {len(all_synsets)} synsets found in {train_root}, "
            f"need at least {NUM_LABELS}."
        )
    selected = all_synsets[:NUM_LABELS]
    label_to_idx = {s: i for i, s in enumerate(selected)}
    print(f"Using {NUM_LABELS} synsets: {selected[0]} … {selected[-1]}")

    # Validate that val/ is organised by synset
    val_entries = os.listdir(val_root)
    val_has_synset_dirs = any(
        os.path.isdir(os.path.join(val_root, e)) and e.startswith("n")
        for e in val_entries
    )
    if not val_has_synset_dirs:
        raise FileNotFoundError(
            f"val/ directory at '{val_root}' does not contain synset subdirectories.\n"
            "The ILSVRC validation images are flat (numbered files) by default.\n"
            "Reorganise them first with:\n\n"
            "  python3 experiments/imagenet_subset_by_splits/preprocess_data.py \\\n"
            "      --imagenet-dir <path> --reorganize-val\n"
        )

    x_list, y_list = [], []
    print("Loading train data (flat format)...")
    for synset in tqdm.tqdm(selected, desc="train"):
        cls_dir = os.path.join(train_root, synset)
        _load_images_from_dir(cls_dir, label_to_idx[synset], x_list, y_list)

    x_train = np.array(x_list, dtype=np.uint8)
    y_train = np.array(y_list, dtype=np.int16)

    xv_list, yv_list = [], []
    print("Loading val data (flat format)...")
    for synset in tqdm.tqdm(selected, desc="val"):
        cls_dir = os.path.join(val_root, synset)
        if not os.path.isdir(cls_dir):
            continue
        _load_images_from_dir(cls_dir, label_to_idx[synset], xv_list, yv_list)

    x_test = np.array(xv_list, dtype=np.uint8)
    y_test = np.array(yv_list, dtype=np.int16)
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Val reorganisation (flat ILSVRC → synset subdirs)
# ---------------------------------------------------------------------------

def _reorganize_val(imagenet_dir):
    """Reorganise flat ILSVRC val images into per-synset subdirectories.

    Reads the ground-truth labels from the devkit and moves (not copies)
    each ILSVRC2012_val_*.JPEG into val/<synset>/.

    Requires:
        <imagenet_dir>/val/ILSVRC2012_val_*.JPEG   (flat numbered images)
        <imagenet_dir>/ILSVRC2012_devkit_t12.tar.gz  OR
        <imagenet_dir>/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt
        <imagenet_dir>/ILSVRC2012_devkit_t12/data/meta.mat
    """
    import shutil

    val_root = os.path.join(imagenet_dir, "val")

    # Locate devkit ground truth
    gt_file = os.path.join(
        imagenet_dir,
        "ILSVRC2012_devkit_t12", "data",
        "ILSVRC2012_validation_ground_truth.txt",
    )
    if not os.path.isfile(gt_file):
        # Try extracting devkit tar
        devkit_tar = os.path.join(imagenet_dir, "ILSVRC2012_devkit_t12.tar.gz")
        if os.path.isfile(devkit_tar):
            print(f"Extracting devkit from {devkit_tar}…")
            import tarfile
            with tarfile.open(devkit_tar, "r:gz") as t:
                t.extractall(imagenet_dir)
        if not os.path.isfile(gt_file):
            raise FileNotFoundError(
                f"Cannot find devkit ground truth at:\n  {gt_file}\n"
                "Download ILSVRC2012_devkit_t12.tar.gz and place it in:\n"
                f"  {imagenet_dir}/"
            )

    # Read ground-truth labels (1-indexed ILSVRC class IDs, 50 000 entries)
    with open(gt_file) as f:
        gt_labels = [int(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(gt_labels)} ground-truth val labels.")

    # Read synset order from meta.mat using scipy (maps ILSVRC_ID → synset)
    try:
        import scipy.io
        meta = scipy.io.loadmat(
            os.path.join(imagenet_dir, "ILSVRC2012_devkit_t12", "data", "meta.mat")
        )
        # synsets field: structured array, WNID is column 1
        wnids = [str(meta["synsets"][i][0][1][0]) for i in range(1000)]
    except Exception as e:
        raise RuntimeError(
            f"Could not read meta.mat ({e}).\n"
            "Install scipy: pip install scipy"
        ) from e

    # Sorted val images (ILSVRC2012_val_00000001.JPEG, etc.)
    val_images = sorted([
        f for f in os.listdir(val_root)
        if f.lower().endswith(".jpeg") or f.lower().endswith(".jpg")
    ])
    if len(val_images) != len(gt_labels):
        raise ValueError(
            f"Found {len(val_images)} val images but {len(gt_labels)} ground-truth labels."
        )

    print(f"Reorganising {len(val_images)} val images into synset subdirectories…")
    moved = 0
    for img_name, label_1idx in tqdm.tqdm(
        zip(val_images, gt_labels), total=len(val_images), desc="reorganize val"
    ):
        synset = wnids[label_1idx - 1]
        dst_dir = os.path.join(val_root, synset)
        os.makedirs(dst_dir, exist_ok=True)
        src = os.path.join(val_root, img_name)
        dst = os.path.join(dst_dir, img_name)
        shutil.move(src, dst)
        moved += 1

    print(f"Done. Moved {moved} images into {len(set(wnids))} synset directories.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_train_test_data(imagenet_dir):
    fmt = _detect_format(imagenet_dir)
    print(f"Detected ImageNet format: {fmt}")
    if fmt == "split":
        return _get_train_test_data_split(imagenet_dir)
    return _get_train_test_data_flat(imagenet_dir)


def make_split_indices(y_train, split):
    indices_by_label = []
    for i in range(NUM_LABELS):
        indices_by_label.append(np.where(y_train == i)[0])

    left_split_indices = []
    for label_idx in range(NUM_LABELS):
        split_by_label = split // NUM_LABELS
        if label_idx < split % NUM_LABELS:
            split_by_label += 1
        left_split_indices.append(indices_by_label[label_idx][:split_by_label])

    left_split_indices = np.concatenate(left_split_indices)
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
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess ILSVRC ImageNet data into numpy arrays for the qsimov experiments.\n\n"
            "Supported input formats:\n"
            "  Legacy split:  <dir>/train.X1/, train.X2/, train.X3/, train.X4/, val.X/\n"
            "  ILSVRC flat:   <dir>/train/<synset>/, val/<synset>/\n\n"
            "If val/ still contains flat numbered images (not yet organised by synset),\n"
            "run with --reorganize-val first, then re-run without it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--imagenet-dir",
        default=get_qsimov_dataset_dir("imagenet_subset"),
        help=(
            "Path to the raw ImageNet directory. "
            "Defaults to data/imagenet_subset/ inside QSIMOV_HOME."
        ),
    )
    parser.add_argument(
        "--npz-dir",
        default=None,
        help=(
            "Path to directory containing the 64×64 downsampled ImageNet npz files "
            "(Imagenet64_train_part1.npz, Imagenet64_train_part2.npz, Imagenet64_val.npz). "
            "When provided, --imagenet-dir and --reorganize-val are ignored."
        ),
    )
    parser.add_argument(
        "--reorganize-val",
        action="store_true",
        help=(
            "Reorganise flat ILSVRC val images (ILSVRC2012_val_*.JPEG) into "
            "per-synset subdirectories using the devkit ground truth. "
            "Requires ILSVRC2012_devkit_t12.tar.gz in --imagenet-dir and scipy."
        ),
    )
    args = parser.parse_args()

    if args.npz_dir:
        np.random.seed(SEED)
        data_dir = get_qsimov_dataset_dir("imagenet_subset")
        os.makedirs(data_dir, exist_ok=True)
        x_train, y_train, x_test, y_test = get_train_test_data_npz(args.npz_dir)
        print("x_train shape:", x_train.shape, sizeof_numpy_array(x_train))
        print("y_train shape:", y_train.shape, sizeof_numpy_array(y_train))
        print("x_test  shape:", x_test.shape,  sizeof_numpy_array(x_test))
        print("y_test  shape:", y_test.shape,  sizeof_numpy_array(y_test))
        np.save(os.path.join(data_dir, "x_train.npy"), x_train)
        np.save(os.path.join(data_dir, "y_train.npy"), y_train)
        np.save(os.path.join(data_dir, "x_test.npy"),  x_test)
        np.save(os.path.join(data_dir, "y_test.npy"),  y_test)
        print(f"\nSaved numpy arrays to {data_dir}")
        return

    if args.reorganize_val:
        _reorganize_val(args.imagenet_dir)
        print("\nVal reorganised. Now run without --reorganize-val to preprocess.")
        return

    np.random.seed(SEED)

    data_dir = get_qsimov_dataset_dir("imagenet_subset")
    os.makedirs(data_dir, exist_ok=True)

    x_train, y_train, x_test, y_test = get_train_test_data(args.imagenet_dir)

    print("x_train shape:", x_train.shape, sizeof_numpy_array(x_train))
    print("y_train shape:", y_train.shape, sizeof_numpy_array(y_train))
    print("x_test  shape:", x_test.shape,  sizeof_numpy_array(x_test))
    print("y_test  shape:", y_test.shape,  sizeof_numpy_array(y_test))

    np.save(os.path.join(data_dir, "x_train.npy"), x_train)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "x_test.npy"),  x_test)
    np.save(os.path.join(data_dir, "y_test.npy"),  y_test)
    print(f"\nSaved numpy arrays to {data_dir}")


if __name__ == "__main__":
    main()
