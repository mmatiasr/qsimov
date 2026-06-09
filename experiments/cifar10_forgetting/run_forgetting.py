"""Class-incremental forgetting experiment on CIFAR-10.

Replicates mnist_forgetting on a harder dataset (32×32 RGB, 10 natural-image
classes). Base model is LeNet with linear output and MSE loss.
initial_layer=-1 → φ_R = Dense(32,relu) + Dense(10,linear) → 33 paths.

Methods:
  base              Reference: base model without any update.
  linear_accum      QsimovLinearSystem: fit(phase1) then fit(phase2) — no reset.
  linear_new_only   QsimovLinearSystem: fit(phase2) only — forgets phase1.
  gradient_new_only QsimovGradient: train W_Q on phase2 only.
  finetune_new_only Standard Adam fine-tune all weights on phase2.
  cumulative        Adam on phase1+phase2 combined — oracle upper bound.
"""

import os
import pickle
import time
import argparse
import multiprocessing as mp
import numpy as np

from experiments.path_utils import get_cifar10_forgetting_results_dir
from experiments.cifar10_forgetting.preprocess_data import load_data, get_data_dir
from experiments.cifar10_forgetting.train_base_model import accuracy

BATCH_SIZE = 256
QR_SHRINKAGE = 2
INITIAL_LAYER = -1


def make_imports():
    global tf, keras, KerasPathSelector, KerasQsimovLinearSystem, KerasQsimovGradient

    import tensorflow as tf
    from tensorflow import keras
    from qsimov.keras_path_selector import KerasPathSelector
    from qsimov.keras_qsimov_linear_system import KerasQsimovLinearSystem
    from qsimov.keras_qsimov_gradient import KerasQsimovGradient


def init_tf(processor):
    import tensorflow as tf
    if processor == "cpu":
        tf.config.set_visible_devices([], "GPU")


def np_accuracy(y_pred, y_true_onehot):
    return float(np.mean(
        np.argmax(y_pred, axis=1) == np.argmax(y_true_onehot, axis=1)
    ))


def eval_splits(y_pred_old, y_pred_new, y_pred_all, data, fit_time):
    return {
        "acc_old": np_accuracy(y_pred_old, data["test_phase1_y"]),
        "acc_new": np_accuracy(y_pred_new, data["test_phase2_y"]),
        "acc_all": np_accuracy(y_pred_all, data["test_y"]),
        "time":    fit_time,
    }


def save_results(method, results, results_dir):
    path = os.path.join(results_dir, f"{method}_results.pkl")
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(
        f"\n[{method}] acc_old={results['acc_old']:.4f}  "
        f"acc_new={results['acc_new']:.4f}  "
        f"acc_all={results['acc_all']:.4f}  "
        f"time={results['time']:.1f}s"
    )


def load_base(base_model_path):
    from tensorflow import keras
    return keras.models.load_model(
        base_model_path, custom_objects={"accuracy": accuracy}
    )


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def run_base(data, results_dir, base_model_path):
    model = load_base(base_model_path)
    kw = {"batch_size": BATCH_SIZE, "verbose": 0}
    results = eval_splits(
        model.predict(data["test_phase1_x"], **kw),
        model.predict(data["test_phase2_x"], **kw),
        model.predict(data["test_x"],        **kw),
        data, 0.0,
    )
    save_results("base", results, results_dir)


def run_qsimov_linear_accum(data, results_dir, base_model_path, device):
    ps = KerasPathSelector(
        load_base(base_model_path),
        initial_layer=INITIAL_LAYER,
        device=device,
        verbose=1,
    )
    qls = KerasQsimovLinearSystem(
        ps, solver="back_substitution", qr_shrinkage_factor=QR_SHRINKAGE, verbose=1
    )
    qls.fit(data["phase1_train_x"], data["phase1_train_y"], batch_size=BATCH_SIZE)

    t0 = time.time()
    qls.fit(data["phase2_train_x"], data["phase2_train_y"], batch_size=BATCH_SIZE)
    update_time = time.time() - t0

    kw = {"batch_size": BATCH_SIZE}
    results = eval_splits(
        qls.predict(data["test_phase1_x"], **kw),
        qls.predict(data["test_phase2_x"], **kw),
        qls.predict(data["test_x"],        **kw),
        data, update_time,
    )
    save_results("linear_accum", results, results_dir)


def run_qsimov_linear_new_only(data, results_dir, base_model_path, device):
    ps = KerasPathSelector(
        load_base(base_model_path),
        initial_layer=INITIAL_LAYER,
        device=device,
        verbose=1,
    )
    qls = KerasQsimovLinearSystem(
        ps, solver="back_substitution", qr_shrinkage_factor=QR_SHRINKAGE, verbose=1
    )
    t0 = time.time()
    qls.fit(data["phase2_train_x"], data["phase2_train_y"], batch_size=BATCH_SIZE)
    fit_time = time.time() - t0

    kw = {"batch_size": BATCH_SIZE}
    results = eval_splits(
        qls.predict(data["test_phase1_x"], **kw),
        qls.predict(data["test_phase2_x"], **kw),
        qls.predict(data["test_x"],        **kw),
        data, fit_time,
    )
    save_results("linear_new_only", results, results_dir)


def run_qsimov_gradient(data, results_dir, base_model_path, device, epochs):
    ps = KerasPathSelector(
        load_base(base_model_path),
        initial_layer=INITIAL_LAYER,
        device=device,
        verbose=1,
    )
    qg = KerasQsimovGradient(ps, verbose=1)
    qg.compile(loss="mse", optimizer="adam", metrics=[accuracy], device=device)

    t0 = time.time()
    qg.fit(
        data["phase2_train_x"], data["phase2_train_y"],
        batch_size=BATCH_SIZE, epochs=epochs,
    )
    fit_time = time.time() - t0

    results = eval_splits(
        qg.predict(data["test_phase1_x"]),
        qg.predict(data["test_phase2_x"]),
        qg.predict(data["test_x"]),
        data, fit_time,
    )
    save_results("gradient_new_only", results, results_dir)


def run_standard_finetune(data, results_dir, base_model_path, epochs):
    model = load_base(base_model_path)
    model.compile(loss="mse", optimizer="adam", metrics=[accuracy])

    t0 = time.time()
    model.fit(
        data["phase2_train_x"], data["phase2_train_y"],
        batch_size=BATCH_SIZE, epochs=epochs, verbose=1,
    )
    fit_time = time.time() - t0

    kw = {"batch_size": BATCH_SIZE, "verbose": 0}
    results = eval_splits(
        model.predict(data["test_phase1_x"], **kw),
        model.predict(data["test_phase2_x"], **kw),
        model.predict(data["test_x"],        **kw),
        data, fit_time,
    )
    save_results("finetune_new_only", results, results_dir)


def run_cumulative(data, results_dir, base_model_path, epochs):
    model = load_base(base_model_path)
    model.compile(loss="mse", optimizer="adam", metrics=[accuracy])

    all_x = np.concatenate([data["phase1_train_x"], data["phase2_train_x"]])
    all_y = np.concatenate([data["phase1_train_y"], data["phase2_train_y"]])
    idx = np.random.default_rng(42).permutation(len(all_x))
    all_x, all_y = all_x[idx], all_y[idx]

    t0 = time.time()
    model.fit(all_x, all_y, batch_size=BATCH_SIZE, epochs=epochs, verbose=1)
    fit_time = time.time() - t0

    kw = {"batch_size": BATCH_SIZE, "verbose": 0}
    results = eval_splits(
        model.predict(data["test_phase1_x"], **kw),
        model.predict(data["test_phase2_x"], **kw),
        model.predict(data["test_x"],        **kw),
        data, fit_time,
    )
    save_results("cumulative", results, results_dir)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def execute_logic(method, data, results_dir, base_model_path, args):
    device = "/gpu:0" if args.processor == "gpu" else "/cpu:0"

    if method == "base":
        run_base(data, results_dir, base_model_path)
    elif method == "linear_accum":
        run_qsimov_linear_accum(data, results_dir, base_model_path, device)
    elif method == "linear_new_only":
        run_qsimov_linear_new_only(data, results_dir, base_model_path, device)
    elif method == "gradient_new_only":
        run_qsimov_gradient(data, results_dir, base_model_path, device, args.epochs)
    elif method == "finetune_new_only":
        run_standard_finetune(data, results_dir, base_model_path, args.epochs)
    elif method == "cumulative":
        run_cumulative(data, results_dir, base_model_path, args.epochs)


def main(args):
    results_dir = get_cifar10_forgetting_results_dir("keras", args.processor)
    data_dir    = get_data_dir()
    base_model_path = os.path.join(results_dir, "base_model.tf")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(base_model_path):
        raise FileNotFoundError(
            f"Base model not found at {base_model_path}. "
            "Run train_base_model.py first."
        )

    def main_subprocess(method):
        make_imports()
        init_tf(args.processor)
        data = load_data(data_dir)
        execute_logic(method, data, results_dir, base_model_path, args)

    methods = [
        "base",
        "linear_accum",
        "linear_new_only",
        "gradient_new_only",
        "finetune_new_only",
        "cumulative",
    ]
    for method in methods:
        print(f"\n\nRunning: {method}\n")
        p = mp.Process(target=main_subprocess, args=(method,))
        p.start()
        p.join()


class RunForgettingParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="cpu")
        self.add_argument("--epochs", type=int, default=5)


if __name__ == "__main__":
    main(RunForgettingParser().parse_args())
