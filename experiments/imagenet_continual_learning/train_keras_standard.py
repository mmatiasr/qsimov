"""Standard fine-tuning baselines for the continual learning experiment.

Both baselines start from the path-selector model trained on ALL 100 classes.

standard_finetune
    The model is fine-tuned on each round's class group WITHOUT replay.
    After round 2 (classes 25-49), accuracy on round-1 val (classes 0-24)
    collapses → catastrophic forgetting.

standard_cumulative
    At each round, a fresh copy of the all-class model is fine-tuned on the
    CUMULATIVE training data (all groups seen so far).
    This is the oracle upper bound: it never forgets because it re-sees all data.
    Its training cost grows each round.
"""

import os
import pickle
import time
import argparse
import multiprocessing as mp
import numpy as np
from experiments.path_utils import get_imagenet_continual_learning_results_dir
from experiments.imagenet_continual_learning.preprocess_data import N_ROUNDS


def make_imports():
    global tf, keras, init_tensorflow
    global load_dataset, make_round_splits, NUM_LABELS
    global get_optimizer, load_model

    import tensorflow as tf
    from tensorflow import keras
    from experiments.mnist_speed_loss.tf_keras.utils import init_tensorflow
    from experiments.imagenet_continual_learning.preprocess_data import (
        load_dataset,
        make_round_splits,
        NUM_LABELS,
    )
    from experiments.imagenet_continual_learning.keras_model_factory import (
        get_optimizer,
        load_model,
    )


BATCH_SIZE = 64
EPOCHS_PER_ROUND = 5


def accuracy(y_pred, y_true_int):
    return float(np.mean(np.argmax(y_pred, axis=1) == y_true_int))


def evaluate(model, x, y_int, n_classes):
    y_pred = model.predict(x, batch_size=BATCH_SIZE)
    acc = accuracy(y_pred, y_int)
    eps = 1e-7
    y_clip = np.clip(y_pred, eps, 1 - eps)
    one_hot = np.eye(n_classes)[y_int.astype(int)]
    loss = float(-np.mean(np.sum(one_hot * np.log(y_clip), axis=1)))
    return {"accuracy": acc, "loss": loss}


def collect_per_round_metrics(model, rounds, current_k):
    per_round = {}
    for prev_k, (_, _, val_x, val_y) in enumerate(rounds[:current_k], 1):
        per_round[f"round_{prev_k}"] = evaluate(model, val_x, val_y, NUM_LABELS)
    return per_round


def compile_fresh(results_dir):
    """Load and compile the all-class model (starting point for fine-tuning)."""
    model = load_model(results_dir, "standard")
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=get_optimizer("vgg16"),
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Standard fine-tuning (sequential, no replay)
# ---------------------------------------------------------------------------

def execute_logic_finetune(results_dir, rounds, test_x, test_y):
    model = compile_fresh(results_dir)

    results = {}
    cumulative_time = 0.0

    for k, (train_x, train_y, _, _) in enumerate(rounds, 1):
        t0 = time.time()
        model.fit(
            x=train_x, y=train_y,
            epochs=EPOCHS_PER_ROUND, batch_size=BATCH_SIZE, verbose=1,
        )
        cumulative_time += time.time() - t0

        round_key = f"after_round_{k}"
        results[round_key] = {
            "time(s)": cumulative_time,
            "overall": evaluate(model, test_x, test_y, NUM_LABELS),
            "per_round_val": collect_per_round_metrics(model, rounds, k),
        }

    return results


# ---------------------------------------------------------------------------
# Cumulative retrain (oracle upper bound)
# ---------------------------------------------------------------------------

def execute_logic_cumulative(results_dir, rounds, test_x, test_y):
    results = {}
    cumulative_time = 0.0
    cum_x, cum_y = [], []

    for k, (train_x, train_y, _, _) in enumerate(rounds, 1):
        cum_x.append(train_x)
        cum_y.append(train_y)
        cx = np.concatenate(cum_x)
        cy = np.concatenate(cum_y)

        # Re-initialise from the all-class model to keep comparisons fair
        model = compile_fresh(results_dir)

        t0 = time.time()
        model.fit(
            x=cx, y=cy,
            epochs=EPOCHS_PER_ROUND, batch_size=BATCH_SIZE, verbose=1,
        )
        cumulative_time += time.time() - t0

        round_key = f"after_round_{k}"
        results[round_key] = {
            "time(s)": cumulative_time,
            "overall": evaluate(model, test_x, test_y, NUM_LABELS),
            "per_round_val": collect_per_round_metrics(model, rounds, k),
        }

        if k == N_ROUNDS:
            model.save(
                f"{results_dir}/standard_cumulative_final_model.tf",
                save_format="tf",
            )

    return results


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def execute_logic(method, results_dir, rounds, test_x, test_y):
    if method == "standard_finetune":
        results = execute_logic_finetune(results_dir, rounds, test_x, test_y)
    else:
        results = execute_logic_cumulative(results_dir, rounds, test_x, test_y)

    output_file = f"{results_dir}/{method}_results.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results for {method} to {output_file}")


def main(args):
    results_dir = get_imagenet_continual_learning_results_dir(args.processor)
    os.makedirs(results_dir, exist_ok=True)

    def main_subprocess(method):
        make_imports()
        init_tensorflow(tf, args.processor)
        x_train, y_train, x_test, y_test = load_dataset()
        rounds = make_round_splits(x_train, y_train, n_rounds=N_ROUNDS)
        execute_logic(method, results_dir, rounds, x_test, y_test)

    for method in ("standard_finetune", "standard_cumulative"):
        print(f"\n\nRunning standard continual learning: {method}\n")
        p = mp.Process(target=main_subprocess, args=(method,))
        p.start()
        p.join()


###############################################################################
# CLI Parser
###############################################################################


class TrainStandardParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("--processor", choices=["cpu", "gpu"], default="gpu")


if __name__ == "__main__":
    main(TrainStandardParser().parse_args())
