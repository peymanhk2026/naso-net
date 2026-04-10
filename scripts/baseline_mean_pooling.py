"""
L213 Baseline Comparison — ResNet50 + Mean Pooling (no WMV).

Addresses Reviewer Comment (L213):
  "Model comparisons are too narrow... Video baselines like 2D-CNN +
   LSTM/GRU, 1D-TCN on frame embeddings, 3D CNNs..."

This script trains a ResNet50 backbone with simple mean pooling of
per-frame sigmoid probabilities (no learned importance weights).
Uses identical settings to the ablation experiments:
  - 128×128 resolution, no augmentation (baseline config)
  - 10-fold patient-wise GroupKFold CV
  - 45-frame sliding window
  - 50 max epochs, early stopping (patience=15)
  - OneCycleLR, batch size 8
  - Patient-level bootstrap CIs (1000 iterations)

Usage:
    python scripts/baseline_mean_pooling.py
"""

import os
import sys
import csv
import math
import random
import time
import warnings
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, TimeDistributed, GlobalAveragePooling2D,
    BatchNormalization, Lambda,
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).parent))
from ablation_resolution_augmentation import (
    load_dataset,
    make_generator,
    predict_sequence_full,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "extracted_sequences"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "weights" / "baseline_comparisons"
TARGET_SIZE = 128
WINDOW = 45
EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-3
N_FOLDS = 10
SEED = 42
N_BOOT = 1000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class OneCycleLR(Callback):
    def __init__(self, max_lr, base_lr, epochs, steps_per_epoch, pct_start=0.3):
        super().__init__()
        self.max_lr = max_lr
        self.base_lr = base_lr
        self.total_steps = epochs * steps_per_epoch
        self.pct_start = pct_start
        self.step = 0

    def _get_lr(self):
        warmup_steps = self.pct_start * self.total_steps
        if self.step < warmup_steps:
            return self.base_lr + (self.max_lr - self.base_lr) * (self.step / warmup_steps)
        progress = (self.step - warmup_steps) / (self.total_steps - warmup_steps)
        return self.max_lr - (self.max_lr - self.base_lr) * progress

    def on_train_batch_begin(self, batch, logs=None):
        self.model.optimizer.learning_rate.assign(self._get_lr())
        self.step += 1


def build_resnet_mean_pooling(window_length, target_size, activation="gelu", lr=1e-3):
    inp = Input(shape=(window_length, target_size, target_size, 3))
    base = ResNet50(weights="imagenet", include_top=False,
                    input_shape=(target_size, target_size, 3))
    for layer in base.layers[:-20]:
        layer.trainable = False

    x = TimeDistributed(base)(inp)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = TimeDistributed(Dense(256, activation=activation))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.5))(x)

    p = TimeDistributed(Dense(1, activation="sigmoid"), name="frame_prob")(x)
    out = Lambda(lambda t: tf.reduce_mean(t, axis=1), name="mean_pool")(p)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def patient_bootstrap_ci(y_true, y_prob, y_pred, patient_ids, n_boot=1000):
    unique_patients = np.unique(patient_ids)
    patient_idx = {p: np.where(patient_ids == p)[0] for p in unique_patients}
    rng = np.random.default_rng(SEED)
    boot_accs, boot_aucs = [], []

    for _ in range(n_boot):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        idx = np.concatenate([patient_idx[p] for p in sampled])
        boot_accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        if len(np.unique(y_true[idx])) > 1:
            boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))

    acc_ci = (np.percentile(boot_accs, 2.5), np.percentile(boot_accs, 97.5))
    auc_ci = (
        (np.percentile(boot_aucs, 2.5), np.percentile(boot_aucs, 97.5))
        if boot_aucs else (0.0, 0.0)
    )
    return acc_ci, auc_ci


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    sequences, patients = load_dataset(str(DATA_DIR))
    labels = np.array([s["label"] for s in sequences])
    groups = np.array([s["patient"] for s in sequences])
    n_folds = min(N_FOLDS, len(patients))

    gkf = GroupKFold(n_splits=n_folds)
    steps_per_epoch = max(10, len(sequences) // (BATCH_SIZE * 2))

    all_y_true, all_y_prob, all_y_pred, all_patients = [], [], [], []
    fold_accs, fold_aucs = [], []
    t_start = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(labels, labels, groups)):
        fold_num = fold_idx + 1
        train_seqs = [sequences[i] for i in train_idx]
        test_seqs = [sequences[i] for i in test_idx]

        model = build_resnet_mean_pooling(WINDOW, TARGET_SIZE, "gelu", LR)

        train_gen = make_generator(
            train_seqs, WINDOW, TARGET_SIZE, BATCH_SIZE,
            aug_config=None, augment=False, use_resnet_preprocess=True,
        )
        val_gen = make_generator(
            test_seqs, WINDOW, TARGET_SIZE, BATCH_SIZE,
            aug_config=None, augment=False, use_resnet_preprocess=True,
        )
        val_steps = max(5, len(test_seqs) // BATCH_SIZE)

        checkpoint_path = str(OUTPUT_DIR / f"mean_pool_fold{fold_num}.weights.h5")

        callbacks = [
            OneCycleLR(max_lr=LR, base_lr=LR / 10,
                       epochs=EPOCHS, steps_per_epoch=steps_per_epoch),
            EarlyStopping(monitor="val_loss", patience=15,
                          restore_best_weights=True, verbose=0),
            ModelCheckpoint(checkpoint_path, monitor="val_auc", mode="max",
                            save_best_only=True, save_weights_only=True, verbose=0),
        ]

        model.fit(
            train_gen,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1,
        )

        if os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path)

        y_true, y_prob, seq_patients = [], [], []
        for seq in test_seqs:
            prob = predict_sequence_full(
                seq, model, WINDOW, TARGET_SIZE, use_resnet_preprocess=True,
            )
            y_true.append(seq["label"])
            y_prob.append(prob)
            seq_patients.append(seq["patient"])

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_true, y_pred)
        try:
            auc_val = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_val = float("nan")

        fold_accs.append(acc)
        if not math.isnan(auc_val):
            fold_aucs.append(auc_val)

        all_y_true.extend(y_true.tolist())
        all_y_prob.extend(y_prob.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_patients.extend(seq_patients)

        del model
        tf.keras.backend.clear_session()

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = np.array(all_y_pred)
    all_patients = np.array(all_patients)

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_auc = roc_auc_score(all_y_true, all_y_prob)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, average="binary", zero_division=0
    )
    acc_ci, auc_ci = patient_bootstrap_ci(
        all_y_true, all_y_prob, all_y_pred, all_patients, N_BOOT
    )
    t_elapsed = time.time() - t_start

    csv_path = OUTPUT_DIR / "baseline_results.csv"
    fieldnames = [
        "model", "resolution", "augmentation",
        "accuracy", "acc_ci_low", "acc_ci_high",
        "auc", "auc_ci_low", "auc_ci_high",
        "precision", "recall", "f1",
        "mean_fold_acc", "std_fold_acc",
        "mean_fold_auc", "std_fold_auc",
        "n_folds", "time_sec",
    ]
    row = {
        "model": "ResNet50+MeanPool",
        "resolution": TARGET_SIZE,
        "augmentation": "none",
        "accuracy": overall_acc,
        "acc_ci_low": acc_ci[0],
        "acc_ci_high": acc_ci[1],
        "auc": overall_auc,
        "auc_ci_low": auc_ci[0],
        "auc_ci_high": auc_ci[1],
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mean_fold_acc": np.mean(fold_accs),
        "std_fold_acc": np.std(fold_accs),
        "mean_fold_auc": np.mean(fold_aucs) if fold_aucs else float("nan"),
        "std_fold_auc": np.std(fold_aucs) if fold_aucs else float("nan"),
        "n_folds": n_folds,
        "time_sec": t_elapsed,
    }

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
