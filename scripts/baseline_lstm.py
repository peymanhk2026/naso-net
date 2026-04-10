"""
L213 Baseline Comparison — ResNet50 + LSTM.

Addresses Reviewer Comment (L213):
  "Model comparisons are too narrow... Video baselines like 2D-CNN +
   LSTM/GRU..."

This script trains a ResNet50 backbone with an LSTM temporal head
instead of WMV.  Uses identical settings to the ablation experiments:
  - 128×128 resolution, no augmentation (baseline config)
  - 10-fold patient-wise GroupKFold CV
  - 45-frame sliding window
  - 50 max epochs, early stopping (patience=15)
  - OneCycleLR, batch size 8
  - Patient-level bootstrap CIs (1000 iterations)

Usage:
    & "D:\PLOS ONE\vpi_env\Scripts\python.exe" L213_baseline_lstm.py
"""

import os
import sys
import csv
import math
import random
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, TimeDistributed, GlobalAveragePooling2D,
    BatchNormalization, LSTM,
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).parent))
from ablation_resolution_augmentation import (
    load_dataset,
    load_and_preprocess_sequence,
    make_generator,
    predict_sequence_full,
)

# ── Config (identical to ablation baseline) ───────────────
DATA_DIR    = r"D:\PLOS ONE\VPI case videos\extracted_sequences"
OUTPUT_DIR  = Path(r"D:\PLOS ONE\naso_net_results\baseline_comparisons")
TARGET_SIZE = 128
WINDOW      = 45
EPOCHS      = 50
BATCH_SIZE  = 8
LR          = 1e-3
N_FOLDS     = 10
SEED        = 42
N_BOOT      = 1000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── OneCycleLR (same as ablation) ─────────────────────────
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
        else:
            progress = (self.step - warmup_steps) / (self.total_steps - warmup_steps)
            return self.max_lr - (self.max_lr - self.base_lr) * progress

    def on_train_batch_begin(self, batch, logs=None):
        self.model.optimizer.learning_rate.assign(self._get_lr())
        self.step += 1


# ── Model: ResNet50 + LSTM ────────────────────────────────
def build_resnet_lstm(window_length, target_size, activation="gelu", lr=1e-3):
    """
    Same ResNet50 backbone and dense head, but replaces WMV with
    an LSTM layer for temporal modeling followed by a sigmoid output.
    """
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

    # LSTM temporal head — takes the (batch, 45, 256) frame embeddings
    # and outputs a single sequence-level representation
    x = LSTM(128, dropout=0.3, recurrent_dropout=0.1)(x)
    x = Dense(64, activation=activation)(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


# ── Patient-level bootstrap CI ────────────────────────────
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


# ═════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════
print("=" * 65)
print("  L213: ResNet50 + LSTM Baseline")
print(f"  Config: {TARGET_SIZE}×{TARGET_SIZE}, no augmentation, {N_FOLDS}-fold CV")
print(f"  {EPOCHS} max epochs, early stopping patience=15")
print("=" * 65)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("\nLoading dataset...")
sequences, patients = load_dataset(DATA_DIR)
labels = np.array([s["label"] for s in sequences])
groups = np.array([s["patient"] for s in sequences])
n_folds = min(N_FOLDS, len(patients))

gkf = GroupKFold(n_splits=n_folds)
steps_per_epoch = max(10, len(sequences) // (BATCH_SIZE * 2))

all_y_true = []
all_y_prob = []
all_y_pred = []
all_patients = []
fold_accs, fold_aucs = [], []

t_start = time.time()

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(labels, labels, groups)):
    fold_num = fold_idx + 1
    train_seqs = [sequences[i] for i in train_idx]
    test_seqs  = [sequences[i] for i in test_idx]

    train_pats = sorted(set(s["patient"] for s in train_seqs))
    test_pats  = sorted(set(s["patient"] for s in test_seqs))

    print(f"\n  Fold {fold_num}/{n_folds}: "
          f"train={len(train_seqs)} seqs ({len(train_pats)} pts), "
          f"test={len(test_seqs)} seqs ({len(test_pats)} pts): "
          f"{', '.join(test_pats)}")

    # Build model
    model = build_resnet_lstm(WINDOW, TARGET_SIZE, "gelu", LR)

    # Generators (no augmentation)
    train_gen = make_generator(
        train_seqs, WINDOW, TARGET_SIZE, BATCH_SIZE,
        aug_config=None, augment=False, use_resnet_preprocess=True,
    )
    val_gen = make_generator(
        test_seqs, WINDOW, TARGET_SIZE, BATCH_SIZE,
        aug_config=None, augment=False, use_resnet_preprocess=True,
    )
    val_steps = max(5, len(test_seqs) // BATCH_SIZE)

    checkpoint_path = str(
        OUTPUT_DIR / f"lstm_fold{fold_num}.weights.h5"
    )
    periodic_path = str(
        OUTPUT_DIR / f"lstm_fold{fold_num}_epoch{{epoch:02d}}.weights.h5"
    )

    callbacks = [
        OneCycleLR(max_lr=LR, base_lr=LR / 10,
                   epochs=EPOCHS, steps_per_epoch=steps_per_epoch),
        EarlyStopping(monitor="val_loss", patience=15,
                      restore_best_weights=True, verbose=0),
        ModelCheckpoint(checkpoint_path, monitor="val_auc", mode="max",
                        save_best_only=True, save_weights_only=True, verbose=0),
        ModelCheckpoint(periodic_path, save_weights_only=True,
                        save_freq=5 * steps_per_epoch, verbose=0),
    ]

    model.fit(
        train_gen, epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen, validation_steps=val_steps,
        callbacks=callbacks, verbose=1,
    )

    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)

    # Evaluate on test sequences
    y_true, y_prob = [], []
    seq_patients = []
    for s in test_seqs:
        prob = predict_sequence_full(
            s, model, WINDOW, TARGET_SIZE, use_resnet_preprocess=True,
        )
        y_true.append(s["label"])
        y_prob.append(prob)
        seq_patients.append(s["patient"])

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

    print(f"    -> Acc={acc:.4f}, AUC={auc_val:.4f}")

    del model
    tf.keras.backend.clear_session()

t_elapsed = time.time() - t_start

# ── Aggregate ─────────────────────────────────────────────
all_y_true = np.array(all_y_true)
all_y_prob = np.array(all_y_prob)
all_y_pred = np.array(all_y_pred)
all_patients = np.array(all_patients)

overall_acc = accuracy_score(all_y_true, all_y_pred)
try:
    overall_auc = roc_auc_score(all_y_true, all_y_prob)
except ValueError:
    overall_auc = float("nan")
prec, rec, f1, _ = precision_recall_fscore_support(
    all_y_true, all_y_pred, average="binary", zero_division=0
)

acc_ci, auc_ci = patient_bootstrap_ci(
    all_y_true, all_y_prob, all_y_pred, all_patients, N_BOOT,
)

# ── Print results ─────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"  ResNet50 + LSTM — {n_folds}-fold patient-wise CV")
print(f"{'=' * 65}")
print(f"  Accuracy : {overall_acc:.4f} [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]")
print(f"  AUC      : {overall_auc:.4f} [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1       : {f1:.4f}")
print(f"  Mean Fold Acc: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
print(f"  Mean Fold AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"  Time: {t_elapsed:.0f}s ({t_elapsed/3600:.1f}h)")

# ── Save to CSV ───────────────────────────────────────────
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
    "model":         "ResNet50+LSTM",
    "resolution":    TARGET_SIZE,
    "augmentation":  "none",
    "accuracy":      overall_acc,
    "acc_ci_low":    acc_ci[0],
    "acc_ci_high":   acc_ci[1],
    "auc":           overall_auc,
    "auc_ci_low":    auc_ci[0],
    "auc_ci_high":   auc_ci[1],
    "precision":     prec,
    "recall":        rec,
    "f1":            f1,
    "mean_fold_acc": np.mean(fold_accs),
    "std_fold_acc":  np.std(fold_accs),
    "mean_fold_auc": np.mean(fold_aucs) if fold_aucs else float("nan"),
    "std_fold_auc":  np.std(fold_aucs) if fold_aucs else float("nan"),
    "n_folds":       n_folds,
    "time_sec":      t_elapsed,
}

write_header = not csv_path.exists()
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerow(row)

print(f"\n  [SAVED] {csv_path}")

# ── Print comparison table ────────────────────────────────
print(f"\n{'=' * 65}")
print("  COMPARISON (128×128, no augmentation)")
print(f"{'=' * 65}")
print(f"  {'Model':<25} {'Acc [95% CI]':<28} {'AUC [95% CI]':<28} {'F1':<8}")
print(f"  {'─' * 89}")
print(f"  {'ResNet50+MeanPool':<25} "
      f"0.690 [0.625, 0.755]         "
      f"0.737 [0.654, 0.827]         "
      f"0.704")
print(f"  {'ResNet50+LSTM':<25} "
      f"{overall_acc:.3f} [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]     "
      f"{overall_auc:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]     "
      f"{f1:.3f}")
print(f"  {'ResNet50+WMV (ours)':<25} "
      f"0.696 [0.646, 0.744]         "
      f"0.732 [0.667, 0.802]         "
      f"0.696")
print(f"\n  Done!")
