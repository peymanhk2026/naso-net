"""
L251-264 — Recompute bootstrap CIs with patient-level resampling.

Addresses Reviewer Comment (L251-264):
  "Bootstrap CI description is muddled... Use ≥1,000 bootstrap iterations
   or fold-wise variance, and ensure the resampling respects patient
   clustering."

Fix: Resample *patients* (with replacement), pull all their sequences,
     then compute the metric.  1,000 iterations.  Applies to all 6
     ablation experiments.  Reads saved checkpoints (inference only).

Outputs:
  - Updated ablation_results.csv  (old one backed up as _backup.csv)
  - Console table comparing old (sequence-level) vs new (patient-level) CIs

Usage:
    & "D:\PLOS ONE\vpi_env\Scripts\python.exe" L251-264_bootstrap_patient_ci.py
"""

import os
import sys
import csv
import math
import random
import shutil
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).parent))
from ablation_resolution_augmentation import (
    load_dataset, build_naso_net_resnet, predict_sequence_full,
)

# ── Config ────────────────────────────────────────────────
DATA_DIR   = r"D:\PLOS ONE\VPI case videos\extracted_sequences"
MODEL_DIR  = Path(r"D:\PLOS ONE\naso_net_results\ablation_res_aug")
CSV_PATH   = MODEL_DIR / "ablation_results.csv"

EXPERIMENTS = [
    {"resolution": 128, "augmentation": "none"},
    {"resolution": 128, "augmentation": "conservative"},
    {"resolution": 128, "augmentation": "moderate"},
    {"resolution": 160, "augmentation": "none"},
    {"resolution": 160, "augmentation": "conservative"},
    {"resolution": 160, "augmentation": "moderate"},
]

WINDOW     = 45
LR         = 1e-3
N_FOLDS    = 10
SEED       = 42
USE_RESNET = True
N_BOOT     = 1000

CSV_FIELDNAMES = [
    "resolution", "augmentation", "aug_desc",
    "accuracy", "acc_ci_low", "acc_ci_high",
    "auc", "auc_ci_low", "auc_ci_high",
    "precision", "recall", "f1",
    "mean_fold_acc", "std_fold_acc",
    "mean_fold_auc", "std_fold_auc",
    "n_folds", "time_sec",
]


def patient_bootstrap_ci(y_true, y_prob, y_pred, patient_ids, n_boot=1000):
    """
    Bootstrap 95% CI resampling at the patient level.
    Draws patients with replacement, includes ALL their sequences,
    computes accuracy and AUC per resample.
    """
    unique_patients = np.unique(patient_ids)
    # Pre-build index: patient -> array of sequence indices
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
print("  L251-264: Recomputing CIs with patient-level bootstrap")
print(f"  {N_BOOT} iterations, resampling patients (not sequences)")
print("=" * 65)

# ── Load existing CSV for time_sec and aug_desc ───────────
old_rows = {}
if CSV_PATH.exists():
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (int(r["resolution"]), r["augmentation"])
            old_rows[key] = r

# ── Backup existing CSV ──────────────────────────────────
if CSV_PATH.exists():
    backup = MODEL_DIR / "ablation_results_backup.csv"
    shutil.copy2(CSV_PATH, backup)
    print(f"\n  Backed up existing CSV → {backup.name}")

# ── Load dataset ─────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("\nLoading dataset...")
sequences, patients = load_dataset(DATA_DIR)
labels = np.array([s["label"] for s in sequences])
groups = np.array([s["patient"] for s in sequences])

n_folds = min(N_FOLDS, len(patients))
gkf = GroupKFold(n_splits=n_folds)
splits = list(gkf.split(labels, labels, groups))

# ── Process each experiment ──────────────────────────────
new_results = []

for exp in EXPERIMENTS:
    res = exp["resolution"]
    aug = exp["augmentation"]
    key = (res, aug)
    print(f"\n{'─' * 65}")
    print(f"  Experiment: {res}×{res} + {aug}")
    print(f"{'─' * 65}")

    all_y_true = []
    all_y_prob = []
    all_y_pred = []
    all_patients = []       # parallel list: patient ID per sequence
    fold_accs, fold_aucs = [], []
    completed_folds = 0

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_num = fold_idx + 1
        ckpt = str(MODEL_DIR / f"ablation_res{res}_aug{aug}_fold{fold_num}.weights.h5")

        if not os.path.exists(ckpt):
            print(f"    Fold {fold_num}: checkpoint NOT found, skipping")
            continue

        test_seqs = [sequences[i] for i in test_idx]
        test_pats = sorted(set(s["patient"] for s in test_seqs))
        print(f"    Fold {fold_num}/{n_folds}: {len(test_seqs)} seqs "
              f"({', '.join(test_pats)})")

        model = build_naso_net_resnet(WINDOW, res, "gelu", LR)
        model.load_weights(ckpt)

        y_true, y_prob = [], []
        seq_patients = []
        for s in test_seqs:
            prob = predict_sequence_full(
                s, model, WINDOW, res, use_resnet_preprocess=USE_RESNET,
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
        completed_folds += 1

        print(f"      -> Acc={acc:.4f}, AUC={auc_val:.4f}")

        del model
        tf.keras.backend.clear_session()

    if not all_y_true:
        print("    No folds found — skipping experiment")
        continue

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

    # ── Patient-level bootstrap CIs ──────────────────────
    acc_ci, auc_ci = patient_bootstrap_ci(
        all_y_true, all_y_prob, all_y_pred, all_patients, N_BOOT,
    )

    print(f"\n    Overall:  Acc={overall_acc:.4f} [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]")
    print(f"              AUC={overall_auc:.4f} [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")
    print(f"              Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")

    # Show old vs new CIs
    if key in old_rows:
        old = old_rows[key]
        print(f"    Old CIs:  Acc [{old['acc_ci_low'][:6]}, {old['acc_ci_high'][:6]}]  "
              f"AUC [{old['auc_ci_low'][:6]}, {old['auc_ci_high'][:6]}]")
        print(f"    New CIs:  Acc [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]  "
              f"AUC [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")

    # Retrieve time_sec and aug_desc from old row if available
    time_sec = float(old_rows[key]["time_sec"]) if key in old_rows else 0
    aug_desc = old_rows[key]["aug_desc"] if key in old_rows else aug

    new_results.append({
        "resolution":    res,
        "augmentation":  aug,
        "aug_desc":      aug_desc,
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
        "n_folds":       completed_folds,
        "time_sec":      time_sec,
    })

# ── Write updated CSV ────────────────────────────────────
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    for r in new_results:
        writer.writerow({k: r[k] for k in CSV_FIELDNAMES})

print(f"\n{'=' * 65}")
print(f"  [SAVED] Updated {CSV_PATH}")
print(f"  {len(new_results)} experiments with patient-level bootstrap CIs")
print(f"  ({N_BOOT} iterations, resampling {len(patients)} patients)")
print(f"{'=' * 65}")

# ── Summary comparison table ─────────────────────────────
print(f"\n  {'Config':<28} {'Old Acc CI':<24} {'New Acc CI (patient)':<24} {'Old AUC CI':<24} {'New AUC CI (patient)':<24}")
print("  " + "─" * 124)
for r in new_results:
    key = (r["resolution"], r["augmentation"])
    cfg = f"{r['resolution']}×{r['resolution']}+{r['augmentation']}"
    new_acc = f"[{r['acc_ci_low']:.4f}, {r['acc_ci_high']:.4f}]"
    new_auc = f"[{r['auc_ci_low']:.4f}, {r['auc_ci_high']:.4f}]"
    if key in old_rows:
        old = old_rows[key]
        old_acc = f"[{float(old['acc_ci_low']):.4f}, {float(old['acc_ci_high']):.4f}]"
        old_auc = f"[{float(old['auc_ci_low']):.4f}, {float(old['auc_ci_high']):.4f}]"
    else:
        old_acc = old_auc = "N/A"
    print(f"  {cfg:<28} {old_acc:<24} {new_acc:<24} {old_auc:<24} {new_auc:<24}")

print("\n  Done!")
