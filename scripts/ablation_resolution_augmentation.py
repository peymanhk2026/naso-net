"""
Ablation Study: Resolution & Augmentation
==========================================
Addresses Reviewer 1 comment:
  "Restrict augmentations to plausible ranges (small rotations, no flips).
   Test higher resolutions (e.g., 160–224 px) and report the trade-off vs 90×90."

Experiments:
  Resolution sweep  : 90×90, 128×128, 160×160, 224×224
  Augmentation configs:
    A) none         — no augmentation (baseline)
    B) conservative — rotation ±5°, brightness/contrast ±10%
    C) moderate     — rotation ±10°, brightness/contrast ±15%  (current default)
    D) aggressive   — rotation ±20°, brightness/contrast ±25%, zoom ±10%

  No horizontal flips in any configuration (anatomical laterality preserved).

Each experiment runs the full patient-wise GroupKFold CV pipeline from
naso_net_train.py, collecting per-fold and aggregate metrics.

Results are written to a summary CSV + text report suitable for a
manuscript table or supplementary material.

Usage:
  python ablation_resolution_augmentation.py                       # full run
  python ablation_resolution_augmentation.py --quick               # 2-fold, 5 epochs (test)
  python ablation_resolution_augmentation.py --resolutions 90 160  # subset
  python ablation_resolution_augmentation.py --aug_configs none moderate
"""

import os
import sys
import math
import time
import random
import argparse
import warnings
import csv
from pathlib import Path
from collections import defaultdict
from datetime import timedelta

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D,
    TimeDistributed, Conv2D, MaxPooling2D, Layer,
)
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

from PIL import Image, ImageEnhance


# ─── Reuse core components from naso_net_train.py ───
@tf.keras.utils.register_keras_serializable(package="NasoNet")
class WeightedMeanVoting(Layer):
    def call(self, inputs):
        p_vals, w_vals = inputs
        weighted_sum = tf.reduce_sum(p_vals * w_vals, axis=1)
        weight_sum = tf.reduce_sum(w_vals, axis=1)
        return weighted_sum / (weight_sum + 1e-8)


# ═══════════════════════════════════════════════════════════
# Augmentation Configurations
# ═══════════════════════════════════════════════════════════
AUGMENTATION_CONFIGS = {
    "none": {
        "rotation": 0, "brightness": 0, "contrast": 0, "zoom": 0,
        "desc": "No augmentation (baseline)",
    },
    "conservative": {
        "rotation": 5, "brightness": 0.10, "contrast": 0.10, "zoom": 0,
        "desc": "±5° rotation, ±10% brightness/contrast",
    },
    "moderate": {
        "rotation": 10, "brightness": 0.15, "contrast": 0.15, "zoom": 0,
        "desc": "±10° rotation, ±15% brightness/contrast (paper default)",
    },
}


# ═══════════════════════════════════════════════════════════
# Dataset Loading (same as naso_net_train.py)
# ═══════════════════════════════════════════════════════════
def load_dataset(data_dir):
    sequences = []
    data_path = Path(data_dir)
    for patient_dir in sorted(data_path.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        for seq_dir in sorted(patient_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            name = seq_dir.name
            if name.startswith("pos_"):
                label = 1
            elif name.startswith("neg_"):
                label = 0
            else:
                continue
            frames = [f for f in seq_dir.iterdir()
                      if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
            if len(frames) == 0:
                continue
            sequences.append({
                "path": str(seq_dir),
                "label": label,
                "patient": patient_id,
                "n_frames": len(frames),
            })
    patients = sorted(set(s["patient"] for s in sequences))
    pos = sum(1 for s in sequences if s["label"] == 1)
    neg = sum(1 for s in sequences if s["label"] == 0)
    print(f"  Dataset: {len(sequences)} seqs ({pos} pos, {neg} neg) "
          f"from {len(patients)} patients")
    return sequences, patients


# ═══════════════════════════════════════════════════════════
# Augmented Preprocessing (parameterized)
# ═══════════════════════════════════════════════════════════
def load_and_preprocess_sequence(seq_path, window_length, target_size,
                                 aug_config=None, augment=False,
                                 use_resnet_preprocess=False):
    frame_files = sorted(
        [str(f) for f in Path(seq_path).iterdir()
         if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda x: int(Path(x).stem)
    )
    n = len(frame_files)

    if n >= window_length:
        start = random.randint(0, n - window_length)
        selected = frame_files[start:start + window_length]
    else:
        selected = frame_files + [frame_files[-1]] * (window_length - n)

    # Determine augmentation params from config
    if augment and aug_config and aug_config["rotation"] > 0:
        rotation = random.uniform(-aug_config["rotation"], aug_config["rotation"])
    else:
        rotation = 0.0

    if augment and aug_config and aug_config["brightness"] > 0:
        brightness = random.uniform(1 - aug_config["brightness"],
                                    1 + aug_config["brightness"])
    else:
        brightness = 1.0

    if augment and aug_config and aug_config["contrast"] > 0:
        contrast = random.uniform(1 - aug_config["contrast"],
                                  1 + aug_config["contrast"])
    else:
        contrast = 1.0

    if augment and aug_config and aug_config.get("zoom", 0) > 0:
        zoom = random.uniform(-aug_config["zoom"], aug_config["zoom"])
    else:
        zoom = 0.0

    images = []
    for fp in selected:
        img = Image.open(fp).convert("RGB")
        w, h = img.size
        crop_pct = 0.15
        left = int(crop_pct * w)
        right = int((1 - crop_pct) * w)
        img = img.crop((left, 0, right, h))

        # Apply zoom (crop center then resize back)
        if zoom != 0.0:
            cw, ch = img.size
            zw = int(cw * abs(zoom) / 2)
            zh = int(ch * abs(zoom) / 2)
            if zoom > 0:  # zoom in
                img = img.crop((zw, zh, cw - zw, ch - zh))
            else:  # zoom out — pad with black
                from PIL import ImageOps
                img = ImageOps.expand(img, border=(zw, zh, zw, zh), fill=0)

        img = img.resize((target_size, target_size), Image.BILINEAR)

        if rotation != 0:
            img = img.rotate(rotation, fillcolor=(0, 0, 0))
        if brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(brightness)
        if contrast != 1.0:
            img = ImageEnhance.Contrast(img).enhance(contrast)

        arr = np.array(img, dtype=np.float32)
        if use_resnet_preprocess:
            arr = resnet_preprocess(arr)
        else:
            arr = arr / 255.0
        images.append(arr)

    return np.stack(images, axis=0)


def make_generator(seqs, window_length, target_size, batch_size,
                   aug_config=None, augment=False, use_resnet_preprocess=False):
    pos_seqs = [s for s in seqs if s["label"] == 1]
    neg_seqs = [s for s in seqs if s["label"] == 0]
    half = batch_size // 2

    while True:
        batch_x, batch_y = [], []
        chosen_pos = random.choices(pos_seqs, k=half)
        chosen_neg = random.choices(neg_seqs, k=batch_size - half)

        for s in chosen_pos + chosen_neg:
            x = load_and_preprocess_sequence(
                s["path"], window_length, target_size,
                aug_config=aug_config, augment=augment,
                use_resnet_preprocess=use_resnet_preprocess,
            )
            batch_x.append(x)
            batch_y.append(s["label"])

        idx = list(range(len(batch_x)))
        random.shuffle(idx)
        batch_x = np.array([batch_x[i] for i in idx])
        batch_y = np.array([batch_y[i] for i in idx], dtype=np.float32)
        yield batch_x, batch_y


# ═══════════════════════════════════════════════════════════
# Model Architecture (same as naso_net_train.py)
# ═══════════════════════════════════════════════════════════
def build_naso_net_resnet(window_length, target_size, activation="gelu", lr=1e-3):
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
    w = TimeDistributed(Dense(1, activation="softplus"), name="frame_weight")(x)
    out = WeightedMeanVoting(name="wmv")([p, w])

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_naso_net_light(window_length, target_size, activation="gelu", lr=1e-3):
    inp = Input(shape=(window_length, target_size, target_size, 3))
    x = TimeDistributed(Conv2D(32, 3, activation=activation, padding="same"))(inp)
    x = TimeDistributed(MaxPooling2D(2))(x)
    x = TimeDistributed(Conv2D(64, 3, activation=activation, padding="same"))(x)
    x = TimeDistributed(MaxPooling2D(2))(x)
    x = TimeDistributed(Conv2D(128, 3, activation=activation, padding="same"))(x)
    x = TimeDistributed(MaxPooling2D(2))(x)
    x = TimeDistributed(Conv2D(256, 3, activation=activation, padding="same"))(x)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = TimeDistributed(Dense(128, activation=activation))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.4))(x)

    p = TimeDistributed(Dense(1, activation="sigmoid"), name="frame_prob")(x)
    w = TimeDistributed(Dense(1, activation="softplus"), name="frame_weight")(x)
    out = WeightedMeanVoting(name="wmv")([p, w])

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


# ═══════════════════════════════════════════════════════════
# OneCycleLR (same as naso_net_train.py)
# ═══════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════
# Sliding-window inference (same as naso_net_train.py)
# ═══════════════════════════════════════════════════════════
def predict_sequence_full(seq_info, model, window_length, target_size,
                          stride=None, use_resnet_preprocess=False):
    if stride is None:
        stride = max(1, window_length // 2)

    frame_files = sorted(
        [str(f) for f in Path(seq_info["path"]).iterdir()
         if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda x: int(Path(x).stem)
    )
    n = len(frame_files)

    if n < window_length:
        x = load_and_preprocess_sequence(
            seq_info["path"], window_length, target_size,
            augment=False, use_resnet_preprocess=use_resnet_preprocess,
        )
        pred = model.predict(np.expand_dims(x, 0), verbose=0)
        return float(np.squeeze(pred))

    all_frames = []
    for fp in frame_files:
        img = Image.open(fp).convert("RGB")
        w, h = img.size
        crop_pct = 0.15
        img = img.crop((int(crop_pct * w), 0, int((1 - crop_pct) * w), h))
        img = img.resize((target_size, target_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        if use_resnet_preprocess:
            arr = resnet_preprocess(arr)
        else:
            arr = arr / 255.0
        all_frames.append(arr)
    all_frames = np.stack(all_frames, axis=0)

    all_windows = []
    for start in range(0, n - window_length + 1, stride):
        all_windows.append(all_frames[start:start + window_length])

    preds = []
    batch_sz = 16
    for i in range(0, len(all_windows), batch_sz):
        batch = np.array(all_windows[i:i + batch_sz])
        batch_preds = model.predict(batch, verbose=0)
        squeezed = np.squeeze(batch_preds)
        if squeezed.ndim == 0:
            preds.append(float(squeezed))
        else:
            preds.extend(squeezed.tolist())

    return float(np.mean(preds))


# ═══════════════════════════════════════════════════════════
# Single Experiment Runner
# ═══════════════════════════════════════════════════════════
def run_single_experiment(sequences, patients, target_size, aug_name,
                          aug_config, backbone, window, epochs, batch_size,
                          lr, n_folds, seed, output_dir):
    """
    Run one full patient-wise CV experiment and return results dict.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    labels = np.array([s["label"] for s in sequences])
    groups = np.array([s["patient"] for s in sequences])
    use_resnet = backbone == "resnet"
    n_folds = min(n_folds, len(patients))

    gkf = GroupKFold(n_splits=n_folds)
    steps_per_epoch = max(10, len(sequences) // (batch_size * 2))

    all_y_true, all_y_prob, all_y_pred = [], [], []
    fold_accs, fold_aucs = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(labels, labels, groups)):
        train_seqs = [sequences[i] for i in train_idx]
        test_seqs = [sequences[i] for i in test_idx]

        train_patients = sorted(set(s["patient"] for s in train_seqs))
        test_patients = sorted(set(s["patient"] for s in test_seqs))

        print(f"    Fold {fold_idx + 1}/{n_folds}: "
              f"train={len(train_seqs)} seqs ({len(train_patients)} pts), "
              f"test={len(test_seqs)} seqs ({len(test_patients)} pts)")

        # Build model
        if use_resnet:
            model = build_naso_net_resnet(window, target_size, "gelu", lr)
        else:
            model = build_naso_net_light(window, target_size, "gelu", lr)

        # Generators
        train_gen = make_generator(
            train_seqs, window, target_size, batch_size,
            aug_config=aug_config, augment=(aug_name != "none"),
            use_resnet_preprocess=use_resnet,
        )
        val_gen = make_generator(
            test_seqs, window, target_size, batch_size,
            aug_config=None, augment=False,
            use_resnet_preprocess=use_resnet,
        )

        val_steps = max(5, len(test_seqs) // batch_size)
        checkpoint_path = str(
            output_dir / f"ablation_res{target_size}_aug{aug_name}_fold{fold_idx+1}.weights.h5"
        )
        periodic_path = str(
            output_dir / f"ablation_res{target_size}_aug{aug_name}_fold{fold_idx+1}_epoch{{epoch:02d}}.weights.h5"
        )

        callbacks = [
            OneCycleLR(max_lr=lr, base_lr=lr / 10,
                       epochs=epochs, steps_per_epoch=steps_per_epoch),
            EarlyStopping(monitor="val_loss", patience=15,
                          restore_best_weights=True, verbose=0),
            ModelCheckpoint(checkpoint_path, monitor="val_auc", mode="max",
                            save_best_only=True, save_weights_only=True, verbose=0),
            ModelCheckpoint(periodic_path, save_weights_only=True,
                            save_freq=5 * steps_per_epoch, verbose=0),
        ]

        model.fit(
            train_gen, epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen, validation_steps=val_steps,
            callbacks=callbacks, verbose=1,
        )

        if os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path)

        # Evaluate
        y_true, y_prob = [], []
        for s in test_seqs:
            prob = predict_sequence_full(
                s, model, window, target_size,
                use_resnet_preprocess=use_resnet,
            )
            y_true.append(s["label"])
            y_prob.append(prob)

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")

        fold_accs.append(acc)
        if not math.isnan(auc):
            fold_aucs.append(auc)
        all_y_true.extend(y_true.tolist())
        all_y_prob.extend(y_prob.tolist())
        all_y_pred.extend(y_pred.tolist())

        print(f"      -> Acc={acc:.4f}, AUC={auc:.4f}")

        del model
        tf.keras.backend.clear_session()

    # Aggregate
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = np.array(all_y_pred)

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except ValueError:
        overall_auc = float("nan")
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, average="binary", zero_division=0
    )

    # Bootstrap 95% CI (1000 iterations)
    boot_accs, boot_aucs = [], []
    for _ in range(1000):
        idx = np.random.choice(len(all_y_true), size=len(all_y_true), replace=True)
        boot_accs.append(accuracy_score(all_y_true[idx], all_y_pred[idx]))
        if len(np.unique(all_y_true[idx])) > 1:
            boot_aucs.append(roc_auc_score(all_y_true[idx], all_y_prob[idx]))

    acc_ci = (np.percentile(boot_accs, 2.5), np.percentile(boot_accs, 97.5))
    auc_ci = (np.percentile(boot_aucs, 2.5), np.percentile(boot_aucs, 97.5)) if boot_aucs else (0, 0)

    return {
        "resolution": target_size,
        "augmentation": aug_name,
        "aug_desc": aug_config["desc"] if aug_config else "No augmentation",
        "accuracy": overall_acc,
        "auc": overall_auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "acc_ci_low": acc_ci[0],
        "acc_ci_high": acc_ci[1],
        "auc_ci_low": auc_ci[0],
        "auc_ci_high": auc_ci[1],
        "mean_fold_acc": np.mean(fold_accs),
        "std_fold_acc": np.std(fold_accs),
        "mean_fold_auc": np.mean(fold_aucs) if fold_aucs else float("nan"),
        "std_fold_auc": np.std(fold_aucs) if fold_aucs else float("nan"),
        "n_folds": n_folds,
    }


# ═══════════════════════════════════════════════════════════
# CLI & Main
# ═══════════════════════════════════════════════════════════
CSV_FIELDNAMES = [
    "resolution", "augmentation", "aug_desc",
    "accuracy", "acc_ci_low", "acc_ci_high",
    "auc", "auc_ci_low", "auc_ci_high",
    "precision", "recall", "f1",
    "mean_fold_acc", "std_fold_acc",
    "mean_fold_auc", "std_fold_auc",
    "n_folds", "time_sec",
]


def _save_csv(results, csv_path):
    """Write all results collected so far to CSV (overwrites each time)."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in CSV_FIELDNAMES})


def parse_args():
    p = argparse.ArgumentParser(description="Ablation: Resolution & Augmentation")
    p.add_argument("--data_dir", type=str,
                   default=r"D:\PLOS ONE\VPI case videos\extracted_sequences")
    p.add_argument("--output_dir", type=str,
                   default=r"D:\PLOS ONE\naso_net_results\ablation_res_aug")
    p.add_argument("--backbone", type=str, default="resnet",
                   choices=["resnet", "light"])
    p.add_argument("--window", type=int, default=45)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resolutions", type=int, nargs="+",
                   default=[128, 160],
                   help="Resolutions to test (default: 128 160)")
    p.add_argument("--aug_configs", type=str, nargs="+",
                   default=["none", "conservative", "moderate"],
                   choices=["none", "conservative", "moderate"],
                   help="Augmentation configs to test")
    p.add_argument("--quick", action="store_true",
                   help="Quick test mode: 2 folds, 5 epochs, light backbone")
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.folds = 2
        args.epochs = 5
        args.backbone = "light"
        print("[QUICK MODE] 2 folds, 5 epochs, light backbone")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset once
    print("Loading dataset...")
    sequences, patients = load_dataset(args.data_dir)

    resolutions = args.resolutions
    aug_configs = args.aug_configs
    total_experiments = len(resolutions) * len(aug_configs)

    print(f"\n{'='*70}")
    print(f"  ABLATION STUDY: Resolution x Augmentation")
    print(f"  Resolutions    : {resolutions}")
    print(f"  Augmentations  : {aug_configs}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Backbone: {args.backbone} | Window: {args.window} | "
          f"Folds: {args.folds} | Epochs: {args.epochs}")
    print(f"{'='*70}\n")

    all_results = []
    exp_num = 0

    for res in resolutions:
        for aug_name in aug_configs:
            exp_num += 1
            aug_cfg = AUGMENTATION_CONFIGS[aug_name]

            print(f"\n{'─'*60}")
            print(f"  Experiment {exp_num}/{total_experiments}: "
                  f"Resolution={res}x{res}, Augmentation={aug_name}")
            print(f"  {aug_cfg['desc']}")
            print(f"{'─'*60}")

            t0 = time.time()

            result = run_single_experiment(
                sequences=sequences,
                patients=patients,
                target_size=res,
                aug_name=aug_name,
                aug_config=aug_cfg,
                backbone=args.backbone,
                window=args.window,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                n_folds=args.folds,
                seed=args.seed,
                output_dir=output_dir,
            )

            elapsed = time.time() - t0
            result["time_sec"] = elapsed

            all_results.append(result)

            print(f"\n  >>> Result: Acc={result['accuracy']:.4f} "
                  f"[{result['acc_ci_low']:.4f}, {result['acc_ci_high']:.4f}] | "
                  f"AUC={result['auc']:.4f} "
                  f"[{result['auc_ci_low']:.4f}, {result['auc_ci_high']:.4f}] | "
                  f"F1={result['f1']:.4f} | "
                  f"Time={timedelta(seconds=int(elapsed))}")

            # ─── Save incrementally after each experiment ───
            _save_csv(all_results, output_dir / "ablation_results.csv")
            print(f"  [SAVED] Results so far written to ablation_results.csv "
                  f"({len(all_results)}/{total_experiments} experiments)")

    # ─── Final CSV (same file, complete) ───
    _save_csv(all_results, output_dir / "ablation_results.csv")

    # ─── Write text report ───
    report_path = output_dir / "ablation_report.txt"
    with open(report_path, "w") as f:
        f.write("Ablation Study: Resolution x Augmentation\n")
        f.write(f"{'='*70}\n")
        f.write(f"Backbone: {args.backbone}\n")
        f.write(f"Window: {args.window} frames (~{args.window/30:.1f}s at 30fps)\n")
        f.write(f"Folds: {args.folds} | Epochs: {args.epochs} | "
                f"LR: {args.lr} | Seed: {args.seed}\n")
        f.write(f"No horizontal flips in any configuration.\n\n")

        # Table header
        f.write(f"{'Res':>5} | {'Augmentation':>14} | {'Accuracy':>8} | "
                f"{'95% CI':>17} | {'AUC':>6} | {'95% CI':>17} | "
                f"{'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'Time':>8}\n")
        f.write(f"{'-'*5}-+-{'-'*14}-+-{'-'*8}-+-{'-'*17}-+-"
                f"{'-'*6}-+-{'-'*17}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}\n")

        for r in all_results:
            f.write(
                f"{r['resolution']:>5} | {r['augmentation']:>14} | "
                f"{r['accuracy']:>8.4f} | "
                f"[{r['acc_ci_low']:.4f}, {r['acc_ci_high']:.4f}] | "
                f"{r['auc']:>6.4f} | "
                f"[{r['auc_ci_low']:.4f}, {r['auc_ci_high']:.4f}] | "
                f"{r['precision']:>6.4f} | {r['recall']:>6.4f} | "
                f"{r['f1']:>6.4f} | "
                f"{timedelta(seconds=int(r['time_sec'])):>8}\n"
            )

        f.write(f"\n{'='*70}\n")
        f.write("Augmentation configurations:\n")
        for name, cfg in AUGMENTATION_CONFIGS.items():
            if name in aug_configs:
                f.write(f"  {name:>14}: {cfg['desc']}\n")

        # Best configuration per metric
        f.write(f"\n{'='*70}\n")
        f.write("Best configurations:\n")
        best_acc = max(all_results, key=lambda r: r["accuracy"])
        best_auc = max(all_results, key=lambda r: r["auc"] if not math.isnan(r["auc"]) else -1)
        best_f1 = max(all_results, key=lambda r: r["f1"])
        f.write(f"  Best Accuracy: {best_acc['resolution']}px + {best_acc['augmentation']} "
                f"= {best_acc['accuracy']:.4f}\n")
        f.write(f"  Best AUC     : {best_auc['resolution']}px + {best_auc['augmentation']} "
                f"= {best_auc['auc']:.4f}\n")
        f.write(f"  Best F1      : {best_f1['resolution']}px + {best_f1['augmentation']} "
                f"= {best_f1['f1']:.4f}\n")

    # ─── Console summary ───
    print(f"\n\n{'='*70}")
    print(f"  ABLATION STUDY COMPLETE")
    print(f"{'='*70}")
    print(f"\n{'Res':>5} | {'Aug':>14} | {'Acc':>6} | {'AUC':>6} | "
          f"{'F1':>6} | {'Time':>8}")
    print(f"{'-'*5}-+-{'-'*14}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")
    for r in all_results:
        print(f"{r['resolution']:>5} | {r['augmentation']:>14} | "
              f"{r['accuracy']:>6.4f} | {r['auc']:>6.4f} | "
              f"{r['f1']:>6.4f} | {timedelta(seconds=int(r['time_sec'])):>8}")

    print(f"\nResults saved to:")
    print(f"  {csv_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
