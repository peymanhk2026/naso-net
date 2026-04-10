"""
Naso-Net Evaluation & Inference Script
=======================================
- Load a trained fold model and run inference on sequences
- Generate ROC curves, confusion matrices, saliency maps (IG)
- Patient-level aggregation
- Window-length ablation study

Usage:
  python naso_net_eval.py                         # full evaluation
  python naso_net_eval.py --mode predict --seq_path <path>  # single sequence
  python naso_net_eval.py --mode ablation         # window length study
  python naso_net_eval.py --mode saliency         # generate IG saliency maps
"""

import os
import sys
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, auc,
    confusion_matrix, classification_report,
    precision_recall_curve,
)

# Import from training script
sys.path.insert(0, str(Path(__file__).parent))
from naso_net_train import (
    load_dataset, load_and_preprocess_sequence,
    predict_sequence_full, build_naso_net_resnet, build_naso_net_light,
    make_generator, OneCycleLR,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not installed — plots will be skipped")


def parse_args():
    p = argparse.ArgumentParser(description="Naso-Net Evaluation")
    p.add_argument("--data_dir", type=str,
                   default=r"D:\PLOS ONE\VPI case videos\extracted_sequences")
    p.add_argument("--model_dir", type=str,
                   default=r"D:\PLOS ONE\naso_net_results")
    p.add_argument("--output_dir", type=str,
                   default=r"D:\PLOS ONE\naso_net_results\eval")
    p.add_argument("--backbone", type=str, default="resnet",
                   choices=["resnet", "light"])
    p.add_argument("--target_size", type=int, default=90)
    p.add_argument("--window", type=int, default=45)
    p.add_argument("--mode", type=str, default="evaluate",
                   choices=["evaluate", "predict", "ablation", "saliency"])
    p.add_argument("--seq_path", type=str, default=None,
                   help="Path to a sequence folder (for --mode predict)")
    p.add_argument("--fold", type=int, default=1,
                   help="Which fold model to use for single predictions")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════
# Plot: ROC Curve
# ═══════════════════════════════════════════════════════════
def plot_roc_curve(y_true, y_prob, save_path):
    if not HAS_MPL:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Naso-Net: ROC Curve (Sequence-Level)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ROC curve saved: {save_path}")


# ═══════════════════════════════════════════════════════════
# Plot: Precision-Recall Curve
# ═══════════════════════════════════════════════════════════
def plot_pr_curve(y_true, y_prob, save_path):
    if not HAS_MPL:
        return
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(rec, prec, color="green", lw=2,
            label=f"PR curve (AUC = {pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Naso-Net: Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  PR curve saved: {save_path}")


# ═══════════════════════════════════════════════════════════
# Plot: Confusion Matrix
# ═══════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, save_path, title=""):
    if not HAS_MPL:
        return
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ["Open (neg)", "Closed (pos)"]
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes,
           ylabel="True label", xlabel="Predicted label",
           title=title or "Confusion Matrix")

    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved: {save_path}")


# ═══════════════════════════════════════════════════════════
# Integrated Gradients (Saliency / Explainability)
# ═══════════════════════════════════════════════════════════
def integrated_gradients(model, input_seq, baseline=None, steps=50):
    """
    Compute Integrated Gradients for a single sequence.
    input_seq: np.array of shape (window, H, W, 3)
    Returns: attribution map of same shape
    """
    if baseline is None:
        baseline = np.zeros_like(input_seq)

    # Interpolate
    alphas = np.linspace(0.0, 1.0, steps + 1)
    interpolated = np.array([
        baseline + alpha * (input_seq - baseline) for alpha in alphas
    ])  # (steps+1, window, H, W, 3)

    # Compute gradients
    interpolated_tf = tf.cast(interpolated, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(interpolated_tf)
        preds = model(interpolated_tf)

    grads = tape.gradient(preds, interpolated_tf)  # (steps+1, window, H, W, 3)

    # Approximate integral via trapezoidal rule
    grads_np = grads.numpy()
    avg_grads = np.mean(grads_np, axis=0)  # (window, H, W, 3)

    # IG = (input - baseline) * avg_gradients
    ig = (input_seq - baseline) * avg_grads

    return ig


def generate_saliency_maps(args):
    """Generate IG saliency maps for sample sequences (paper Fig 7)."""
    output_dir = Path(args.output_dir) / "saliency"
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences, _ = load_dataset(args.data_dir)

    model_path = Path(args.model_dir) / f"naso_net_fold{args.fold}.keras"
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return

    model = tf.keras.models.load_model(str(model_path))
    use_resnet = args.backbone == "resnet"

    # Sample a few positive and negative sequences
    pos_seqs = [s for s in sequences if s["label"] == 1]
    neg_seqs = [s for s in sequences if s["label"] == 0]

    random.seed(args.seed)
    samples = random.sample(pos_seqs, min(3, len(pos_seqs))) + \
              random.sample(neg_seqs, min(2, len(neg_seqs)))

    for idx, s in enumerate(samples):
        print(f"\n  Generating saliency for: {s['path']}")
        x = load_and_preprocess_sequence(
            s["path"], args.window, args.target_size,
            augment=False, use_resnet_preprocess=use_resnet,
        )

        ig_attr = integrated_gradients(model, x)

        if not HAS_MPL:
            continue

        # Show 8 evenly-spaced frames with their saliency overlay
        n_show = min(8, args.window)
        frame_indices = np.linspace(0, args.window - 1, n_show, dtype=int)

        fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2.5, 5))

        label_text = "Closed (pos)" if s["label"] == 1 else "Open (neg)"

        for j, fi in enumerate(frame_indices):
            # Original frame
            frame_img = x[fi]
            if use_resnet:
                # Undo resnet preprocessing approximately
                display = (frame_img + 128) / 255.0
                display = np.clip(display, 0, 1)
            else:
                display = frame_img

            axes[0, j].imshow(display)
            axes[0, j].set_title(f"Frame {fi+1}", fontsize=8)
            axes[0, j].axis("off")

            # Saliency heatmap
            ig_frame = ig_attr[fi]
            heatmap = np.mean(np.abs(ig_frame), axis=-1)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            axes[1, j].imshow(display)
            axes[1, j].imshow(heatmap, cmap="jet", alpha=0.5)
            axes[1, j].axis("off")

        fig.suptitle(f"IG Saliency: {Path(s['path']).parent.name}/{Path(s['path']).name} "
                     f"[{label_text}]", fontsize=10)
        fig.tight_layout()
        save_path = output_dir / f"saliency_{idx}_{Path(s['path']).name}.png"
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
        print(f"  Saved: {save_path}")

    del model
    tf.keras.backend.clear_session()


# ═══════════════════════════════════════════════════════════
# Window Length Ablation (paper Fig 6)
# ═══════════════════════════════════════════════════════════
def run_ablation(args):
    """Test different window lengths and report accuracy/AUC for each."""
    from sklearn.model_selection import GroupKFold

    sequences, patients = load_dataset(args.data_dir)
    labels = np.array([s["label"] for s in sequences])
    groups = np.array([s["patient"] for s in sequences])
    use_resnet = args.backbone == "resnet"

    windows = [15, 20, 30, 45, 60, 75, 90]
    results = {}

    for wlen in windows:
        print(f"\n{'='*50}")
        print(f"  Window Length: {wlen} frames (~{wlen/30:.1f}s at 30fps)")
        print(f"{'='*50}")

        gkf = GroupKFold(n_splits=min(5, len(patients)))
        fold_accs, fold_aucs = [], []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(labels, labels, groups)):
            train_seqs = [sequences[i] for i in train_idx]
            test_seqs = [sequences[i] for i in test_idx]

            if use_resnet:
                model = build_naso_net_resnet(wlen, args.target_size, "gelu", 1e-3)
            else:
                model = build_naso_net_light(wlen, args.target_size, "gelu", 1e-3)

            train_gen = make_generator(
                train_seqs, wlen, args.target_size, args.batch_size,
                augment=True, use_resnet_preprocess=use_resnet,
            )

            steps = max(10, len(train_seqs) // (args.batch_size * 2))
            model.fit(train_gen, epochs=30, steps_per_epoch=steps, verbose=0)

            y_true, y_prob = [], []
            for s in test_seqs:
                prob = predict_sequence_full(
                    s, model, wlen, args.target_size,
                    use_resnet_preprocess=use_resnet,
                )
                y_true.append(s["label"])
                y_prob.append(prob)

            y_true = np.array(y_true)
            y_prob = np.array(y_prob)
            y_pred = (y_prob >= 0.5).astype(int)

            acc = accuracy_score(y_true, y_pred)
            try:
                auc_val = roc_auc_score(y_true, y_prob)
            except ValueError:
                auc_val = float("nan")

            fold_accs.append(acc)
            if not np.isnan(auc_val):
                fold_aucs.append(auc_val)

            del model
            tf.keras.backend.clear_session()

        mean_acc = np.mean(fold_accs)
        mean_auc = np.mean(fold_aucs) if fold_aucs else float("nan")
        results[wlen] = {"accuracy": mean_acc, "auc": mean_auc}
        print(f"  Window {wlen}: Accuracy={mean_acc:.4f}, AUC={mean_auc:.4f}")

    # Plot ablation results (like paper Fig 6)
    if HAS_MPL:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ws = sorted(results.keys())
        accs = [results[w]["accuracy"] for w in ws]
        aucs = [results[w]["auc"] for w in ws]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(ws, accs, "b-o", label="Accuracy")
        ax1.set_xlabel("Window Length (frames)")
        ax1.set_ylabel("Accuracy", color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        ax2 = ax1.twinx()
        ax2.plot(ws, aucs, "r-s", label="AUC")
        ax2.set_ylabel("AUC", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        # Add secondary x-axis for seconds
        ax3 = ax1.twiny()
        ax3.set_xlim(ax1.get_xlim())
        sec_ticks = [w / 30 for w in ws]
        ax3.set_xticks(ws)
        ax3.set_xticklabels([f"{s:.1f}s" for s in sec_ticks])
        ax3.set_xlabel("Window Length (seconds at 30fps)")

        fig.suptitle("Impact of Temporal Sliding Window on Classification", y=1.02)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

        fig.tight_layout()
        save_path = output_dir / "ablation_window_length.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Ablation plot saved: {save_path}")


# ═══════════════════════════════════════════════════════════
# Full Evaluation (load saved predictions or re-run)
# ═══════════════════════════════════════════════════════════
def run_evaluation(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preds_file = Path(args.model_dir) / "predictions.npz"

    if preds_file.exists():
        print(f"  Loading saved predictions from: {preds_file}")
        data = np.load(str(preds_file))
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        y_pred = data["y_pred"]
    else:
        print("[ERROR] No predictions found. Run naso_net_train.py first.")
        return

    print(f"\n  Total predictions: {len(y_true)}")
    print(f"  Positives: {sum(y_true == 1)}, Negatives: {sum(y_true == 0)}")

    acc = accuracy_score(y_true, y_pred)
    try:
        auc_val = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_val = float("nan")

    print(f"\n  Accuracy: {acc:.4f}")
    print(f"  AUC     : {auc_val:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Open (neg)', 'Closed (pos)'])}")

    # Generate plots
    plot_roc_curve(y_true, y_prob, str(output_dir / "roc_curve.png"))
    plot_pr_curve(y_true, y_prob, str(output_dir / "pr_curve.png"))
    plot_confusion_matrix(y_true, y_pred, str(output_dir / "confusion_matrix.png"),
                          title="Naso-Net: Sequence-Level Confusion Matrix")


# ═══════════════════════════════════════════════════════════
# Single Sequence Prediction
# ═══════════════════════════════════════════════════════════
def run_predict(args):
    if not args.seq_path:
        print("[ERROR] --seq_path is required for predict mode")
        return

    model_path = Path(args.model_dir) / f"naso_net_fold{args.fold}.keras"
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return

    model = tf.keras.models.load_model(str(model_path))
    use_resnet = args.backbone == "resnet"

    seq_info = {"path": args.seq_path, "label": -1}
    prob = predict_sequence_full(
        seq_info, model, args.window, args.target_size,
        use_resnet_preprocess=use_resnet,
    )

    pred_label = "CLOSED (positive)" if prob >= 0.5 else "OPEN (negative)"
    print(f"\n  Sequence: {args.seq_path}")
    print(f"  Probability (closed): {prob:.4f}")
    print(f"  Prediction: {pred_label}")
    print(f"  Confidence: {max(prob, 1-prob)*100:.1f}%")

    del model
    tf.keras.backend.clear_session()


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()

    if args.mode == "evaluate":
        run_evaluation(args)
    elif args.mode == "predict":
        run_predict(args)
    elif args.mode == "ablation":
        run_ablation(args)
    elif args.mode == "saliency":
        generate_saliency_maps(args)
