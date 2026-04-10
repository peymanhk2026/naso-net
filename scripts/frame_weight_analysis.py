"""
Frame Weight Analysis — Quantifying alignment of learned frame-level
importance weights with velopharyngeal closure transition boundaries.

Addresses Reviewer Comment (L218):
  "Please show a quantitative contribution (e.g. top-k attention frames
   aligning with peak velar movement)"

Uses trained Naso-Net checkpoints (inference only, no training) to extract
per-frame w_i (importance) and p_i (closure probability), cross-references
with annotation-derived transition boundaries, and produces:
  1. Spearman correlation between w_i and distance-to-nearest-transition
  2. Top-k enrichment analysis
  3. Mann-Whitney U test (near vs. far from transitions)
  4. Figure saved as frame_weight_analysis.png
  5. Raw per-frame data saved as frame_weight_data.csv

Usage:
    & "D:\PLOS ONE\vpi_env\Scripts\python.exe" frame_weight_analysis.py
"""

import os
import sys
import math
import random
import warnings
from pathlib import Path
from collections import defaultdict
from itertools import groupby

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
from sklearn.model_selection import GroupKFold
from scipy import stats
from scipy.stats import fisher_exact
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from ablation_resolution_augmentation import (
    load_dataset,
    build_naso_net_resnet,
    WeightedMeanVoting,
)
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# ── Config (must match ablation run) ──────────────────────────
DATA_DIR   = r"D:\PLOS ONE\VPI case videos\extracted_sequences"
MODEL_DIR  = Path(r"D:\PLOS ONE\naso_net_results\ablation_res_aug")
OUTPUT_DIR = Path(r"D:\PLOS ONE\naso_net_results")

TARGET_SIZE = 160          # Best-performing resolution
AUG_NAME   = "none"        # Best-performing augmentation
WINDOW     = 45
LR         = 1e-3
N_FOLDS    = 10
SEED       = 42
USE_RESNET = True
STRIDE     = 22            # window_length // 2  (50 % overlap)

NEAR_THRESHOLD = 3         # Frames within ±3 of transition = "near" (~100 ms)
TOP_K          = 5         # Top-k frames by importance weight per window

# ── Seed ──────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── Helpers ───────────────────────────────────────────────────

def get_transitions_for_patient(patient_dir):
    """
    Derive transition frame numbers from the sequence-folder naming
    convention.  Folders like  pos_220-272 / neg_273-332  imply a state
    change between frames 272 and 273 (midpoint = 272.5).
    Returns a sorted list of transition midpoints.
    """
    entries = []
    for d in Path(patient_dir).iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if not (name.startswith("pos_") or name.startswith("neg_")):
            continue
        rng = name.split("_", 1)[1].split("-")
        if len(rng) == 2:
            try:
                entries.append((int(rng[0]), int(rng[1])))
            except ValueError:
                continue
    entries.sort()

    transitions = []
    for i in range(len(entries) - 1):
        end_curr   = entries[i][1]
        start_next = entries[i + 1][0]
        transitions.append((end_curr + start_next) / 2.0)
    return transitions


def preprocess_single_frame(filepath, target_size, use_resnet):
    """Replicate the exact preprocessing from predict_sequence_full."""
    img = Image.open(filepath).convert("RGB")
    w, h = img.size
    crop_pct = 0.15
    img = img.crop((int(crop_pct * w), 0, int((1 - crop_pct) * w), h))
    img = img.resize((target_size, target_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    if use_resnet:
        arr = resnet_preprocess(arr)
    else:
        arr /= 255.0
    return arr


def extract_frame_weights(sub_model, seq_info, window_length, target_size,
                          stride, use_resnet):
    """
    Sliding-window inference with the sub-model that outputs per-frame
    (p_i, w_i).  For frames appearing in multiple overlapping windows,
    the outputs are averaged.

    Returns  { video_frame_number: {"p": mean_p_i, "w": mean_w_i} }
    or empty dict if sequence is shorter than window_length.
    """
    frame_files = sorted(
        [str(f) for f in Path(seq_info["path"]).iterdir()
         if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda x: int(Path(x).stem),
    )
    n = len(frame_files)
    if n < window_length:
        return {}

    frame_nums = [int(Path(f).stem) for f in frame_files]

    # Pre-load all frames once
    all_frames = np.stack(
        [preprocess_single_frame(fp, target_size, use_resnet) for fp in frame_files],
        axis=0,
    )

    # Build sliding windows
    windows = []
    win_frame_idx = []          # parallel list: frame-number indices per window
    for start in range(0, n - window_length + 1, stride):
        windows.append(all_frames[start:start + window_length])
        win_frame_idx.append(list(range(start, start + window_length)))

    if not windows:
        return {}

    # Accumulate per-frame outputs across overlapping windows
    accum = defaultdict(lambda: {"p_sum": 0.0, "w_sum": 0.0, "count": 0})

    batch_sz = 16
    for i in range(0, len(windows), batch_sz):
        batch = np.array(windows[i:i + batch_sz])
        p_out, w_out = sub_model.predict(batch, verbose=0)
        # shapes: (batch, window_length, 1)

        for j in range(p_out.shape[0]):
            for k in range(window_length):
                fidx = win_frame_idx[i + j][k]
                vf   = frame_nums[fidx]
                accum[vf]["p_sum"] += float(p_out[j, k, 0])
                accum[vf]["w_sum"] += float(w_out[j, k, 0])
                accum[vf]["count"] += 1

    return {
        vf: {"p": d["p_sum"] / d["count"], "w": d["w_sum"] / d["count"]}
        for vf, d in accum.items()
    }


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("  Frame Weight Analysis")
print(f"  Config: {TARGET_SIZE}×{TARGET_SIZE} + {AUG_NAME}")
print("=" * 60)

# ── 1. Load dataset & reconstruct splits ──────────────────────
print("\nLoading dataset...")
sequences, patients = load_dataset(DATA_DIR)
labels = np.array([s["label"] for s in sequences])
groups = np.array([s["patient"] for s in sequences])
n_folds = min(N_FOLDS, len(patients))
gkf = GroupKFold(n_splits=n_folds)

# ── 2. Per-fold inference ─────────────────────────────────────
all_records = []           # one dict per frame-measurement
example_sequences = []     # a few sequences for Panel C figure
completed_folds = 0

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(labels, labels, groups)):
    fold_num = fold_idx + 1
    ckpt = str(MODEL_DIR / f"ablation_res{TARGET_SIZE}_aug{AUG_NAME}_fold{fold_num}.weights.h5")

    if not os.path.exists(ckpt):
        print(f"  Fold {fold_num}: checkpoint NOT found, skipping")
        continue

    test_seqs = [sequences[i] for i in test_idx]
    test_patients = sorted(set(s["patient"] for s in test_seqs))
    print(f"\n  Fold {fold_num}/{n_folds}: {len(test_seqs)} test seqs "
          f"({', '.join(test_patients)})")

    # Build model → load weights → create sub-model
    model = build_naso_net_resnet(WINDOW, TARGET_SIZE, "gelu", LR)
    model.load_weights(ckpt)

    sub_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer("frame_prob").output,
                 model.get_layer("frame_weight").output],
    )

    # Pre-compute transitions for each test patient
    patient_transitions = {}
    for pid in test_patients:
        pdir = Path(DATA_DIR) / pid
        patient_transitions[pid] = (
            get_transitions_for_patient(pdir) if pdir.exists() else []
        )

    # Process test sequences
    for s_idx, s in enumerate(test_seqs):
        fw = extract_frame_weights(
            sub_model, s, WINDOW, TARGET_SIZE, STRIDE, USE_RESNET,
        )
        if not fw:
            continue

        transitions = patient_transitions.get(s["patient"], [])

        for vf, vals in fw.items():
            dist = min(abs(vf - t) for t in transitions) if transitions else float("inf")
            all_records.append({
                "fold":                fold_num,
                "patient":             s["patient"],
                "seq_path":            s["path"],
                "label":               s["label"],
                "video_frame":         vf,
                "p_i":                 vals["p"],
                "w_i":                 vals["w"],
                "dist_to_transition":  dist,
            })

        # Stash a few long sequences for example plots
        if len(example_sequences) < 8 and len(fw) >= 30:
            sorted_frames = sorted(fw.keys())
            example_sequences.append({
                "patient":     s["patient"],
                "label":       s["label"],
                "frames":      sorted_frames,
                "w_values":    [fw[f]["w"] for f in sorted_frames],
                "p_values":    [fw[f]["p"] for f in sorted_frames],
                "transitions": transitions,
                "seq_name":    Path(s["path"]).name,
            })

        if (s_idx + 1) % 10 == 0:
            print(f"    {s_idx + 1}/{len(test_seqs)} sequences...")

    completed_folds += 1
    del model, sub_model
    tf.keras.backend.clear_session()

# ═══════════════════════════════════════════════════════════════
#  STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print(f"  Analysis: {len(all_records)} frame-level measurements, "
      f"{completed_folds} folds")
print(f"{'=' * 60}")

if not all_records:
    print("  No data collected — exiting.")
    sys.exit(1)

w_all    = np.array([r["w_i"]                for r in all_records])
dist_all = np.array([r["dist_to_transition"] for r in all_records])

valid = np.isfinite(dist_all)
w_v    = w_all[valid]
dist_v = dist_all[valid]
print(f"\n  Frames with finite transition distance: {valid.sum()}/{len(all_records)}")

# ── 1. Spearman correlation ──────────────────────────────────
rho, p_spear = stats.spearmanr(w_v, dist_v)
print(f"\n  1. Spearman (w_i vs distance-to-transition):")
print(f"     rho = {rho:.4f},  p = {p_spear:.2e}")

# ── 2. Near vs Far comparison ────────────────────────────────
near = dist_v <= NEAR_THRESHOLD
far  = ~near
w_near = w_v[near]
w_far  = w_v[far]

print(f"\n  2. Near-transition (≤{NEAR_THRESHOLD} frames) vs Far:")
print(f"     Near: n={len(w_near):,}, mean w_i = {np.mean(w_near):.4f} ± {np.std(w_near):.4f}")
print(f"     Far:  n={len(w_far):,}, mean w_i = {np.mean(w_far):.4f} ± {np.std(w_far):.4f}")

u_stat, p_mann = stats.mannwhitneyu(w_near, w_far, alternative="two-sided")
r_rb = 1 - (2 * u_stat) / (len(w_near) * len(w_far))   # rank-biserial r
print(f"     Mann-Whitney U = {u_stat:,.0f},  p = {p_mann:.2e}")
print(f"     Rank-biserial r = {r_rb:.4f}")

# ── 3. Top-k enrichment ─────────────────────────────────────
print(f"\n  3. Top-{TOP_K} enrichment analysis:")

seq_groups = defaultdict(list)
for r in all_records:
    if np.isfinite(r["dist_to_transition"]):
        seq_groups[(r["fold"], r["seq_path"])].append(r)

n_topk_near = n_topk_total = 0
n_all_near  = n_all_total  = 0

for key, recs in seq_groups.items():
    if len(recs) < TOP_K:
        continue
    top_k = sorted(recs, key=lambda r: r["w_i"], reverse=True)[:TOP_K]
    for r in top_k:
        n_topk_total += 1
        if r["dist_to_transition"] <= NEAR_THRESHOLD:
            n_topk_near += 1
    for r in recs:
        n_all_total += 1
        if r["dist_to_transition"] <= NEAR_THRESHOLD:
            n_all_near += 1

topk_rate   = n_topk_near / n_topk_total if n_topk_total else 0
random_rate = n_all_near  / n_all_total  if n_all_total  else 0
enrichment  = topk_rate / random_rate if random_rate > 0 else float("inf")

print(f"     Top-{TOP_K} near rate:    {topk_rate:.4f}  ({n_topk_near}/{n_topk_total})")
print(f"     Baseline near rate: {random_rate:.4f}  ({n_all_near}/{n_all_total})")
print(f"     Enrichment:         {enrichment:.2f}×")

# Contingency table: top-k vs non-top-k × near vs far
a = n_topk_near
b = n_topk_total - n_topk_near
c = n_all_near - n_topk_near
d = (n_all_total - n_topk_total) - c
table = [[a, b], [c, d]]
_, p_fisher = fisher_exact(table, alternative="two-sided")
print(f"     Fisher exact p = {p_fisher:.2e}")

# ── 4. Segment-relative position analysis ────────────────────
print(f"\n  4. Weight by relative position within segment "
      f"(0 = boundary, 1 = centre):")

pos_w = []     # (relative_position, w_i)
for r in all_records:
    sname = Path(r["seq_path"]).name
    rng = sname.split("_", 1)[1].split("-")
    if len(rng) != 2:
        continue
    seg_start, seg_end = int(rng[0]), int(rng[1])
    seg_len = seg_end - seg_start + 1
    if seg_len <= 1:
        continue
    frac   = (r["video_frame"] - seg_start) / (seg_len - 1)   # 0 → 1
    rel_centre = min(frac, 1 - frac) * 2                       # 0 = boundary, 1 = centre
    pos_w.append((rel_centre, r["w_i"]))

if pos_w:
    rel_arr = np.array([x[0] for x in pos_w])
    w_arr   = np.array([x[1] for x in pos_w])
    rho_p, p_p = stats.spearmanr(w_arr, rel_arr)
    print(f"     Spearman (w_i vs relative position): rho = {rho_p:.4f}, p = {p_p:.2e}")

    bins = np.linspace(0, 1, 6)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        m = (rel_arr >= lo) & (rel_arr < hi) if i < len(bins) - 2 else (rel_arr >= lo)
        if m.sum():
            print(f"       [{lo:.1f}–{hi:.1f}]: mean w_i = {np.mean(w_arr[m]):.4f}  "
                  f"(n={m.sum():,})")

# ═══════════════════════════════════════════════════════════════
#  FIGURE
# ═══════════════════════════════════════════════════════════════
print("\n  Generating figure...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"Frame Weight Analysis — {TARGET_SIZE}×{TARGET_SIZE} + {AUG_NAME}\n"
    f"({completed_folds} folds, {len(all_records):,} frame-level measurements)",
    fontsize=14, fontweight="bold",
)

# ── Panel A: mean w_i vs distance-to-transition ──────────────
ax = axes[0, 0]
max_d = min(50, int(np.percentile(dist_v, 95)))
d_bins = np.arange(0, max_d + 1)
means, sems = [], []
for d in d_bins:
    m = (dist_v >= d - 0.5) & (dist_v < d + 0.5)
    means.append(np.mean(w_v[m]) if m.sum() else np.nan)
    sems.append(stats.sem(w_v[m]) if m.sum() > 1 else np.nan)
means, sems = np.array(means), np.array(sems)
ok = ~np.isnan(means)
ax.plot(d_bins[ok], means[ok], "b-", lw=1.5)
ax.fill_between(d_bins[ok], means[ok] - sems[ok], means[ok] + sems[ok], alpha=0.2)
ax.axvline(NEAR_THRESHOLD, color="red", ls="--", alpha=0.7,
           label=f"Near threshold (±{NEAR_THRESHOLD})")
ax.set_xlabel("Distance to nearest transition (frames)")
ax.set_ylabel("Mean importance weight (w_i)")
ax.set_title("A. Weight vs Distance to Transition")
ax.legend(fontsize=9)

# ── Panel B: box-plot near vs far ────────────────────────────
ax = axes[0, 1]
bp = ax.boxplot([w_near, w_far],
                labels=[f"Near\n(≤{NEAR_THRESHOLD} frames)", f"Far\n(>{NEAR_THRESHOLD} frames)"],
                patch_artist=True, showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="black", markersize=6))
bp["boxes"][0].set_facecolor("#ff9999")
bp["boxes"][1].set_facecolor("#9999ff")
ax.set_ylabel("Importance weight (w_i)")
ax.set_title(f"B. Near vs Far from Transition\n(Mann-Whitney p = {p_mann:.2e})")

# ── Panel C: example sequence weight profiles ────────────────
ax = axes[1, 0]
if example_sequences:
    n_ex = min(4, len(example_sequences))
    colors = plt.cm.tab10(np.linspace(0, 0.4, n_ex))
    for i in range(n_ex):
        ex = example_sequences[i]
        frames = np.array(ex["frames"])
        wv = np.array(ex["w_values"])
        wn = (wv - wv.min()) / (wv.max() - wv.min() + 1e-8)
        tag = "pos" if ex["label"] == 1 else "neg"
        ax.plot(range(len(frames)), wn, color=colors[i], lw=1.2,
                label=f"{ex['patient']} {ex['seq_name'][:20]} ({tag})")
        for t in ex["transitions"]:
            idx = np.searchsorted(frames, t)
            if 0 < idx < len(frames):
                ax.axvline(idx, color=colors[i], ls=":", alpha=0.5)
    ax.set_xlabel("Frame position in sequence")
    ax.set_ylabel("Normalised w_i")
    ax.set_title("C. Example Weight Profiles (dashed = transitions)")
    ax.legend(fontsize=7, loc="upper right")

# ── Panel D: enrichment bar chart ────────────────────────────
ax = axes[1, 1]
bars = ax.bar(
    [f"Top-{TOP_K}\nframes", "All frames\n(baseline)"],
    [topk_rate * 100, random_rate * 100],
    color=["#ff6666", "#6666ff"], edgecolor="black",
)
ax.set_ylabel(f"% frames near transition (≤{NEAR_THRESHOLD} frames)")
ax.set_title(f"D. Top-{TOP_K} Enrichment "
             f"({enrichment:.1f}×, Fisher p = {p_fisher:.2e})")
ax.bar_label(bars, fmt="%.1f%%", fontsize=10)

plt.tight_layout()
fig_path = OUTPUT_DIR / "frame_weight_analysis.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"  [SAVED] {fig_path}")

# ═══════════════════════════════════════════════════════════════
#  SAVE RAW DATA
# ═══════════════════════════════════════════════════════════════
import csv

csv_path = OUTPUT_DIR / "frame_weight_data.csv"
fields = ["fold", "patient", "seq_path", "label", "video_frame",
          "p_i", "w_i", "dist_to_transition"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in all_records:
        writer.writerow(r)
print(f"  [SAVED] {csv_path}")

# ═══════════════════════════════════════════════════════════════
#  MANUSCRIPT SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  SUMMARY FOR MANUSCRIPT")
print(f"{'=' * 60}")
print(f"  Spearman rho (w_i vs dist):   {rho:.4f}  (p = {p_spear:.2e})")
print(f"  Near mean w_i:                {np.mean(w_near):.4f} ± {np.std(w_near):.4f}")
print(f"  Far mean w_i:                 {np.mean(w_far):.4f} ± {np.std(w_far):.4f}")
print(f"  Mann-Whitney U p:             {p_mann:.2e}")
print(f"  Rank-biserial effect size:    {r_rb:.4f}")
print(f"  Top-{TOP_K} enrichment:            {enrichment:.2f}× (Fisher p = {p_fisher:.2e})")
if pos_w:
    print(f"  Position Spearman rho:        {rho_p:.4f}  (p = {p_p:.2e})")
print(f"\n  Done!")
