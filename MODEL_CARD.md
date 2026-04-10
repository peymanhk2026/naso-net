# Model Card: Naso-Net

## Model Details

| Field | Value |
|---|---|
| **Model name** | Naso-Net |
| **Version** | 1.0 |
| **Type** | Temporal-aware image classification (binary) |
| **Architecture** | Time-Distributed ResNet50 + Weighted Mean Voting (WMV) |
| **Framework** | TensorFlow 2.10 / Keras |
| **Task** | Velopharyngeal port (VPP) closure prediction from nasopharyngoscopy video |
| **License** | MIT |
| **Paper** | *Automated detection of velopharyngeal port dynamics from nasopharyngoscopy videos using deep learning*, PLOS ONE (2026) |

---

## Intended Use

- **Primary use**: Research tool for automated analysis of velopharyngeal port dynamics in nasopharyngoscopy videos.
- **Intended users**: Researchers in medical image analysis, speech-language pathology, and craniofacial surgery.
- **Out-of-scope**: This model is **not** intended for standalone clinical diagnosis. It has been validated only on a single-center dataset and requires further multicenter validation before any clinical deployment.

---

## Architecture

```
Input: (batch, 45, H, W, 3) — 45-frame temporal window
  │
  ├── TimeDistributed(ResNet50, ImageNet pretrained)
  │     └── Last 20 layers fine-tuned
  ├── TimeDistributed(GlobalAveragePooling2D)
  ├── TimeDistributed(Dense(256, gelu))
  ├── TimeDistributed(BatchNormalization)
  ├── TimeDistributed(Dropout(0.5))
  │
  ├── Frame probability head: Dense(1, sigmoid) → p_i
  ├── Frame weight head: Dense(1, softplus) → w_i
  │
  └── WMV: output = Σ(p_i × w_i) / Σ(w_i)
```

**WMV (Weighted Mean Voting)**: Each frame produces a probability $p_i$ and an importance weight $w_i$. The sequence-level prediction is the weighted average, allowing the model to learn which frames are most informative for the classification decision.

---

## Training Details

### Hyperparameters

| Parameter | Value |
|---|---|
| Backbone | ResNet50 (ImageNet pretrained) |
| Trainable layers | Last 20 of ResNet50 + all dense heads |
| Input resolution | 128×128 / 160×160 (reviewer-compliant ablation runs) |
| Temporal window | 45 frames (~1.8 seconds at 25 fps) |
| Batch size | 8 |
| Optimizer | Adam |
| Learning rate | 1e-3 (max), 1e-4 (base) |
| LR schedule | OneCycleLR (pct_start=0.3) |
| Max epochs | 60 |
| Early stopping | patience=15, monitor=val_loss |
| Checkpoint | Best val_auc per fold |
| Loss | Binary cross-entropy |
| Activation | GELU |
| Dropout | 0.5 (backbone head), varies by layer |

### Cross-Validation

- **Method**: GroupKFold (scikit-learn), 10 folds
- **Grouping variable**: Patient ID (ensures no patient appears in both train and test)
- **Random seed**: 42

### Data Augmentation (when applied)

| Level | Rotation | Brightness | Contrast | Zoom | Flip |
|---|---|---|---|---|---|
| None | — | — | — | — | — |
| Conservative | ±5° | ±10% | ±10% | — | None |
| Moderate | ±10° | ±15% | ±15% | — | None |

---

## Dataset

| Statistic | Value |
|---|---|
| Patients | 24 |
| Total frames | 93,315 |
| Temporal sequences | 629 (322 closed, 307 open) |
| Frame rate | 25 fps |
| Center | Single tertiary academic children's hospital |
| Video type | Nasopharyngoscopy (NP) |

---

## Performance

### Primary Reported Result

| Metric | Value | 95% CI |
|---|---|---|
| Configuration | Naso-Net (WMV), 128×128, no augmentation, window=45 | — |
| Accuracy | 76.8% | [69.6%, 83.4%] |
| AUC | 77.5% | [70.7%, 85.2%] |
| F1 | 0.748 | — |

### Baseline Comparisons (128×128, no augmentation)

| Model | Acc (%) [95% CI] | AUC (%) [95% CI] | F1 | Fold AUC σ |
|---|---|---|---|---|
| ResNet50 + Mean Pool | 74.6 [66.5, 79.5] | 75.4 [68.4, 84.7] | 72.7 | 0.115 |
| ResNet50 + LSTM | 68.7 [61.9, 75.2] | 71.7 [64.6, 79.5] | 68.3 | 0.132 |
| **Naso-Net (ResNet50 + WMV)** | **76.8 [69.6, 83.4]** | **77.5 [70.7, 85.2]** | **74.8** | **0.094** |

### ResNet50 vs Naso-Net (sequence-level)

| Model | Accuracy (%) | AUC (%) | F1 (%) |
|---|---|---|---|
| ResNet50 | 68.29 | 72.5¹ | 70.39 |
| Naso-Net | 76.8 | 78.2 | 74.8 |

¹Baseline AUC for ResNet50 (spatial-only) measured on the same validation set.

### Ablation: Resolution × Augmentation (best result per config)

| Resolution | Augmentation | Accuracy [95% CI] | AUC [95% CI] |
|---|---|---|---|
| 128×128 | none | 0.768 [0.696, 0.834] | 0.775 [0.684, 0.847] |
| 128×128 | conservative | 0.753 [0.680, 0.827] | 0.755 [0.675, 0.825] |
| 128×128 | moderate | 0.750 [0.686, 0.824] | 0.758 [0.668, 0.820] |
| 160×160 | none | 0.763 [0.697, 0.829] | 0.749 [0.684, 0.832] |
| 160×160 | conservative | 0.766 [0.681, 0.831] | 0.776 [0.666, 0.859] |
| 160×160 | moderate | 0.759 [0.679, 0.836] | 0.761 [0.630, 0.847] |

---

## Limitations

- **Single-center data**: Model was trained and validated on data from one pediatric hospital. Generalization to other centers, patient populations, or endoscope hardware has not been tested.
- **Small sample size**: 24 patients / 629 sequences. Performance metrics have wide confidence intervals.
- **Binary classification only**: The model predicts open vs. closed; it does not characterize closure patterns (coronal, sagittal, circular) or degree of closure.
- **No audio integration**: Only visual frames are analyzed; speech audio is not used.

---

## Ethical Considerations

- The dataset was collected under IRB approval. Raw videos are not publicly shared due to patient privacy (HIPAA).
- The model is intended for research purposes only and should not be used for clinical decision-making without further validation.
- Annotation was performed by trained experts; inter-rater variability was not formally quantified.

---

## Contact

For questions, data access requests, or collaboration inquiries, please contact the corresponding author:
- **Miles J. Pfaff, MD** — mpfaff@hs.uci.edu
