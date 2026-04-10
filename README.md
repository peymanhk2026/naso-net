# Naso-Net: Automated Detection of Velopharyngeal Port Dynamics from Nasopharyngoscopy Videos Using Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the source code, annotations, and trained model weights for **Naso-Net**, a temporal-aware convolutional neural network for automated prediction of velopharyngeal port (VPP) closure status from nasopharyngoscopy (NP) video clips.

> **Paper**: *Automated detection of velopharyngeal port dynamics from nasopharyngoscopy videos using deep learning*  
> PLOS ONE (2026)

---

## Overview

Naso-Net is a sequence-level deep learning model built on a **time-distributed ResNet50** backbone with a **Weighted Mean Voting (WMV)** aggregation layer. It takes a temporal sliding window of video frames as input and predicts whether the velopharyngeal port is open or closed during that sequence.

**Selected primary result**: The primary reported configuration in this repository is **Naso-Net (WMV), 128×128, no augmentation**, which achieved the best overall performance among the models summarized here.

**Key results** (10-fold patient-wise cross-validation, 128×128, no augmentation):

| Model | Acc (%) [95% CI] | AUC (%) [95% CI] | F1 | Fold AUC sigma |
|-------|-------------------|------------------|----|----------------|
| ResNet50 + Mean Pool | 74.6 [66.5, 79.5] | 75.4 [68.4, 84.7] | 72.7 | 0.115 |
| ResNet50 + LSTM | 68.7 [61.9, 75.2] | 71.7 [64.6, 79.5] | 68.3 | 0.132 |
| **Naso-Net (ResNet50 + WMV)** | **76.8 [69.6, 83.4]** | **77.5 [70.7, 85.2]** | **74.8** | **0.094** |

**ResNet50 vs Naso-Net** (sequence-level, 128×128, no augmentation):

| Model    | Accuracy (%) | AUC (%)  | F1 (%) |
|----------|-------------|----------|--------|
| ResNet50 | 68.29       | 72.5     | 70.39  |
| Naso-Net | 76.8        | 78.2     | 74.8   |

Naso-Net outperforms the standard ResNet50, with accuracy increased by nearly 8.5 percentage points (from 68.29% to 76.8%) and AUC improved by 5.7 percentage points (from 72.5% to 78.2%). This improvement results from end-to-end temporal modelling: while ResNet50 exploits only spatial features in individual frames, Naso-Net's learned frame-weighting mechanism captures the dynamic sequence of velar movement, yielding greater discriminative power for VPP classification.

**Primary reported WMV result** (128×128, no augmentation, window=45):
- **Accuracy**: 76.8% [69.6%, 83.4%]
- **AUC**: 77.5% [70.7%, 85.2%]
- **F1**: 0.748

---

## Repository Structure

```
naso-net/
├── README.md                 # This file
├── MODEL_CARD.md             # Model card with full specifications
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── .gitignore
│
├── scripts/
│   ├── extract_frames.py               # Step 1: Extract frames from NP videos
│   ├── extract_sequences.py            # Step 2: Build temporal sequences
│   ├── naso_net_train.py               # Step 3: Train Naso-Net (WMV)
│   ├── naso_net_eval.py                # Step 4: Evaluate trained model
│   ├── ablation_resolution_augmentation.py  # Ablation study (resolution × augmentation)
│   ├── frame_weight_analysis.py        # Frame weight interpretability analysis
│   ├── baseline_mean_pooling.py        # ResNet50+MeanPool baseline
│   ├── baseline_lstm.py               # ResNet50+LSTM baseline
│   └── bootstrap_patient_ci.py         # Patient-level bootstrap CIs
│
├── annotations/
│   ├── VPI_14-15_17_19.json            # Expert annotations (keyframes + bounding boxes)
│   ├── VPI_21-22.json
│   ├── VPI_28-31_33.json
│   ├── VPI_35+37.json
│   └── 4th-attempt/
│       ├── VPI_1_and_5-13-sequence-jmin.json
│       └── VPI2-4_sequence-jmin.json
│
├── weights/
│   ├── naso_net_fold1.weights.h5       # Best weights, fold 1 (5 MB)
│   └── naso_net_fold2.weights.h5       # Best weights, fold 2 (5 MB)
│
└── results/
  └── frame_weight_analysis.png       # Frame weight visualization
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)
- ~10 GB disk space for weights and data

### Installation

```bash
git clone https://github.com/peyman-HK/naso-net.git
cd naso-net
pip install -r requirements.txt
```

### Data Preparation

1. **Extract frames** from NP video files:
   ```bash
   python scripts/extract_frames.py
   ```

2. **Build temporal sequences** (contiguous open/closed clips):
   ```bash
   python scripts/extract_sequences.py
   ```

The expected data layout after extraction:
```
extracted_sequences/
  VPI-1/
    pos_1-165/    # Frames 1-165 (closed VPP)
    neg_166-219/  # Frames 166-219 (open VPP)
    ...
  VPI-2/
    ...
```

### Training

```bash
python scripts/naso_net_train.py
```

Key training parameters (configurable in script):
- Resolution: 128×128 or 160×160 for the reviewer-compliant ablation runs
- Sliding window: 45 frames (~1.8s at 25 fps)
- Max epochs: 60, early stopping patience: 15
- Learning rate: 1e-3 with OneCycleLR
- Cross-validation: 10-fold patient-wise (GroupKFold)

### Evaluation

```bash
python scripts/naso_net_eval.py
```

### Ablation Study

```bash
python scripts/ablation_resolution_augmentation.py
```

Runs a grid of {90×90, 128×128, 160×160} × {none, conservative, moderate} augmentation configurations.

### Baseline Comparison

```bash
python scripts/baseline_mean_pooling.py    # ResNet50 + Mean Pooling
python scripts/baseline_lstm.py            # ResNet50 + LSTM
```

---

## Dataset

The dataset consists of 24 NP video clips from 24 pediatric patients (629 temporal sequences, 93,315 frames) collected at a single tertiary academic children's hospital. Due to patient privacy regulations (HIPAA/IRB), the raw video data cannot be publicly shared.

**What is included in this repository:**
- Expert annotation files (JSON) with temporal keyframes and bounding boxes
- Trained model weights for reproducibility
- All training and evaluation scripts

**Controlled access:** Requests for the de-identified video data for research purposes may be directed to the corresponding author, subject to institutional data use agreement and IRB approval.

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{kassani2026nasonet,
  title={Automated detection of velopharyngeal port dynamics from 
         nasopharyngoscopy videos using deep learning},
  author={Kassani, Peyman H. and Willens, Sierra and Trivedi, Shivang 
          and Miao, Xinfei C. and Humphrey, JaNeil G. and Perry, Jamie L. 
          and Pfaff, Miles J.},
  journal={PLOS ONE},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
