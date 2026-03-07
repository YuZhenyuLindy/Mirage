# Mirage: Representation-Level Certification of Visual Unlearning

Official implementation of **"Can Vision Models Truly Forget? *Mirage*: Representation-Level Certification of Visual Unlearning"** (ECCV 2026).

## Overview

Mirage is a representation-level auditing framework that exposes the *forgetting illusion* in Vertical Federated Learning (VFL) unlearning. While existing methods certify forgetting based on output-level metrics (e.g., accuracy on forgotten classes), Mirage reveals that class-discriminative structure persists in intermediate representations.

The framework comprises four complementary diagnostics:

- **Linear Probe Recovery (LPR)**: Binary classification accuracy of a logistic regression probe on frozen features
- **Centered Kernel Alignment (CKA)**: Structural similarity between model representations
- **Feature Separability Scoring**: Fisher-inspired geometric class discrimination metric
- **Layer-Wise Recovery Analysis**: LPR at multiple network depths

### Key Findings

1. **Forgetting gap**: Methods that pass output-level certification still retain substantial class structure (LPR exceeding retrained baseline by up to 15.4 pp)
2. **Unlearning trilemma**: No method simultaneously achieves utility, output-level forgetting, and representation-level forgetting
3. **Class-sample asymmetry**: Class-level forgetting leaves strong traces (LPR up to 97%), while sample-level forgetting is indistinguishable from chance (~50%)

## Setup

```bash
pip install -r requirements.txt
```

### Dataset Preparation

CIFAR-10, CIFAR-100, and MNIST are downloaded automatically via torchvision. For the remaining datasets (ModelNet10, BrainTumor, COVID-19, Yahoo Answers):

```bash
python prepare_datasets.py
```

For faster I/O in repeated experiments, optionally pre-cache slow datasets:

```bash
python cache_modelnet10.py
python cache_datasets.py
```

## Experiments

All experiment scripts save results to the `results/` directory.

### Main Experiments (Tables 1 & 2)

7 datasets x 8 unlearning methods x 3 seeds:

```bash
python run_main.py
python run_main.py --datasets CIFAR10 COVID19  # subset
```

### Ablation Studies

| Script | Description | Paper Section |
|--------|-------------|---------------|
| `run_layerwise.py` | Layer-wise LPR at 3 network depths | Appendix B |
| `run_perclass.py` | Per-class forgetting gap analysis | Appendix C |
| `run_nonlinear_probe.py` | Linear vs MLP probe comparison | Appendix D |
| `run_timing.py` | Computational cost measurement | Appendix E |
| `run_bu_epoch_ablation.py` | BU epoch ablation {1,3,5,10,20} | Table 4 |
| `run_class_ablation.py` | Forget-class ablation across methods | Supplementary |

### Extended Experiments

```bash
python run_extended.py --exp all --device cuda
```

Includes: sample-level unlearning, K-party ablation, t-SNE visualization, epoch ablation.

## Project Structure

```
mirage-release/
├── mirage_lib.py              # Core library: models, datasets, training, unlearning, audit
├── fast_datasets.py           # Cached dataset loaders (monkey-patches mirage_lib)
├── prepare_datasets.py        # Download ModelNet10, BrainTumor, COVID-19, Yahoo Answers
├── cache_datasets.py          # Pre-cache slow datasets to .pt files
├── cache_modelnet10.py        # Pre-render ModelNet10 meshes to tensors
├── run_main.py                # Main experiment (Tables 1 & 2)
├── run_layerwise.py           # Layer-wise recovery analysis
├── run_perclass.py            # Per-class forgetting gap
├── run_nonlinear_probe.py     # Nonlinear probe comparison
├── run_timing.py              # Computational cost
├── run_bu_epoch_ablation.py   # BU epoch ablation
├── run_class_ablation.py      # Class ablation across methods
├── run_extended.py            # Extended experiments (sample-level, K-party, t-SNE)
├── notebooks/
│   └── Mirage_ECCV_Colab.ipynb  # Complete pipeline in notebook format
├── results/                   # Pre-computed experiment results (CSV)
├── requirements.txt
└── .gitignore
```

## Datasets

| Dataset | Domain | Classes | Architecture |
|---------|--------|---------|--------------|
| MNIST | Digits | 10 | ResNet-18 |
| CIFAR-10 | Natural images | 10 | ResNet-18 |
| CIFAR-100 | Natural images | 100 | ResNet-18 |
| ModelNet10 | 3D objects | 10 | ResNet-18 |
| BrainTumor | Medical MRI | 4 | ResNet-18 |
| COVID-19 | Medical X-ray | 4 | ResNet-18 |
| Yahoo Answers | Text (TF-IDF) | 10 | MLP |

## Unlearning Methods

| Method | Reference |
|--------|-----------|
| Retrain | Baseline (retrain from scratch) |
| Fine-Tuning (FT) | Continue training without forget data |
| Fisher | Diagonal Fisher information erasure |
| Amnesiac | Class-wise logit subtraction |
| UNSIR | Saliency-guided unlearning |
| Boundary Unlearning (BU) | Decision boundary shifting |
| SSD | Sample-wise signed erasure |
| Target | Maximize logit for forget class |

## Citation

```bibtex
@inproceedings{mirage2026,
  title={Can Vision Models Truly Forget? Mirage: Representation-Level Certification of Visual Unlearning},
  author={Anonymous},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```
