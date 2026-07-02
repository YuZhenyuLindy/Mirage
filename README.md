# Mirage: Representation-Level Certification of Visual Unlearning

Code for the ECCV 2026 paper *Do Vision Models Truly Forget? New Findings from Representation-Level Certification of Visual Unlearning in Vertical Federated Learning*.

Mirage checks whether an unlearned vision model has actually removed a class from its
representations, rather than only suppressing it at the output. It compares the original
model, the unlearned model, and a from-scratch retrained reference on their frozen
embeddings, reporting linear probe recovery (LPR) together with its gap to the retrained
baseline, centered kernel alignment (CKA), a Fisher-style separability score, and a
layer-wise recovery profile. The recurring observation is a *forgetting illusion*: methods
that pass output-level certification often leave the forgotten class linearly recoverable
in feature space, well above what retraining alone would preserve.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch 1.12+.

## Data

MNIST, CIFAR-10 and CIFAR-100 are downloaded automatically through torchvision. The other
four datasets (ModelNet10, Brain Tumor MRI, COVID-19 Radiography, Yahoo Answers) are
fetched with:

```bash
python prepare_datasets.py
```

ModelNet10 and the medical sets can be cached to tensors for faster reloading:

```bash
python cache_modelnet10.py
python cache_datasets.py
```

## Running the experiments

Main comparison — seven datasets, eight methods:

```bash
python run_main.py                              # all datasets
python run_main.py --datasets CIFAR10 COVID19   # a subset
```

Ablations and analyses:

```bash
python run_bu_epoch_ablation.py   # BU unlearning strength (Table 4)
python run_layerwise.py           # layer-wise LPR (Appendix A2)
python run_perclass.py            # per-class forgetting gap (Appendix A3)
python run_nonlinear_probe.py     # linear vs. MLP probe (Appendix A4)
python run_timing.py              # audit cost (Appendix A5)
python run_class_ablation.py      # sensitivity to the forgotten class
```

Sample-level unlearning, K-party scaling, and the t-SNE figures are driven by a single
entry point:

```bash
python run_extended.py --exp all --device cuda
```

A self-contained walkthrough of the full pipeline is provided in
`notebooks/Mirage_ECCV_Colab.ipynb`.

## Datasets

| Dataset | Domain | Classes | Backbone |
|---------|--------|---------|----------|
| MNIST | Handwritten digits | 10 | ResNet-18 |
| CIFAR-10 | Natural images | 10 | ResNet-18 |
| CIFAR-100 | Natural images | 100 | ResNet-18 |
| ModelNet10 | 3D objects (depth) | 10 | ResNet-18 |
| Brain Tumor MRI | Medical imaging | 4 | ResNet-18 |
| COVID-19 Radiography | Medical imaging | 4 | ResNet-18 |
| Yahoo Answers | Text (TF-IDF) | 10 | MLP |

Features are split equally between two passive parties (VFL setting); the active party
holds the labels and the top classifier.

## Unlearning methods

| Method | Description |
|--------|-------------|
| Retrain | Retrain from scratch on the retained data (reference) |
| Fine-Tuning (FT) | Fine-tune the trained model on retained data (Golatkar et al., 2020) |
| Fisher | Fisher-information forgetting (Golatkar et al., 2020) |
| Amnesiac | Amnesiac unlearning (Graves et al., 2021) |
| UNSIR | Impair–repair unlearning (Tarun et al., 2023) |
| Boundary Unlearning (BU) | Decision-boundary shifting (Chen et al., 2023) |
| SSD | Selective synaptic dampening (Foster et al., 2024) |
| Target | Few-shot label unlearning via manifold mixup (Gu et al., 2026) |

## Citation

```bibtex
@inproceedings{yu2026mirage,
  title={Do Vision Models Truly Forget? New Findings from Representation-Level Certification of Visual Unlearning in Vertical Federated Learning},
  author={Yu, Zhenyu and Zeng, Yangchen and Meng, Chunlei and Yao, Guangzhen and Zhou, Shuigeng},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```
