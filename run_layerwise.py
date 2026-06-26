#!/usr/bin/env python3
"""
Layer-wise Recovery Analysis (Appendix B)
Extract features at each internal layer of the passive models,
then run LPR at each layer to show WHERE information is retained.

ResNet18Bottom structure (nn.Sequential):
  Block A (64-ch):  [0] Conv  [1] BN  [2] ReLU  [3] Conv  [4] BN  [5] ReLU  [6] MaxPool
  Block B (128-ch): [7] Conv  [8] BN  [9] ReLU  [10] Conv [11] BN [12] ReLU [13] MaxPool
  Block C (256-ch): [14] Conv [15] BN [16] ReLU [17] Conv [18] BN [19] ReLU [20] AvgPool

We tap after each ReLU following a Conv pair: indices 5, 12, 19
(i.e. after Block A, Block B, Block C before pooling layers).

For MLP/tabular models we skip this experiment (not enough layers).

Datasets: CIFAR10, CIFAR100, COVID19 (image datasets with resnet18)
Methods:  Retrain, FT, BU, Target
Seeds:    42, 123, 456

Output: results/layerwise_{dataset}.csv
        results/layerwise_all.csv

Usage:

    python run_layerwise.py
"""

import os, sys, gc, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mirage_lib import (
    set_seed, CONFIG,
    get_dataset, VFLDataset, prepare_unlearning_data,
    create_vfl_model, train_vfl, evaluate_accuracy,
    extract_features, linear_probe_recovery,
    fine_tuning_unlearn, boundary_unlearn,
    manifold_mixup_vfl_unlearn, _free_model,
)
from torch.utils.data import DataLoader

SEEDS = [42, 123, 456]
DATASETS = [
    ('CIFAR10',  'resnet18'),
    ('CIFAR100', 'resnet18'),
    ('COVID19',  'resnet18'),
]

# Tap points: after ReLU at end of each conv block (before pool)
# These are indices into ResNet18Bottom.features (nn.Sequential)
TAP_POINTS = {
    'block_A': 5,   # after 2nd ReLU, 64-ch
    'block_B': 12,  # after 2nd ReLU, 128-ch
    'block_C': 19,  # after 2nd ReLU, 256-ch (final)
}

def extract_layerwise_features(model, loader, device, tap_index):
    """Extract features from a specific layer of each passive model.

    For each passive model k, we run the input through
    model.passive_models[k].features[:tap_index+1], then
    AdaptiveAvgPool + flatten to get a fixed-size vector.
    Concatenate across parties.
    """
    model.eval()
    pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
    all_feats, all_labs = [], []

    with torch.no_grad():
        for px, la in loader:
            px = [x.to(device) for x in px]
            party_feats = []
            for k, x_k in enumerate(px):
                # Run through partial Sequential
                h = x_k
                for i, layer in enumerate(model.passive_models[k].features):
                    h = layer(h)
                    if i == tap_index:
                        break
                # Pool + flatten
                h = pool(h).view(h.size(0), -1)
                party_feats.append(h)
            cat = torch.cat(party_feats, dim=1)
            all_feats.append(cat.cpu().numpy())
            all_labs.append(la.numpy())

    features = np.concatenate(all_feats)
    labels = np.concatenate(all_labs)
    if not np.isfinite(features).all():
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features, labels

def lpr_from_features(features, labels, unlearn_labels, seed=42):
    """Run LPR on pre-extracted features (same logic as mirage_lib)."""
    bl = np.isin(labels, unlearn_labels).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        features, bl, test_size=0.3, random_state=seed, stratify=bl)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    clf = LogisticRegression(
        C=CONFIG['probe_C'], max_iter=CONFIG['probe_max_iter'],
        random_state=seed, solver='lbfgs')
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xte)
    acc = balanced_accuracy_score(yte, preds) * 100.
    try:
        auroc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    except:
        auroc = 0.5
    return acc, auroc

def _cleanup(device, *models):
    for m in models:
        if m is not None:
            try: _free_model(m, device)
            except: pass
    gc.collect()
    if 'cuda' in str(device): torch.cuda.empty_cache()

def run_one_dataset(ds_name, default_arch, device, data_root):
    print(f"  {ds_name} — Layer-wise Recovery")

    unlearn_labels = [0]
    rows = []

    for i, seed in enumerate(SEEDS):
        set_seed(seed)
        print(f"\n  --- Seed {seed} ({i+1}/{len(SEEDS)}) ---")

        train_ds, test_ds, num_classes, in_ch, img_size = get_dataset(ds_name, data_root)
        np_ = CONFIG['num_passive_parties']
        input_width = img_size // np_
        bs = CONFIG['batch_size']

        Dr_train, Du_train, Dp_u, Dp_r, Dr_test, Du_test = \
            prepare_unlearning_data(train_ds, test_ds, unlearn_labels,
                                    CONFIG['num_public_samples'])

        vfl_train = VFLDataset(train_ds, np_)
        vfl_test  = VFLDataset(test_ds, np_)
        vfl_Dr    = VFLDataset(Dr_train, np_)
        vfl_Du    = VFLDataset(Du_train, np_)
        vfl_Dpu   = VFLDataset(Dp_u, np_)
        vfl_Dpr   = VFLDataset(Dp_r, np_)

        train_loader     = DataLoader(vfl_train, bs, shuffle=True, num_workers=2, pin_memory=True)
        Dr_loader        = DataLoader(vfl_Dr, bs, shuffle=True, num_workers=2, pin_memory=True)
        Du_loader        = DataLoader(vfl_Du, bs, shuffle=True, num_workers=2, pin_memory=True)
        Dp_u_loader      = DataLoader(vfl_Dpu, min(bs, len(Dp_u)), shuffle=True, num_workers=2, pin_memory=True)
        Dp_r_loader      = DataLoader(vfl_Dpr, min(bs, len(Dp_r)), shuffle=True, num_workers=2, pin_memory=True)
        full_test_loader = DataLoader(vfl_test, bs, shuffle=False, num_workers=2, pin_memory=True)

        ul_lr = 0.001

        try:
            # Train original
            print("    Training original...")
            orig = create_vfl_model(default_arch, in_ch, input_width, np_, num_classes)
            orig = train_vfl(orig, train_loader, CONFIG['train_epochs'],
                             CONFIG['lr'], device, verbose=True)

            # Retrain
            print("    Retraining...")
            retrained = create_vfl_model(default_arch, in_ch, input_width, np_, num_classes)
            retrained = train_vfl(retrained, Dr_loader, CONFIG['train_epochs'],
                                  CONFIG['lr'], device, verbose=True)

            # Unlearning methods
            print("    Running FT...")
            ft_mdl = fine_tuning_unlearn(orig, Dr_loader, 5, ul_lr, device)
            print("    Running BU...")
            bu_mdl = boundary_unlearn(orig, Du_loader, Dr_loader, 5, ul_lr, device)
            print("    Running Target...")
            target_mdl = manifold_mixup_vfl_unlearn(
                orig, Dp_u_loader, Dp_r_loader,
                CONFIG['unlearn_epochs'], CONFIG['unlearn_lr'],
                CONFIG['mixup_alpha'], device)

            # Evaluate at each layer
            for name, mdl in [('Retrain', retrained), ('FT', ft_mdl),
                              ('BU', bu_mdl), ('Target', target_mdl)]:
                # Final layer (concatenated) — standard LPR
                fm_final, lm = extract_features(mdl, full_test_loader, device)
                lpr_final, _ = linear_probe_recovery(
                    mdl, full_test_loader, unlearn_labels, device,
                    cached_features=(fm_final, lm))
                rows.append(dict(dataset=ds_name, seed=seed, method=name,
                                 layer='final', lpr_acc=lpr_final))

                # Intermediate layers
                for layer_name, tap_idx in TAP_POINTS.items():
                    fm_layer, lm_layer = extract_layerwise_features(
                        mdl, full_test_loader, device, tap_idx)
                    lpr_layer, _ = lpr_from_features(
                        fm_layer, lm_layer, unlearn_labels, seed)
                    rows.append(dict(dataset=ds_name, seed=seed, method=name,
                                     layer=layer_name, lpr_acc=lpr_layer))
                    del fm_layer, lm_layer

                print(f"    {name}: " +
                      " | ".join(f"{r['layer']}={r['lpr_acc']:.1f}%"
                                 for r in rows if r['seed'] == seed
                                 and r['method'] == name))
                del fm_final, lm

            _cleanup(device, target_mdl, bu_mdl, ft_mdl, retrained, orig)

        except Exception as e:
            print(f"    [ERROR] {ds_name} seed={seed}: {e}")
            import traceback; traceback.print_exc()

    return rows

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = "./data_raw"

    print("=" * 60)
    print("  LAYER-WISE RECOVERY (Appendix B)")
    print(f"  Datasets: {[d[0] for d in DATASETS]}")
    print(f"  Tap points: {list(TAP_POINTS.keys())}")
    print(f"  Seeds:    {SEEDS}")
    print(f"  Device:   {device}")
    print(f"  Start:    {datetime.now()}")
    print("=" * 60)

    os.makedirs('results', exist_ok=True)
    all_rows = []

    for ds_name, arch in DATASETS:
        ds_rows = run_one_dataset(ds_name, arch, device, data_root)
        all_rows.extend(ds_rows)

        df = pd.DataFrame(ds_rows)
        path = f'results/layerwise_{ds_name}.csv'
        df.to_csv(path, index=False)
        print(f"  Saved {path}")

    df_all = pd.DataFrame(all_rows)
    path = 'results/layerwise_all.csv'
    df_all.to_csv(path, index=False)
    print(f"\nSaved {path} ({len(df_all)} rows)")

    # Summary
    print("\n" + "=" * 60)
    print("  Layer-wise Summary (mean over seeds)")
    print("=" * 60)
    for ds_name, _ in DATASETS:
        df_ds = df_all[df_all['dataset'] == ds_name]
        print(f"\n  {ds_name}:")
        for method in ['Retrain', 'FT', 'BU', 'Target']:
            sub = df_ds[df_ds['method'] == method]
            line = f"    {method:8s}:"
            for layer in ['block_A', 'block_B', 'block_C', 'final']:
                vals = sub[sub['layer'] == layer]['lpr_acc']
                if len(vals) > 0:
                    line += f"  {layer}={vals.mean():.1f}%"
            print(line)

    print(f"\nDone: {datetime.now()}")

if __name__ == "__main__":
    main()
