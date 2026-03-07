#!/usr/bin/env python3
"""
BU Epoch Ablation (supplement to Table 4)
==========================================
Vary boundary_unlearn epochs {1,3,5,10,20} — same grid as the existing
Target epoch ablation — to answer:

  "Can BU's forgetting illusion be eliminated by increasing epochs?"

Datasets: CIFAR-10, CIFAR-100, COVID-19 (same as Table 4)
Seeds:    42, 123, 456

Output: results/bu_epoch_ablation_{dataset}.csv
        results/bu_epoch_ablation_all.csv

Usage:

    python run_bu_epoch_ablation.py
"""

import os, sys, gc, time
import numpy as np
import pandas as pd
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mirage_lib import (
    set_seed, CONFIG,
    get_dataset, VFLDataset, prepare_unlearning_data,
    create_vfl_model, train_vfl, evaluate_accuracy,
    extract_features, linear_probe_recovery,
    feature_separability, boundary_unlearn, _free_model,
)
from torch.utils.data import DataLoader

SEEDS = [42, 123, 456]
DATASETS = [
    ('CIFAR10',  'resnet18'),
    ('CIFAR100', 'resnet18'),
    ('COVID19',  'resnet18'),
]
EPOCH_VALUES = [1, 3, 5, 10, 20]


def _cleanup(device, *models):
    for m in models:
        if m is not None:
            try: _free_model(m, device)
            except: pass
    gc.collect()
    if 'cuda' in str(device): torch.cuda.empty_cache()


def run_one_dataset(ds_name, default_arch, device, data_root):
    print(f"\n{'='*60}")
    print(f"  {ds_name} — BU Epoch Ablation")
    print(f"{'='*60}")

    unlearn_labels = [0]
    rows = []

    for i, seed in enumerate(SEEDS):
        set_seed(seed)
        print(f"\n  --- Seed {seed} ({i+1}/{len(SEEDS)}) ---")

        train_ds, test_ds, num_classes, in_ch, img_size = get_dataset(ds_name, data_root)
        is_tabular = (in_ch == 0)
        arch = 'mlp' if is_tabular else default_arch
        train_lr = 1e-3 if is_tabular else CONFIG['lr']
        np_ = CONFIG['num_passive_parties']
        input_width = img_size // np_
        bs = CONFIG['batch_size']

        # Prepare data
        Dr_train, Du_train, Dp_u, Dp_r, Dr_test, Du_test = \
            prepare_unlearning_data(train_ds, test_ds, unlearn_labels,
                                    CONFIG['num_public_samples'])
        vfl_train = VFLDataset(train_ds, np_)
        vfl_test = VFLDataset(test_ds, np_)
        vfl_Dr = VFLDataset(Dr_train, np_)
        vfl_Du = VFLDataset(Du_train, np_)
        vfl_DrT = VFLDataset(Dr_test, np_)
        vfl_DuT = VFLDataset(Du_test, np_)

        train_loader = DataLoader(vfl_train, bs, shuffle=True, num_workers=2, pin_memory=True)
        Dr_loader = DataLoader(vfl_Dr, bs, shuffle=True, num_workers=2, pin_memory=True)
        Du_loader = DataLoader(vfl_Du, bs, shuffle=True, num_workers=2, pin_memory=True)
        Dr_test_loader = DataLoader(vfl_DrT, bs, shuffle=False, num_workers=2, pin_memory=True)
        Du_test_loader = DataLoader(vfl_DuT, bs, shuffle=False, num_workers=2, pin_memory=True)
        full_test_loader = DataLoader(vfl_test, bs, shuffle=False, num_workers=2, pin_memory=True)

        try:
            # Train original (shared across epoch values)
            print("    Training original...")
            orig = create_vfl_model(arch, in_ch, input_width, np_, num_classes)
            orig = train_vfl(orig, train_loader, CONFIG['train_epochs'],
                             train_lr, device, verbose=True, use_adam=is_tabular)

            # Retrain baseline (shared)
            print("    Retraining...")
            retrained = create_vfl_model(arch, in_ch, input_width, np_, num_classes)
            retrained = train_vfl(retrained, Dr_loader, CONFIG['train_epochs'],
                                  train_lr, device, verbose=True, use_adam=is_tabular)

            # Retrain baseline metrics
            fm_r, lm_r = extract_features(retrained, full_test_loader, device)
            lpr_r, _ = linear_probe_recovery(
                retrained, full_test_loader, unlearn_labels, device,
                cached_features=(fm_r, lm_r))
            dr_r = evaluate_accuracy(retrained, Dr_test_loader, device)
            yu_r = evaluate_accuracy(retrained, Du_test_loader, device)
            sep_r = feature_separability(fm_r, lm_r, unlearn_labels)
            rows.append(dict(dataset=ds_name, epochs=0, seed=seed,
                             method='Retrain', dr_acc=dr_r, yu_acc=yu_r,
                             lpr_acc=lpr_r, separability=sep_r))
            print(f"    Retrain: Dr={dr_r:.1f}% LPR={lpr_r:.1f}%")
            del fm_r, lm_r

            # Vary BU epochs
            ul_lr = 1e-4 if is_tabular else 0.001
            for n_ep in EPOCH_VALUES:
                bu_mdl = boundary_unlearn(
                    orig, Du_loader, Dr_loader,
                    epochs=n_ep, lr=ul_lr,
                    device=device, use_adam=is_tabular)

                dr = evaluate_accuracy(bu_mdl, Dr_test_loader, device)
                yu = evaluate_accuracy(bu_mdl, Du_test_loader, device)
                fm, lm = extract_features(bu_mdl, full_test_loader, device)
                lpr_acc, _ = linear_probe_recovery(
                    bu_mdl, full_test_loader, unlearn_labels, device,
                    cached_features=(fm, lm))
                sep = feature_separability(fm, lm, unlearn_labels)

                delta = lpr_acc - lpr_r
                rows.append(dict(dataset=ds_name, epochs=n_ep, seed=seed,
                                 method='BU', dr_acc=dr, yu_acc=yu,
                                 lpr_acc=lpr_acc, separability=sep))
                print(f"    BU(ep={n_ep:2d}): Dr={dr:.1f}% yu={yu:.1f}% "
                      f"LPR={lpr_acc:.1f}% Δ={delta:+.1f}")

                _free_model(bu_mdl, device); del fm, lm

            _cleanup(device, retrained, orig)

        except Exception as e:
            print(f"    [ERROR] {ds_name} seed={seed}: {e}")
            import traceback; traceback.print_exc()

    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = "./data_raw"

    print("=" * 60)
    print("  BU EPOCH ABLATION (supplement to Table 4)")
    print(f"  Datasets: {[d[0] for d in DATASETS]}")
    print(f"  Epochs:   {EPOCH_VALUES}")
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
        path = f'results/bu_epoch_ablation_{ds_name}.csv'
        df.to_csv(path, index=False)
        print(f"  Saved {path}")

    # Save combined
    df_all = pd.DataFrame(all_rows)
    path = 'results/bu_epoch_ablation_all.csv'
    df_all.to_csv(path, index=False)
    print(f"\nSaved {path} ({len(df_all)} rows)")

    # Summary
    print("\n" + "=" * 60)
    print("  BU Epoch Ablation Summary (mean over seeds)")
    print("=" * 60)
    for ds_name, _ in DATASETS:
        df_ds = df_all[df_all['dataset'] == ds_name]
        ret_lpr = df_ds[df_ds['method'] == 'Retrain']['lpr_acc'].mean()
        print(f"\n  {ds_name} (Retrain LPR={ret_lpr:.1f}%):")
        for ep in EPOCH_VALUES:
            sub = df_ds[(df_ds['method'] == 'BU') & (df_ds['epochs'] == ep)]
            if len(sub) == 0: continue
            dr_m = sub['dr_acc'].mean()
            lpr_m = sub['lpr_acc'].mean()
            lpr_s = sub['lpr_acc'].std()
            delta = lpr_m - ret_lpr
            print(f"    ep={ep:2d}: Dr={dr_m:.1f}% LPR={lpr_m:.1f}±{lpr_s:.1f} Δ={delta:+.1f}")

    print(f"\nDone: {datetime.now()}")


if __name__ == "__main__":
    main()
