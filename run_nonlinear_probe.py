#!/usr/bin/env python3
"""
Nonlinear Probe Recovery (Appendix D)
======================================
Replace linear probe (LogisticRegression) with MLP probe to show
that the LPR gap is not an artifact of linear probe capacity.

Datasets: All 7
Methods:  Retrain, FT, BU, Target
Seeds:    42, 123, 456

Output: results/nonlinear_probe_{dataset}.csv / results/nonlinear_probe_all.csv

Usage:

    python run_nonlinear_probe.py
"""

import argparse, os, sys, gc
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
    fine_tuning_unlearn, boundary_unlearn,
    manifold_mixup_vfl_unlearn, _free_model,
)
from torch.utils.data import DataLoader

SEEDS = [42, 123, 456]
ALL_DATASETS = [
    ('MNIST',        'resnet18'),
    ('CIFAR10',      'resnet18'),
    ('CIFAR100',     'resnet18'),
    ('ModelNet10',   'resnet18'),
    ('BrainTumor',   'resnet18'),
    ('COVID19',      'resnet18'),
    ('YahooAnswers', 'mlp'),
]


def _cleanup(device, *models):
    for m in models:
        if m is not None:
            try: _free_model(m, device)
            except: pass
    gc.collect()
    if 'cuda' in str(device): torch.cuda.empty_cache()


def run_one_dataset(ds_name, default_arch, device, data_root):
    print(f"\n{'='*60}")
    print(f"  {ds_name} — Nonlinear Probe")
    print(f"{'='*60}")

    unlearn_labels = [0]
    rows = []

    for i, seed in enumerate(SEEDS):
        set_seed(seed)
        print(f"\n  --- Seed {seed} ({i+1}/{len(SEEDS)}) ---")

        try:
            train_ds, test_ds, num_classes, in_ch, img_size = get_dataset(
                ds_name, data_root)
            is_tabular = (in_ch == 0)
            arch = 'mlp' if is_tabular else default_arch
            train_lr = 1e-3 if is_tabular else CONFIG['lr']
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

            ul_lr = 1e-4 if is_tabular else 0.001

            # Train original
            print("    Training original...")
            orig = create_vfl_model(arch, in_ch, input_width, np_, num_classes)
            orig = train_vfl(orig, train_loader, CONFIG['train_epochs'],
                             train_lr, device, verbose=True, use_adam=is_tabular)

            # Retrain
            print("    Retraining...")
            retrained = create_vfl_model(arch, in_ch, input_width, np_, num_classes)
            retrained = train_vfl(retrained, Dr_loader, CONFIG['train_epochs'],
                                  train_lr, device, verbose=True, use_adam=is_tabular)

            # Unlearning
            print("    Running FT/BU/Target...")
            ft_mdl = fine_tuning_unlearn(orig, Dr_loader, 5, ul_lr, device, is_tabular)
            bu_mdl = boundary_unlearn(orig, Du_loader, Dr_loader, 5, ul_lr, device, is_tabular)
            target_mdl = manifold_mixup_vfl_unlearn(
                orig, Dp_u_loader, Dp_r_loader,
                CONFIG['unlearn_epochs'],
                ul_lr * 10 if is_tabular else CONFIG['unlearn_lr'],
                CONFIG['mixup_alpha'], device, is_tabular)

            # Evaluate with BOTH linear and nonlinear probes
            for name, mdl in [('Retrain', retrained), ('FT', ft_mdl),
                              ('BU', bu_mdl), ('Target', target_mdl)]:
                fm, lm = extract_features(mdl, full_test_loader, device)

                # Linear probe
                lpr_lin, auc_lin = linear_probe_recovery(
                    mdl, full_test_loader, unlearn_labels, device,
                    probe_type='linear', cached_features=(fm, lm))

                # Nonlinear (MLP) probe
                lpr_mlp, auc_mlp = linear_probe_recovery(
                    mdl, full_test_loader, unlearn_labels, device,
                    probe_type='mlp', cached_features=(fm, lm))

                rows.append(dict(
                    dataset=ds_name, seed=seed, method=name,
                    lpr_linear=lpr_lin, auroc_linear=auc_lin,
                    lpr_mlp=lpr_mlp, auroc_mlp=auc_mlp))
                print(f"    {name}: Linear={lpr_lin:.1f}% MLP={lpr_mlp:.1f}%")
                del fm, lm

            _cleanup(device, target_mdl, bu_mdl, ft_mdl, retrained, orig)

        except Exception as e:
            print(f"    [ERROR] {ds_name} seed={seed}: {e}")
            import traceback; traceback.print_exc()

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Only run these datasets (e.g. COVID19)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", default="./data_raw")
    args = parser.parse_args()

    device = torch.device(args.device)
    data_root = args.data_root

    if args.datasets:
        ds_list = [t for t in ALL_DATASETS if t[0] in args.datasets]
    else:
        ds_list = ALL_DATASETS

    print("=" * 60)
    print("  NONLINEAR PROBE RECOVERY (Appendix D)")
    print(f"  Datasets: {[d[0] for d in ds_list]}")
    print(f"  Seeds:  {SEEDS}")
    print(f"  Device: {device}")
    print(f"  Start:  {datetime.now()}")
    print("=" * 60)

    os.makedirs('results', exist_ok=True)
    all_rows = []

    for ds_name, arch in ds_list:
        ds_rows = run_one_dataset(ds_name, arch, device, data_root)
        all_rows.extend(ds_rows)
        if ds_rows:
            df = pd.DataFrame(ds_rows)
            df.to_csv(f'results/nonlinear_probe_{ds_name}.csv', index=False)
            print(f"  Saved results/nonlinear_probe_{ds_name}.csv")

    if all_rows:
        df_all = pd.DataFrame(all_rows)
        df_all.to_csv('results/nonlinear_probe_all.csv', index=False)
        print(f"\nSaved results/nonlinear_probe_all.csv ({len(df_all)} rows)")

        # Summary
        print("\n" + "=" * 60)
        print("  Nonlinear Probe Summary (mean over seeds)")
        print("=" * 60)
        for ds_name, _ in ALL_DATASETS:
            df_ds = df_all[df_all['dataset'] == ds_name]
            print(f"\n  {ds_name}:")
            for m in ['Retrain', 'FT', 'BU', 'Target']:
                sub = df_ds[df_ds['method'] == m]
                if len(sub) == 0: continue
                print(f"    {m:8s}: Linear={sub['lpr_linear'].mean():.1f}%  "
                      f"MLP={sub['lpr_mlp'].mean():.1f}%")

    print(f"\nDone: {datetime.now()}")


if __name__ == "__main__":
    main()
