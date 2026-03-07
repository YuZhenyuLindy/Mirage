#!/usr/bin/env python3
"""
Timing / Computational Cost (Appendix E)
==========================================
Measure wall-clock time for each unlearning method + Mirage audit overhead.

Datasets: All 7
Methods:  Retrain, FT, Fisher, Amnesiac, UNSIR, BU, SSD, Target + Mirage audit
Seeds:    42, 123, 456

Output: results/timing_{dataset}.csv / results/timing_all.csv

Usage:

    python run_timing.py
"""

import argparse, os, sys, gc, time
import numpy as np
import pandas as pd
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_datasets  # noqa: patches get_dataset for cached ModelNet10

from mirage_lib import (
    set_seed, CONFIG,
    get_dataset, VFLDataset, prepare_unlearning_data,
    create_vfl_model, train_vfl, evaluate_accuracy, measure_runtime,
    extract_features, linear_probe_recovery,
    compute_cka_similarity, feature_separability,
    fine_tuning_unlearn, fisher_forgetting, amnesiac_unlearn,
    unsir_unlearn, boundary_unlearn, ssd_unlearn,
    manifold_mixup_vfl_unlearn, retrain_from_scratch, _free_model,
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


def run_one_dataset(ds_name, default_arch, device, data_root, skip_seeds=None):
    print(f"\n{'='*60}")
    print(f"  {ds_name} — Timing")
    print(f"{'='*60}")

    unlearn_labels = [0]
    rows = []
    seeds = [s for s in SEEDS if s not in (skip_seeds or [])]

    for i, seed in enumerate(seeds):
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

            # Train original (shared)
            print("    Training original...")
            orig = create_vfl_model(arch, in_ch, input_width, np_, num_classes)
            orig = train_vfl(orig, train_loader, CONFIG['train_epochs'],
                             train_lr, device, verbose=True, use_adam=is_tabular)

            # All methods with timing
            specs = [
                ('Retrain', retrain_from_scratch,
                 (arch, in_ch, input_width, np_, num_classes,
                  Dr_loader, CONFIG['train_epochs'], train_lr, device, is_tabular)),
                ('FT', fine_tuning_unlearn,
                 (orig, Dr_loader, 5, ul_lr, device, is_tabular)),
                ('Fisher', fisher_forgetting,
                 (orig, Du_loader, Dr_loader, device)),
                ('Amnesiac', amnesiac_unlearn,
                 (orig, Du_loader, num_classes, 5, ul_lr, device, is_tabular)),
                ('UNSIR', unsir_unlearn,
                 (orig, Du_loader, Dr_loader, 5, ul_lr, 0.1, device, is_tabular)),
                ('BU', boundary_unlearn,
                 (orig, Du_loader, Dr_loader, 5, ul_lr, device, is_tabular)),
                ('SSD', ssd_unlearn,
                 (orig, Du_loader, Dr_loader, device)),
                ('Target', manifold_mixup_vfl_unlearn,
                 (orig, Dp_u_loader, Dp_r_loader,
                  CONFIG['unlearn_epochs'],
                  ul_lr * 10 if is_tabular else CONFIG['unlearn_lr'],
                  CONFIG['mixup_alpha'], device, is_tabular)),
            ]

            for name, fn, args in specs:
                print(f"    Timing {name}...", end=" ", flush=True)
                mdl, rt = measure_runtime(fn, *args)
                print(f"{rt:.2f}s")

                # Also time the Mirage audit (extract_features + LPR)
                t0 = time.time()
                fm, lm = extract_features(mdl, full_test_loader, device)
                lpr_acc, _ = linear_probe_recovery(
                    mdl, full_test_loader, unlearn_labels, device,
                    cached_features=(fm, lm))
                audit_time = time.time() - t0

                rows.append(dict(
                    dataset=ds_name, seed=seed, method=name,
                    unlearn_time=rt, audit_time=audit_time,
                    total_time=rt + audit_time, lpr_acc=lpr_acc))
                del fm, lm
                if name != 'Retrain':
                    _free_model(mdl, device)
                else:
                    _free_model(mdl, device)

            _cleanup(device, orig)

        except Exception as e:
            print(f"    [ERROR] {ds_name} seed={seed}: {e}")
            import traceback; traceback.print_exc()

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--skip-seeds", nargs="+", type=int, default=None,
                        help="Skip these seeds (e.g. 42)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", default="./data_raw")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.datasets:
        ds_list = [t for t in ALL_DATASETS if t[0] in args.datasets]
        if not ds_list:
            ds_list = [(d, 'mlp' if d == 'YahooAnswers' else 'resnet18')
                       for d in args.datasets]
    else:
        ds_list = ALL_DATASETS

    print("=" * 60)
    print("  TIMING / COMPUTATIONAL COST (Appendix E)")
    print(f"  Datasets: {[d[0] for d in ds_list]}")
    if args.skip_seeds:
        print(f"  Skip seeds: {args.skip_seeds}")
    print(f"  Seeds:  {[s for s in SEEDS if s not in (args.skip_seeds or [])]}")
    print(f"  Device: {device}")
    print(f"  Start:  {datetime.now()}")
    print("=" * 60)

    os.makedirs('results', exist_ok=True)
    all_rows = []

    for ds_name, arch in ds_list:
        ds_rows = run_one_dataset(ds_name, arch, device, args.data_root,
                                  skip_seeds=args.skip_seeds)
        all_rows.extend(ds_rows)
        if ds_rows:
            df = pd.DataFrame(ds_rows)
            df.to_csv(f'results/timing_{ds_name}.csv', index=False)
            print(f"  Saved results/timing_{ds_name}.csv")

    if all_rows:
        df_all = pd.DataFrame(all_rows)
        df_all.to_csv('results/timing_all.csv', index=False)
        print(f"\nSaved results/timing_all.csv ({len(df_all)} rows)")

        # Summary
        print("\n" + "=" * 60)
        print("  Timing Summary (mean over seeds, in seconds)")
        print("=" * 60)
        for ds_name, _ in ALL_DATASETS:
            df_ds = df_all[df_all['dataset'] == ds_name]
            print(f"\n  {ds_name}:")
            print(f"    {'Method':10s} {'Unlearn':>8s} {'Audit':>8s} {'Total':>8s}")
            for m in ['Retrain', 'FT', 'Fisher', 'Amnesiac', 'UNSIR',
                       'BU', 'SSD', 'Target']:
                sub = df_ds[df_ds['method'] == m]
                if len(sub) == 0: continue
                print(f"    {m:10s} {sub['unlearn_time'].mean():8.2f} "
                      f"{sub['audit_time'].mean():8.2f} "
                      f"{sub['total_time'].mean():8.2f}")

    print(f"\nDone: {datetime.now()}")


if __name__ == "__main__":
    main()
