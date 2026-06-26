#!/usr/bin/env python3
"""
Main Experiment — Per-Seed Results (for ±std in Tables 1 & 2)
Re-run the main experiment for 7 datasets × 8 methods × 3 seeds,
saving per-seed results instead of averaged values.

Output: results/main_perseed_{dataset}.csv
        results/main_perseed_all.csv

Usage:

    python run_main.py
    python run_main.py --datasets CIFAR10 COVID19   # subset
"""

import argparse, os, sys, time, gc
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
            try:
                _free_model(m, device)
            except Exception:
                pass
    gc.collect()
    if 'cuda' in str(device):
        torch.cuda.empty_cache()

def run_one_seed(ds_name, default_arch, seed, device, data_root):
    """Run full pipeline for one dataset + one seed. Returns list of row dicts."""
    print(f"\n--- {ds_name} Seed {seed} ---")
    set_seed(seed)

    train_ds, test_ds, num_classes, in_ch, img_size = get_dataset(ds_name, data_root)
    is_tabular = (in_ch == 0)
    arch = 'mlp' if is_tabular else default_arch
    train_lr = 1e-3 if is_tabular else CONFIG['lr']
    np_ = CONFIG['num_passive_parties']
    input_width = img_size // np_
    bs = CONFIG['batch_size']

    unlearn_labels = [0]

    # Prepare data
    vfl_train = VFLDataset(train_ds, np_)
    vfl_test = VFLDataset(test_ds, np_)
    Dr_train, Du_train, Dp_u, Dp_r, Dr_test, Du_test = \
        prepare_unlearning_data(train_ds, test_ds, unlearn_labels,
                                CONFIG['num_public_samples'])
    vfl_Dr = VFLDataset(Dr_train, np_)
    vfl_Du = VFLDataset(Du_train, np_)
    vfl_Dpu = VFLDataset(Dp_u, np_)
    vfl_Dpr = VFLDataset(Dp_r, np_)
    vfl_DrT = VFLDataset(Dr_test, np_)
    vfl_DuT = VFLDataset(Du_test, np_)

    train_loader = DataLoader(vfl_train, bs, shuffle=True, num_workers=2, pin_memory=True)
    Dr_loader = DataLoader(vfl_Dr, bs, shuffle=True, num_workers=2, pin_memory=True)
    Du_loader = DataLoader(vfl_Du, bs, shuffle=True, num_workers=2, pin_memory=True)
    Dp_u_loader = DataLoader(vfl_Dpu, min(bs, len(Dp_u)), shuffle=True, num_workers=2, pin_memory=True)
    Dp_r_loader = DataLoader(vfl_Dpr, min(bs, len(Dp_r)), shuffle=True, num_workers=2, pin_memory=True)
    Dr_test_loader = DataLoader(vfl_DrT, bs, shuffle=False, num_workers=2, pin_memory=True)
    Du_test_loader = DataLoader(vfl_DuT, bs, shuffle=False, num_workers=2, pin_memory=True)
    full_test_loader = DataLoader(vfl_test, bs, shuffle=False, num_workers=2, pin_memory=True)

    # [1/4] Train original
    print("  [1/4] Training original VFL model...")
    orig = create_vfl_model(arch, in_ch, input_width, np_, num_classes)
    orig = train_vfl(orig, train_loader, CONFIG['train_epochs'], train_lr, device,
                     use_adam=is_tabular)

    # [2/4] Retrain
    print("  [2/4] Retraining from scratch...")
    retrained, rt_time = measure_runtime(
        retrain_from_scratch, arch, in_ch, input_width, np_, num_classes,
        Dr_loader, CONFIG['train_epochs'], train_lr, device, is_tabular)

    # [3/4] Extract reference features
    print("  [3/4] Extracting reference features...")
    feat_orig, lab_orig = extract_features(orig, full_test_loader, device)
    feat_retr, _ = extract_features(retrained, full_test_loader, device)

    # [4/4] Unlearning methods
    print("  [4/4] Running unlearning methods...")
    ul_lr = 1e-4 if is_tabular else 0.001
    specs = [
        ('Retrain', None, None, rt_time),
        ('FT', fine_tuning_unlearn,
         (orig, Dr_loader, 5, ul_lr, device, is_tabular), None),
        ('Fisher', fisher_forgetting,
         (orig, Du_loader, Dr_loader, device), None),
        ('Amnesiac', amnesiac_unlearn,
         (orig, Du_loader, num_classes, 5, ul_lr, device, is_tabular), None),
        ('UNSIR', unsir_unlearn,
         (orig, Du_loader, Dr_loader, 5, ul_lr, 0.1, device, is_tabular), None),
        ('BU', boundary_unlearn,
         (orig, Du_loader, Dr_loader, 5, ul_lr, device, is_tabular), None),
        ('SSD', ssd_unlearn,
         (orig, Du_loader, Dr_loader, device), None),
        ('Target', manifold_mixup_vfl_unlearn,
         (orig, Dp_u_loader, Dp_r_loader,
          CONFIG['unlearn_epochs'],
          ul_lr * 10 if is_tabular else CONFIG['unlearn_lr'],
          CONFIG['mixup_alpha'], device, is_tabular), None),
    ]

    rows = []
    for name, fn, args, pt in specs:
        print(f"  Running {name}...")
        try:
            if name == 'Retrain':
                mdl, rt = retrained, pt
            else:
                mdl, rt = measure_runtime(fn, *args)

            dr = evaluate_accuracy(mdl, Dr_test_loader, device)
            yu = evaluate_accuracy(mdl, Du_test_loader, device)
            fm, lm = extract_features(mdl, full_test_loader, device)
            lpr_acc, lpr_auc = linear_probe_recovery(
                mdl, full_test_loader, unlearn_labels, device,
                cached_features=(fm, lm))
            cka_o, cka_r = compute_cka_similarity(
                fm, feat_orig, feat_retr, CONFIG['cka_num_samples'])
            sep = feature_separability(fm, lm, unlearn_labels)

            rows.append(dict(
                dataset=ds_name, architecture=arch, seed=seed,
                method=name, dr_acc=dr, yu_acc=yu,
                lpr_acc=lpr_acc, lpr_auroc=lpr_auc,
                cka_original=cka_o, cka_retrain=cka_r,
                separability=sep, runtime=rt))
            print(f"    Dr={dr:.2f}% | yu={yu:.2f}% | LPR={lpr_acc:.1f}% | "
                  f"CKA_O={cka_o:.3f} | CKA_R={cka_r:.3f} | Sep={sep:.3f}")
            del fm, lm
            if name != 'Retrain':
                _free_model(mdl, device)

        except Exception as e:
            print(f"    [ERROR] {name}: {e}")
            import traceback; traceback.print_exc()
            rows.append(dict(
                dataset=ds_name, architecture=arch, seed=seed,
                method=name, dr_acc=0, yu_acc=0,
                lpr_acc=50, lpr_auroc=0.5,
                cka_original=0, cka_retrain=0,
                separability=0, runtime=0))

    _cleanup(device, retrained, orig)
    del feat_orig, feat_retr, lab_orig
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
    datasets = args.datasets
    seeds = [s for s in SEEDS if s not in (args.skip_seeds or [])]

    if datasets:
        ds_list = [(d, 'mlp' if d == 'YahooAnswers' else 'resnet18') for d in datasets]
    else:
        ds_list = ALL_DATASETS

    print("=" * 70)
    print("  MAIN EXPERIMENT — PER-SEED RESULTS (for ±std)")
    print(f"  Datasets: {[d[0] for d in ds_list]}")
    if args.skip_seeds:
        print(f"  Skip seeds: {args.skip_seeds}")
    print(f"  Seeds:    {seeds}")
    print(f"  Device:   {device}")
    print(f"  Start:    {datetime.now()}")
    print("=" * 70)

    os.makedirs('results', exist_ok=True)
    all_rows = []

    for ds_name, default_arch in ds_list:
        ds_rows = []
        for seed in seeds:
            rows = run_one_seed(ds_name, default_arch, seed, device, args.data_root)
            ds_rows.extend(rows)
            all_rows.extend(rows)

        # Save per-dataset
        df = pd.DataFrame(ds_rows)
        path = f'results/main_perseed_{ds_name}.csv'
        df.to_csv(path, index=False)
        print(f"\nSaved {path}")

        # Print summary
        print(f"\n  {ds_name}: Mean ± Std over {len(SEEDS)} seeds")
        methods = df['method'].unique()
        for m in methods:
            sub = df[df['method'] == m]
            dr_m, dr_s = sub['dr_acc'].mean(), sub['dr_acc'].std()
            lpr_m, lpr_s = sub['lpr_acc'].mean(), sub['lpr_acc'].std()
            print(f"    {m:10s}: Dr={dr_m:.1f}±{dr_s:.1f}  LPR={lpr_m:.1f}±{lpr_s:.1f}")

    # Save combined
    df_all = pd.DataFrame(all_rows)
    path = 'results/main_perseed_all.csv'
    df_all.to_csv(path, index=False)
    print(f"\nSaved {path} ({len(df_all)} rows)")
    print(f"Done: {datetime.now()}")

if __name__ == "__main__":
    main()
