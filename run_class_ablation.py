#!/usr/bin/env python3
"""
Class Ablation — Full Methods (supplement to existing class ablation)
The existing class ablation only includes Retrain + Target.
This adds BU and FT to show that the "forgetting illusion" is
consistent across forget-classes and methods.

Datasets: All 7 (same as main experiments)
Classes:  Up to 4 evenly-spaced classes per dataset
Methods:  Retrain, FT, BU, Target
Seeds:    42, 123, 456

Output: results/class_ablation_full_{dataset}.csv
        results/class_ablation_full_all.csv

Usage:

    python run_class_ablation.py
    python run_class_ablation.py --datasets CIFAR10 CIFAR100
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
    create_vfl_model, train_vfl, evaluate_accuracy,
    extract_features, linear_probe_recovery,
    compute_cka_similarity, feature_separability,
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

def _get_ablation_classes(num_classes, max_n=4):
    """Select up to max_n evenly-spaced classes for forget-class ablation."""
    if num_classes <= max_n:
        return list(range(num_classes))
    return np.linspace(0, num_classes - 1, max_n, dtype=int).tolist()

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

def run_one_dataset(ds_name, default_arch, device, data_root, skip_classes=None,
                    skip_seeds=None):
    print(f"  {ds_name} — Class Ablation (Full Methods)")

    # Determine classes to ablate
    _train, _test, num_classes, in_ch, img_size = get_dataset(ds_name, data_root)
    del _train, _test

    forget_classes = _get_ablation_classes(num_classes)
    if skip_classes:
        forget_classes = [c for c in forget_classes if c not in skip_classes]
    is_tabular = (in_ch == 0)
    arch = 'mlp' if is_tabular else default_arch

    print(f"  {num_classes} classes, ablate classes {forget_classes}")

    ds_rows = []

    for fc in forget_classes:
        unlearn_labels = [fc]
        print(f"\n--- {ds_name}: Forget class {fc} ---")

        seeds = [s for s in SEEDS if s not in (skip_seeds or [])]
        for i, seed in enumerate(seeds):
            set_seed(seed)
            print(f"  Seed {seed} ({i+1}/{len(seeds)})")

            try:
                # Setup data
                train_ds, test_ds, num_classes, in_ch, img_size = get_dataset(
                    ds_name, data_root)
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
                vfl_DrT   = VFLDataset(Dr_test, np_)
                vfl_DuT   = VFLDataset(Du_test, np_)

                train_loader     = DataLoader(vfl_train, bs, shuffle=True,  num_workers=2, pin_memory=True)
                Dr_loader        = DataLoader(vfl_Dr,    bs, shuffle=True,  num_workers=2, pin_memory=True)
                Du_loader        = DataLoader(vfl_Du,    bs, shuffle=True,  num_workers=2, pin_memory=True)
                Dp_u_loader      = DataLoader(vfl_Dpu, min(bs, len(Dp_u)), shuffle=True,  num_workers=2, pin_memory=True)
                Dp_r_loader      = DataLoader(vfl_Dpr, min(bs, len(Dp_r)), shuffle=True,  num_workers=2, pin_memory=True)
                Dr_test_loader   = DataLoader(vfl_DrT,   bs, shuffle=False, num_workers=2, pin_memory=True)
                Du_test_loader   = DataLoader(vfl_DuT,   bs, shuffle=False, num_workers=2, pin_memory=True)
                full_test_loader = DataLoader(vfl_test,  bs, shuffle=False, num_workers=2, pin_memory=True)

                ul_lr = 1e-4 if is_tabular else 0.001

                # Train original
                print("    Training original...")
                orig = create_vfl_model(arch, in_ch, input_width, np_, num_classes)
                orig = train_vfl(orig, train_loader, CONFIG['train_epochs'],
                                 train_lr, device, verbose=True,
                                 use_adam=is_tabular)

                # Retrain
                print("    Retraining...")
                retrained = create_vfl_model(arch, in_ch, input_width, np_, num_classes)
                retrained = train_vfl(retrained, Dr_loader, CONFIG['train_epochs'],
                                      train_lr, device, verbose=True,
                                      use_adam=is_tabular)

                # Reference features
                feat_orig, _ = extract_features(orig, full_test_loader, device)
                feat_retr, _ = extract_features(retrained, full_test_loader, device)

                # Unlearning methods
                print("    Running FT...")
                ft_mdl = fine_tuning_unlearn(
                    orig, Dr_loader, 5, ul_lr, device, is_tabular)

                print("    Running BU...")
                bu_mdl = boundary_unlearn(
                    orig, Du_loader, Dr_loader, 5, ul_lr, device, is_tabular)

                print("    Running Target...")
                target_mdl = manifold_mixup_vfl_unlearn(
                    orig, Dp_u_loader, Dp_r_loader,
                    CONFIG['unlearn_epochs'],
                    ul_lr * 10 if is_tabular else CONFIG['unlearn_lr'],
                    CONFIG['mixup_alpha'], device, is_tabular)

                # Evaluate all methods
                for name, mdl in [('Retrain', retrained),
                                  ('FT', ft_mdl),
                                  ('BU', bu_mdl),
                                  ('Target', target_mdl)]:
                    dr = evaluate_accuracy(mdl, Dr_test_loader, device)
                    yu = evaluate_accuracy(mdl, Du_test_loader, device)
                    fm, lm = extract_features(mdl, full_test_loader, device)
                    lpr_acc, lpr_auc = linear_probe_recovery(
                        mdl, full_test_loader, unlearn_labels, device,
                        cached_features=(fm, lm))
                    cka_o, cka_r = compute_cka_similarity(
                        fm, feat_orig, feat_retr, CONFIG['cka_num_samples'])
                    sep = feature_separability(fm, lm, unlearn_labels)

                    row = dict(dataset=ds_name, forget_class=fc, seed=seed,
                               method=name, dr_acc=dr, yu_acc=yu,
                               lpr_acc=lpr_acc, lpr_auroc=lpr_auc,
                               cka_original=cka_o, cka_retrain=cka_r,
                               separability=sep)
                    ds_rows.append(row)
                    print(f"    {name}: Dr={dr:.1f}% yu={yu:.1f}% "
                          f"LPR={lpr_acc:.1f}% Sep={sep:.3f}")
                    del fm, lm

                _cleanup(device, target_mdl, bu_mdl, ft_mdl, retrained, orig)
                del feat_orig, feat_retr

            except Exception as e:
                print(f"  [ERROR] {ds_name} class={fc} seed={seed}: {e}")
                import traceback; traceback.print_exc()

    return ds_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--skip-classes", nargs="+", type=int, default=None,
                        help="Skip these class indices (e.g. 0 33 66)")
    parser.add_argument("--skip-seeds", nargs="+", type=int, default=None,
                        help="Skip these seeds (e.g. 42)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", default="./data_raw")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.datasets:
        ds_list = [(d, 'mlp' if d == 'YahooAnswers' else 'resnet18')
                   for d in args.datasets]
    else:
        ds_list = ALL_DATASETS

    print("=" * 60)
    print("  CLASS ABLATION — FULL METHODS (Retrain/FT/BU/Target)")
    print(f"  Datasets: {[d[0] for d in ds_list]}")
    if args.skip_classes:
        print(f"  Skip classes: {args.skip_classes}")
    if args.skip_seeds:
        print(f"  Skip seeds: {args.skip_seeds}")
    print(f"  Seeds:    {[s for s in SEEDS if s not in (args.skip_seeds or [])]}")
    print(f"  Device:   {device}")
    print(f"  Start:    {datetime.now()}")
    print("=" * 60)

    os.makedirs('results', exist_ok=True)
    all_rows = []

    for ds_name, default_arch in ds_list:
        ds_rows = run_one_dataset(ds_name, default_arch, device, args.data_root,
                                  skip_classes=args.skip_classes,
                                  skip_seeds=args.skip_seeds)
        all_rows.extend(ds_rows)

        if ds_rows:
            df = pd.DataFrame(ds_rows)
            path = f'results/class_ablation_full_{ds_name}.csv'
            df.to_csv(path, index=False)
            print(f"\n  Saved {path}")

            # Print summary
            print(f"\n  {ds_name}: Mean over seeds")
            summary = df.groupby(['forget_class', 'method']).agg(
                dr_acc=('dr_acc', 'mean'),
                lpr_acc=('lpr_acc', 'mean'),
                sep=('separability', 'mean')
            ).reset_index()
            print(summary.to_string(index=False))

    # Save combined
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        path = 'results/class_ablation_full_all.csv'
        df_all.to_csv(path, index=False)
        print(f"\nSaved {path} ({len(df_all)} rows)")

    print(f"\nDone: {datetime.now()}")

if __name__ == "__main__":
    main()
