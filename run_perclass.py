#!/usr/bin/env python3
"""
Per-class LPR Gap (Appendix C)
For each class as the forget target, compute LPR gap (method_LPR - retrain_LPR).
Shows the forgetting illusion is not specific to any one class.

Datasets: CIFAR10, CIFAR100, COVID19
Methods:  Retrain, FT, BU, Target
Seeds:    42, 123, 456

Output: results/perclass_{dataset}.csv / results/perclass_all.csv

Usage:

    python run_perclass.py
"""

import argparse, os, sys, gc
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
    fine_tuning_unlearn, boundary_unlearn,
    manifold_mixup_vfl_unlearn, _free_model,
)
from torch.utils.data import DataLoader

SEEDS = [42, 123, 456]
DATASETS = [
    ('CIFAR10',  'resnet18', 10),
    ('CIFAR100', 'resnet18', 100),
    ('COVID19',  'resnet18', 4),
]

def _cleanup(device, *models):
    for m in models:
        if m is not None:
            try: _free_model(m, device)
            except: pass
    gc.collect()
    if 'cuda' in str(device): torch.cuda.empty_cache()

def run_one_dataset(ds_name, default_arch, total_classes, device, data_root,
                    skip_classes=None, skip_seeds=None):
    print(f"  {ds_name} — Per-class LPR Gap ({total_classes} classes)")

    # For CIFAR100, sample every 10th class to keep runtime reasonable
    if total_classes > 20:
        class_list = list(range(0, total_classes, 10))  # 0,10,20,...,90
    else:
        class_list = list(range(total_classes))

    if skip_classes:
        class_list = [c for c in class_list if c not in skip_classes]

    print(f"  Testing classes: {class_list}")
    rows = []

    for fc in class_list:
        unlearn_labels = [fc]
        print(f"\n  --- Forget class {fc} ---")

        seeds = [s for s in SEEDS if s not in (skip_seeds or [])]
        for i, seed in enumerate(seeds):
            set_seed(seed)

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
                orig = create_vfl_model(arch, in_ch, input_width, np_, num_classes)
                orig = train_vfl(orig, train_loader, CONFIG['train_epochs'],
                                 train_lr, device, verbose=True, use_adam=is_tabular)

                # Retrain
                retrained = create_vfl_model(arch, in_ch, input_width, np_, num_classes)
                retrained = train_vfl(retrained, Dr_loader, CONFIG['train_epochs'],
                                      train_lr, device, verbose=True, use_adam=is_tabular)

                # Unlearning
                ft_mdl = fine_tuning_unlearn(orig, Dr_loader, 5, ul_lr, device, is_tabular)
                bu_mdl = boundary_unlearn(orig, Du_loader, Dr_loader, 5, ul_lr, device, is_tabular)
                target_mdl = manifold_mixup_vfl_unlearn(
                    orig, Dp_u_loader, Dp_r_loader,
                    CONFIG['unlearn_epochs'],
                    ul_lr * 10 if is_tabular else CONFIG['unlearn_lr'],
                    CONFIG['mixup_alpha'], device, is_tabular)

                # Evaluate
                for name, mdl in [('Retrain', retrained), ('FT', ft_mdl),
                                  ('BU', bu_mdl), ('Target', target_mdl)]:
                    fm, lm = extract_features(mdl, full_test_loader, device)
                    lpr_acc, _ = linear_probe_recovery(
                        mdl, full_test_loader, unlearn_labels, device,
                        cached_features=(fm, lm))
                    rows.append(dict(dataset=ds_name, forget_class=fc,
                                     seed=seed, method=name, lpr_acc=lpr_acc))
                    del fm, lm

                ret_lpr = [r['lpr_acc'] for r in rows
                           if r['forget_class'] == fc and r['seed'] == seed
                           and r['method'] == 'Retrain'][0]
                print(f"    seed={seed}: " +
                      " ".join(f"{r['method']}={r['lpr_acc']:.1f}(Δ{r['lpr_acc']-ret_lpr:+.1f})"
                               for r in rows
                               if r['forget_class'] == fc and r['seed'] == seed))

                _cleanup(device, target_mdl, bu_mdl, ft_mdl, retrained, orig)

            except Exception as e:
                print(f"    [ERROR] class={fc} seed={seed}: {e}")
                import traceback; traceback.print_exc()

    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Only run these datasets (e.g. CIFAR10 COVID19)")
    parser.add_argument("--skip-classes", nargs="+", type=int, default=None,
                        help="Skip these class indices (e.g. 0 1 2)")
    parser.add_argument("--skip-seeds", nargs="+", type=int, default=None,
                        help="Skip these seeds (e.g. 42)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", default="./data_raw")
    args = parser.parse_args()

    device = torch.device(args.device)
    data_root = args.data_root

    if args.datasets:
        ds_list = [t for t in DATASETS if t[0] in args.datasets]
    else:
        ds_list = DATASETS

    print("=" * 60)
    print("  PER-CLASS LPR GAP (Appendix C)")
    print(f"  Datasets: {[d[0] for d in ds_list]}")
    if args.skip_classes:
        print(f"  Skip classes: {args.skip_classes}")
    if args.skip_seeds:
        print(f"  Skip seeds: {args.skip_seeds}")
    print(f"  Seeds:  {[s for s in SEEDS if s not in (args.skip_seeds or [])]}")
    print(f"  Device: {device}")
    print(f"  Start:  {datetime.now()}")
    print("=" * 60)

    os.makedirs('results', exist_ok=True)
    all_rows = []

    for ds_name, arch, n_cls in ds_list:
        ds_rows = run_one_dataset(ds_name, arch, n_cls, device, data_root,
                                  skip_classes=args.skip_classes,
                                  skip_seeds=args.skip_seeds)
        all_rows.extend(ds_rows)
        if ds_rows:
            df = pd.DataFrame(ds_rows)
            df.to_csv(f'results/perclass_{ds_name}.csv', index=False)
            print(f"  Saved results/perclass_{ds_name}.csv")

    if all_rows:
        df_all = pd.DataFrame(all_rows)
        df_all.to_csv('results/perclass_all.csv', index=False)
        print(f"\nSaved results/perclass_all.csv ({len(df_all)} rows)")

    print(f"\nDone: {datetime.now()}")

if __name__ == "__main__":
    main()
