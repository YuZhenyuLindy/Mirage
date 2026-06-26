#!/usr/bin/env python3
"""
Mirage Extended Experiments — ALL remaining experiments for Strong Accept.
All ablations run across all 7 datasets (results for supplementary/appendix).

Experiments:
  1. yahoo       — Yahoo Answers full diagnostics (missing from all CSVs)
  2. sample      — Sample-level unlearning (7 datasets × 5%/10% × 3 seeds)
  3. kparty      — K-party ablation (7 datasets × K values × 3 seeds)
  4. tsne        — t-SNE feature visualization (7 datasets × 4 methods)
  5. classabl    — Forget-class ablation (7 datasets × N classes × 3 seeds)
  6. epochabl    — Unlearning epochs ablation (7 datasets × 5 epochs × 3 seeds)
  7. all         — Run everything

Usage:
    python run_extended.py --exp all        --device cuda
    python run_extended.py --exp sample     --device cuda
    python run_extended.py --exp kparty     --device cuda

Estimated time (single A100):
    yahoo:    ~30 min
    sample:   ~6 hours   (7 datasets × 2 ratios × 3 seeds)
    kparty:   ~12 hours  (7 datasets × 2–3 K values × 3 seeds)
    tsne:     ~2 hours   (7 datasets × 4 methods)
    classabl: ~10 hours  (7 datasets × 2–4 classes × 3 seeds)
    epochabl: ~7 hours   (7 datasets × 5 epoch values × 3 seeds)
    all:      ~38 hours total
"""

import argparse
import os
import sys
import time
import copy
import numpy as np
import pandas as pd
import torch
import gc
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TeeStream:
    """Write to both console and log file simultaneously."""
    def __init__(self, log_file, stream):
        self.log_file = log_file
        self.stream = stream

    def write(self, msg):
        self.stream.write(msg)
        self.stream.flush()
        self.log_file.write(msg)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def fileno(self):
        return self.stream.fileno()

from torch.utils.data import DataLoader

from mirage_lib import (
    set_seed, CONFIG,
    get_dataset, VFLDataset, prepare_unlearning_data,
    create_vfl_model, train_vfl, evaluate_accuracy, measure_runtime,
    extract_features, linear_probe_recovery,
    compute_cka_similarity, feature_separability,
    fine_tuning_unlearn, fisher_forgetting, amnesiac_unlearn,
    unsir_unlearn, boundary_unlearn, ssd_unlearn,
    manifold_mixup_vfl_unlearn, retrain_from_scratch,
    run_sample_experiment, run_kparty_experiment, run_single_experiment,
    visualize_tsne, _free_model,
)

# Constants

ALL_DATASETS = [
    ('MNIST',        'resnet18'),
    ('CIFAR10',      'resnet18'),
    ('CIFAR100',     'resnet18'),
    ('ModelNet10',   'resnet18'),
    ('BrainTumor',   'resnet18'),
    ('COVID19',      'resnet18'),
    ('YahooAnswers', 'mlp'),
]

# K values per dataset.
# MNIST is 28px wide: K=8 → 3px per party, too narrow for two MaxPool2d(2,2).
# All 32px datasets and tabular data support K=8.
K_VALUES = {
    'MNIST':        (2, 4),
    'CIFAR10':      (2, 4, 8),
    'CIFAR100':     (2, 4, 8),
    'ModelNet10':   (2, 4, 8),
    'BrainTumor':   (2, 4, 8),
    'COVID19':      (2, 4, 8),
    'YahooAnswers': (2, 4, 8),
}

SEEDS = [42, 123, 456]

# Helpers

def banner(title):
    print(f"  {title}")

def _get_ablation_classes(num_classes, max_n=4):
    """Select up to max_n evenly-spaced classes for forget-class ablation."""
    if num_classes <= max_n:
        return list(range(num_classes))
    return np.linspace(0, num_classes - 1, max_n, dtype=int).tolist()

def _setup_experiment(dataset_name, unlearn_labels, device, data_root,
                      num_parties=2):
    """Common setup: load data, create VFL loaders, return config dict.

    Returns dict with keys:
        train_ds, test_ds, num_classes, in_ch, img_size,
        is_tabular, arch, train_lr, input_width, np_, loaders
    where loaders = {train, Dr, Du, Dp_u, Dp_r, Dr_test, Du_test, full_test}
    """
    train_ds, test_ds, num_classes, in_ch, img_size = get_dataset(
        dataset_name, data_root)
    is_tabular = (in_ch == 0)
    arch = 'mlp' if is_tabular else 'resnet18'
    train_lr = 1e-3 if is_tabular else CONFIG['lr']
    np_ = num_parties
    input_width = img_size // np_

    vfl_train = VFLDataset(train_ds, np_)
    vfl_test  = VFLDataset(test_ds, np_)
    Dr_train, Du_train, Dp_u, Dp_r, Dr_test, Du_test = \
        prepare_unlearning_data(train_ds, test_ds, unlearn_labels,
                                CONFIG['num_public_samples'])
    vfl_Dr  = VFLDataset(Dr_train, np_)
    vfl_Du  = VFLDataset(Du_train, np_)
    vfl_Dpu = VFLDataset(Dp_u, np_)
    vfl_Dpr = VFLDataset(Dp_r, np_)
    vfl_DrT = VFLDataset(Dr_test, np_)
    vfl_DuT = VFLDataset(Du_test, np_)

    bs = CONFIG['batch_size']

    return dict(
        train_ds=train_ds, test_ds=test_ds,
        num_classes=num_classes, in_ch=in_ch, img_size=img_size,
        is_tabular=is_tabular, arch=arch, train_lr=train_lr,
        input_width=input_width, np_=np_,
        loaders=dict(
            train     = DataLoader(vfl_train, bs, shuffle=True,  num_workers=0),
            Dr        = DataLoader(vfl_Dr,    bs, shuffle=True,  num_workers=0),
            Du        = DataLoader(vfl_Du,    bs, shuffle=True,  num_workers=0),
            Dp_u      = DataLoader(vfl_Dpu, min(bs, len(Dp_u)), shuffle=True,
                                   num_workers=0),
            Dp_r      = DataLoader(vfl_Dpr, min(bs, len(Dp_r)), shuffle=True,
                                   num_workers=0),
            Dr_test   = DataLoader(vfl_DrT,   bs, shuffle=False, num_workers=0),
            Du_test   = DataLoader(vfl_DuT,   bs, shuffle=False, num_workers=0),
            full_test = DataLoader(vfl_test,  bs, shuffle=False, num_workers=0),
        )
    )

def _cleanup(device, *models):
    """Free GPU memory for multiple models."""
    for m in models:
        if m is not None:
            try:
                _free_model(m, device)
            except Exception:
                pass
    gc.collect()
    if 'cuda' in str(device):
        torch.cuda.empty_cache()

# Yahoo Answers — CRITICAL missing data

def run_yahoo_experiments(device, data_root='./data_raw'):
    """Run full Yahoo Answers experiments: output metrics + Mirage diagnostics.

    This fills the CRITICAL gap: paper claims 7 datasets but Yahoo Answers
    has NO data in any CSV file.
    """
    banner("YAHOO ANSWERS — FULL EXPERIMENT (3 seeds)")

    for i, seed in enumerate(SEEDS):
        print(f"\n--- Yahoo Answers Seed {seed} ({i+1}/{len(SEEDS)}) ---")
        set_seed(seed)
        run_single_experiment(
            'YahooAnswers', 'mlp', [0],
            device=device, data_root=data_root
        )

# Sample-Level Unlearning — ALL 7 datasets

def run_sample_level_experiments(device, data_root='./data_raw'):
    """Sample-level unlearning: forget random 5%/10% of training samples.

    Runs across all 7 datasets × {5%, 10%} × 3 seeds.
    """
    banner("SAMPLE-LEVEL UNLEARNING — ALL 7 DATASETS × 5%/10% × 3 seeds")

    all_rows = []

    for ds_name, arch in ALL_DATASETS:
        for ratio in [0.05, 0.10]:
            for i, seed in enumerate(SEEDS):
                print(f"\n--- {ds_name} forget={ratio*100:.0f}% seed={seed} "
                      f"({i+1}/{len(SEEDS)}) ---")
                set_seed(seed)
                try:
                    results = run_sample_experiment(
                        ds_name, arch, forget_ratio=ratio,
                        device=device, data_root=data_root, seed=seed)
                    for method, metrics in results.items():
                        all_rows.append(dict(
                            dataset=ds_name, architecture=arch,
                            forget_ratio=ratio, seed=seed,
                            method=method, **metrics))
                except Exception as e:
                    print(f"  [ERROR] {ds_name} ratio={ratio} seed={seed}: {e}")
                    import traceback; traceback.print_exc()

    # Save combined CSV
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(all_rows)
    csv_path = 'results/sample_level_all.csv'
    df.to_csv(csv_path, index=False)
    print(f"Sample-level results saved to {csv_path} ({len(df)} rows)")

# K-Party Ablation — ALL 7 datasets

def run_kparty_experiments(device, data_root='./data_raw'):
    """K-party ablation across all 7 datasets.

    K values are dataset-specific:
      - MNIST: K={2,4}  (28px too narrow for K=8)
      - Others: K={2,4,8}
    """
    banner("K-PARTY ABLATION — ALL 7 DATASETS × K values × 3 seeds")

    all_rows = []

    for ds_name, arch in ALL_DATASETS:
        k_vals = K_VALUES[ds_name]
        print(f"\n>>> {ds_name}: K = {k_vals}")

        for i, seed in enumerate(SEEDS):
            print(f"\n--- {ds_name} seed={seed} ({i+1}/{len(SEEDS)}) ---")
            set_seed(seed)
            try:
                results = run_kparty_experiment(
                    ds_name, arch, [0], K_values=k_vals,
                    device=device, data_root=data_root)
                for K, k_res in results.items():
                    for method, metrics in k_res.items():
                        all_rows.append(dict(
                            dataset=ds_name, architecture=arch,
                            seed=seed, method=method, **metrics))
            except Exception as e:
                print(f"  [ERROR] {ds_name} seed={seed}: {e}")
                import traceback; traceback.print_exc()

    # Save combined CSV
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(all_rows)
    csv_path = 'results/kparty_all.csv'
    df.to_csv(csv_path, index=False)
    print(f"K-party results saved to {csv_path} ({len(df)} rows)")

# t-SNE Visualization — ALL 7 datasets

def run_tsne_visualization(device, data_root='./data_raw'):
    """Generate t-SNE feature visualizations for all 7 datasets.

    For each dataset, trains original + 3 unlearning methods (Retrain, Target,
    BU, FT) and generates a multi-panel t-SNE figure.
    """
    banner("t-SNE VISUALIZATION — ALL 7 DATASETS")

    for ds_name, default_arch in ALL_DATASETS:
        print(f"\n>>> t-SNE: {ds_name}")
        set_seed(42)
        unlearn_labels = [0]

        try:
            ctx = _setup_experiment(ds_name, unlearn_labels, device, data_root)
            L = ctx['loaders']
            arch = ctx['arch']
            is_tab = ctx['is_tabular']
            ul_lr = 1e-4 if is_tab else 0.001

            # Train original
            print("  Training original model...")
            orig = create_vfl_model(arch, ctx['in_ch'], ctx['input_width'],
                                    ctx['np_'], ctx['num_classes'])
            orig = train_vfl(orig, L['train'], CONFIG['train_epochs'],
                             ctx['train_lr'], device, use_adam=is_tab)

            # Retrain
            print("  Retraining from scratch...")
            retrained = create_vfl_model(arch, ctx['in_ch'], ctx['input_width'],
                                         ctx['np_'], ctx['num_classes'])
            retrained = train_vfl(retrained, L['Dr'], CONFIG['train_epochs'],
                                  ctx['train_lr'], device, verbose=False,
                                  use_adam=is_tab)

            # Unlearning methods
            print("  Running Target, BU, FT...")
            target_model = manifold_mixup_vfl_unlearn(
                orig, L['Dp_u'], L['Dp_r'],
                CONFIG['unlearn_epochs'],
                ul_lr * 10 if is_tab else CONFIG['unlearn_lr'],
                CONFIG['mixup_alpha'], device, is_tab)
            bu_model = boundary_unlearn(
                orig, L['Du'], L['Dr'], 5, ul_lr, device, is_tab)
            ft_model = fine_tuning_unlearn(
                orig, L['Dr'], 5, ul_lr, device, is_tab)

            model_dict = {
                'Retrain': retrained,
                'Target': target_model,
                'BU': bu_model,
                'FT': ft_model,
            }

            save_dir = os.path.join(
                os.path.dirname(os.path.abspath(data_root)), 'figures')
            os.makedirs(save_dir, exist_ok=True)

            print(f"  Generating t-SNE ({len(model_dict)} methods)...")
            for fmt in ['pdf', 'png']:
                save_path = os.path.join(save_dir,
                                         f'tsne_{ds_name}.{fmt}')
                visualize_tsne(model_dict, L['full_test'], unlearn_labels,
                               device=device, save_path=save_path)

            print(f"  Saved to {save_dir}/tsne_{ds_name}.pdf|png")
            _cleanup(device, orig, target_model, bu_model, ft_model, retrained)

        except Exception as e:
            print(f"  [ERROR] t-SNE {ds_name}: {e}")
            import traceback; traceback.print_exc()

# Forget-Class Ablation — ALL 7 datasets

def run_class_ablation(device, data_root='./data_raw'):
    """Vary which class is forgotten to show the gap is not class-specific.

    For each dataset, selects up to 4 evenly-spaced classes.
    Examples: CIFAR-10 → {0,3,6,9}, BrainTumor (4 cls) → {0,1,2,3}.
    """
    banner("FORGET-CLASS ABLATION — ALL 7 DATASETS")

    grand_rows = []

    for ds_name, default_arch in ALL_DATASETS:
        # Quick load to determine num_classes
        _train, _test, num_classes, in_ch, img_size = get_dataset(
            ds_name, data_root)
        del _train, _test

        forget_classes = _get_ablation_classes(num_classes)
        is_tabular = (in_ch == 0)
        arch = 'mlp' if is_tabular else 'resnet18'

        print(f"\n>>> {ds_name}: {num_classes} classes, "
              f"ablate classes {forget_classes}")

        ds_rows = []

        for fc in forget_classes:
            unlearn_labels = [fc]
            print(f"\n--- {ds_name}: Forget class {fc} ---")

            for i, seed in enumerate(SEEDS):
                set_seed(seed)
                print(f"  Seed {seed} ({i+1}/{len(SEEDS)})")

                try:
                    ctx = _setup_experiment(ds_name, unlearn_labels, device,
                                            data_root)
                    L = ctx['loaders']
                    ul_lr = 1e-4 if ctx['is_tabular'] else 0.001

                    # Train original
                    orig = create_vfl_model(
                        ctx['arch'], ctx['in_ch'], ctx['input_width'],
                        ctx['np_'], ctx['num_classes'])
                    orig = train_vfl(orig, L['train'], CONFIG['train_epochs'],
                                     ctx['train_lr'], device, verbose=False,
                                     use_adam=ctx['is_tabular'])

                    # Retrain
                    retrained = create_vfl_model(
                        ctx['arch'], ctx['in_ch'], ctx['input_width'],
                        ctx['np_'], ctx['num_classes'])
                    retrained = train_vfl(retrained, L['Dr'],
                                          CONFIG['train_epochs'],
                                          ctx['train_lr'], device,
                                          verbose=False,
                                          use_adam=ctx['is_tabular'])

                    feat_orig, _ = extract_features(orig, L['full_test'],
                                                    device)
                    feat_retr, _ = extract_features(retrained, L['full_test'],
                                                    device)

                    # Target method
                    target_mdl = manifold_mixup_vfl_unlearn(
                        orig, L['Dp_u'], L['Dp_r'],
                        CONFIG['unlearn_epochs'],
                        ul_lr * 10 if ctx['is_tabular']
                        else CONFIG['unlearn_lr'],
                        CONFIG['mixup_alpha'], device, ctx['is_tabular'])

                    for name, mdl in [('Retrain', retrained),
                                      ('Target', target_mdl)]:
                        dr = evaluate_accuracy(mdl, L['Dr_test'], device)
                        yu = evaluate_accuracy(mdl, L['Du_test'], device)
                        fm, lm = extract_features(mdl, L['full_test'], device)
                        lpr_acc, lpr_auc = linear_probe_recovery(
                            mdl, L['full_test'], unlearn_labels, device,
                            cached_features=(fm, lm))
                        cka_o, _ = compute_cka_similarity(
                            fm, feat_orig, feat_retr,
                            CONFIG['cka_num_samples'])
                        sep = feature_separability(fm, lm, unlearn_labels)
                        row = dict(dataset=ds_name, forget_class=fc,
                                   seed=seed, method=name, dr_acc=dr,
                                   yu_acc=yu, lpr_acc=lpr_acc,
                                   cka_original=cka_o, separability=sep)
                        ds_rows.append(row)
                        grand_rows.append(row)
                        print(f"    {name}: Dr={dr:.1f}% yu={yu:.1f}% "
                              f"LPR={lpr_acc:.1f}% Sep={sep:.3f}")
                        del fm, lm

                    _cleanup(device, target_mdl, retrained, orig)
                    del feat_orig, feat_retr

                except Exception as e:
                    print(f"  [ERROR] {ds_name} class={fc} seed={seed}: {e}")
                    import traceback; traceback.print_exc()

        # Save per-dataset CSV
        if ds_rows:
            os.makedirs('results', exist_ok=True)
            df = pd.DataFrame(ds_rows)
            csv_path = f'results/class_ablation_{ds_name}.csv'
            df.to_csv(csv_path, index=False)
            print(f"\n  Class ablation saved to {csv_path}")

            summary = df.groupby(['forget_class', 'method']).agg(
                dr_acc=('dr_acc', 'mean'), lpr_acc=('lpr_acc', 'mean'),
                sep=('separability', 'mean')
            ).reset_index()
            print(summary.to_string(index=False))

    # Save combined CSV
    if grand_rows:
        os.makedirs('results', exist_ok=True)
        df = pd.DataFrame(grand_rows)
        csv_path = 'results/class_ablation_all.csv'
        df.to_csv(csv_path, index=False)
        print(f"Class ablation (all datasets) saved to {csv_path} "
              f"({len(df)} rows)")

# Unlearning Epochs Ablation — ALL 7 datasets

def run_epoch_ablation(device, data_root='./data_raw'):
    """Vary unlearning epochs {1,3,5,10,20} for Target method.

    Shows that more aggressive unlearning doesn't close the forgetting gap.
    Runs across all 7 datasets × 5 epoch values × 3 seeds.
    """
    banner("UNLEARNING EPOCHS ABLATION — ALL 7 DATASETS × "
           "epochs={1,3,5,10,20}")

    epoch_values = [1, 3, 5, 10, 20]
    grand_rows = []

    for ds_name, default_arch in ALL_DATASETS:
        print(f"\n>>> {ds_name}")
        unlearn_labels = [0]
        ds_rows = []

        for i, seed in enumerate(SEEDS):
            set_seed(seed)
            print(f"\n--- {ds_name} Seed {seed} ({i+1}/{len(SEEDS)}) ---")

            try:
                ctx = _setup_experiment(ds_name, unlearn_labels, device,
                                        data_root)
                L = ctx['loaders']
                ul_lr = 1e-4 if ctx['is_tabular'] else 0.001

                # Train original (shared across epoch values)
                orig = create_vfl_model(
                    ctx['arch'], ctx['in_ch'], ctx['input_width'],
                    ctx['np_'], ctx['num_classes'])
                orig = train_vfl(orig, L['train'], CONFIG['train_epochs'],
                                 ctx['train_lr'], device, verbose=False,
                                 use_adam=ctx['is_tabular'])

                # Retrain baseline (shared)
                retrained = create_vfl_model(
                    ctx['arch'], ctx['in_ch'], ctx['input_width'],
                    ctx['np_'], ctx['num_classes'])
                retrained = train_vfl(retrained, L['Dr'],
                                      CONFIG['train_epochs'],
                                      ctx['train_lr'], device, verbose=False,
                                      use_adam=ctx['is_tabular'])

                feat_orig, _ = extract_features(orig, L['full_test'], device)
                feat_retr, _ = extract_features(retrained, L['full_test'],
                                                device)

                # Retrain baseline metrics
                fm_r, lm_r = extract_features(retrained, L['full_test'],
                                              device)
                lpr_r, _ = linear_probe_recovery(
                    retrained, L['full_test'], unlearn_labels, device,
                    cached_features=(fm_r, lm_r))
                dr_r = evaluate_accuracy(retrained, L['Dr_test'], device)
                yu_r = evaluate_accuracy(retrained, L['Du_test'], device)
                sep_r = feature_separability(fm_r, lm_r, unlearn_labels)
                row = dict(dataset=ds_name, epochs=0, seed=seed,
                           method='Retrain', dr_acc=dr_r, yu_acc=yu_r,
                           lpr_acc=lpr_r, separability=sep_r)
                ds_rows.append(row); grand_rows.append(row)
                print(f"  Retrain: Dr={dr_r:.1f}% yu={yu_r:.1f}% "
                      f"LPR={lpr_r:.1f}%")
                del fm_r, lm_r

                # Vary unlearning epochs for Target
                for n_ep in epoch_values:
                    target_mdl = manifold_mixup_vfl_unlearn(
                        orig, L['Dp_u'], L['Dp_r'],
                        unlearn_epochs=n_ep,
                        lr=(ul_lr * 10 if ctx['is_tabular']
                            else CONFIG['unlearn_lr']),
                        mixup_alpha=CONFIG['mixup_alpha'],
                        device=device, use_adam=ctx['is_tabular'])

                    dr = evaluate_accuracy(target_mdl, L['Dr_test'], device)
                    yu = evaluate_accuracy(target_mdl, L['Du_test'], device)
                    fm, lm = extract_features(target_mdl, L['full_test'],
                                              device)
                    lpr_acc, _ = linear_probe_recovery(
                        target_mdl, L['full_test'], unlearn_labels, device,
                        cached_features=(fm, lm))
                    sep = feature_separability(fm, lm, unlearn_labels)

                    row = dict(dataset=ds_name, epochs=n_ep, seed=seed,
                               method='Target', dr_acc=dr, yu_acc=yu,
                               lpr_acc=lpr_acc, separability=sep)
                    ds_rows.append(row); grand_rows.append(row)
                    print(f"  Target(ep={n_ep:2d}): Dr={dr:.1f}% yu={yu:.1f}% "
                          f"LPR={lpr_acc:.1f}% Sep={sep:.3f}")

                    _free_model(target_mdl, device); del fm, lm

                _cleanup(device, retrained, orig)
                del feat_orig, feat_retr

            except Exception as e:
                print(f"  [ERROR] {ds_name} seed={seed}: {e}")
                import traceback; traceback.print_exc()

        # Save per-dataset CSV
        if ds_rows:
            os.makedirs('results', exist_ok=True)
            df = pd.DataFrame(ds_rows)
            csv_path = f'results/epoch_ablation_{ds_name}.csv'
            df.to_csv(csv_path, index=False)
            print(f"\n  Epoch ablation saved to {csv_path}")

            summary = df.groupby(['epochs', 'method']).agg(
                dr=('dr_acc', 'mean'), yu=('yu_acc', 'mean'),
                lpr=('lpr_acc', 'mean'), sep=('separability', 'mean')
            ).reset_index()
            print(summary.to_string(index=False))

    # Save combined CSV
    if grand_rows:
        os.makedirs('results', exist_ok=True)
        df = pd.DataFrame(grand_rows)
        csv_path = 'results/epoch_ablation_all.csv'
        df.to_csv(csv_path, index=False)
        print(f"Epoch ablation (all datasets) saved to {csv_path} "
              f"({len(df)} rows)")

# Main

def main():
    parser = argparse.ArgumentParser(
        description='Mirage Extended Experiments — All 7 Datasets')
    parser.add_argument('--exp',
                        choices=['yahoo', 'sample', 'kparty', 'tsne',
                                 'classabl', 'epochabl', 'all'],
                        required=True, help='Which experiment to run')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_root', default='./data_raw')
    parser.add_argument('--no-log', action='store_true',
                        help='Disable logging to file')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Directory for log files (default: ./logs)')
    args = parser.parse_args()

    # --- Setup logging ---
    log_file = None
    tee_out = None
    tee_err = None
    if not args.no_log:
        os.makedirs(args.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_name = f"extended_{args.exp}_{timestamp}.log"
        log_path = os.path.join(args.log_dir, log_name)
        log_file = open(log_path, 'w', encoding='utf-8')
        tee_out = TeeStream(log_file, sys.stdout)
        tee_err = TeeStream(log_file, sys.stderr)
        sys.stdout = tee_out
        sys.stderr = tee_err
        print(f"Logging to: {log_path}")

    print(f"Device: {args.device}")
    print(f"Data root: {args.data_root}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    t0 = time.time()

    dispatch = {
        'yahoo':    run_yahoo_experiments,
        'sample':   run_sample_level_experiments,
        'kparty':   run_kparty_experiments,
        'tsne':     run_tsne_visualization,
        'classabl': run_class_ablation,
        'epochabl': run_epoch_ablation,
    }

    try:
        if args.exp == 'all':
            for key in ['yahoo', 'tsne', 'sample', 'kparty',
                         'classabl', 'epochabl']:
                dispatch[key](args.device, args.data_root)
        else:
            dispatch[args.exp](args.device, args.data_root)
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        elapsed = time.time() - t0
        print(f"  ALL DONE! Total time: {elapsed/60:.1f} minutes")
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if log_file:
            sys.stdout = tee_out.stream
            sys.stderr = tee_err.stream
            log_file.close()
            print(f"Log saved: {log_path}")

if __name__ == '__main__':
    main()
