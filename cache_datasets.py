#!/usr/bin/env python3
"""
Pre-cache slow datasets (ModelNet10, BrainTumor, COVID19) to .pt files.
Avoids repeated disk I/O + PIL decode + Python rendering in DataLoader.

After running this, fast_datasets.py auto-detects and uses the caches.

Usage:
    python cache_datasets.py                    # cache all 3
    python cache_datasets.py --data-root /path/to/data
"""
import argparse, os, sys, time
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mirage_lib import ModelNet10Dataset, ImageFolderFlat

def cache_modelnet10(data_root):
    """Cache ModelNet10: .off → rendered depth images."""
    for split_name, train in [('train', True), ('test', False)]:
        cache_path = os.path.join(data_root, f'ModelNet10_{split_name}_cache.pt')
        if os.path.exists(cache_path):
            print(f"  [skip] {cache_path} already exists")
            continue
        print(f"  Rendering ModelNet10 {split_name}...")
        ds = ModelNet10Dataset(data_root, train=train)
        images, labels = [], []
        t0 = time.time()
        for i in range(len(ds)):
            img, lab = ds[i]
            images.append(img)
            labels.append(lab)
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(ds)}")
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        torch.save({'images': images, 'labels': labels}, cache_path)
        print(f"  Saved {cache_path}: {images.shape}, {time.time()-t0:.1f}s")

def cache_image_folder(data_root, name, train_dir, test_dir):
    """Cache image folder dataset: disk images → resized 32x32 tensors.

    Saves images AFTER Resize+ToTensor+Normalize (no random augmentation).
    RandomHorizontalFlip is applied at runtime by CachedImageDataset.
    """
    t_cache = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    for split_name, split_dir in [('train', train_dir), ('test', test_dir)]:
        cache_path = os.path.join(data_root, f'{name}_{split_name}_cache.pt')
        if os.path.exists(cache_path):
            print(f"  [skip] {cache_path} already exists")
            continue
        if not os.path.isdir(split_dir):
            print(f"  [skip] {split_dir} not found")
            continue
        print(f"  Caching {name} {split_name} from {split_dir}...")
        ds = ImageFolderFlat(split_dir, transform=t_cache)
        images, labels = [], []
        t0 = time.time()
        for i in range(len(ds)):
            img, lab = ds[i]
            images.append(img)
            labels.append(lab)
            if (i + 1) % 1000 == 0:
                print(f"    {i+1}/{len(ds)}")
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        classes = ds.classes
        torch.save({'images': images, 'labels': labels, 'classes': classes}, cache_path)
        print(f"  Saved {cache_path}: {images.shape}, {time.time()-t0:.1f}s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data_raw")
    args = parser.parse_args()
    dr = args.data_root

    print("=" * 50)
    print("  Caching slow datasets for fast DataLoader")
    print("=" * 50)

    # ModelNet10
    mn_dir = os.path.join(dr, 'ModelNet10')
    if os.path.isdir(mn_dir):
        print("\n[1/3] ModelNet10")
        cache_modelnet10(dr)
    else:
        print(f"\n[1/3] ModelNet10 — skipped ({mn_dir} not found)")

    # BrainTumor
    bt_train = os.path.join(dr, 'brain_tumor', 'Training')
    bt_test = os.path.join(dr, 'brain_tumor', 'Testing')
    if os.path.isdir(bt_train):
        print("\n[2/3] BrainTumor")
        cache_image_folder(dr, 'BrainTumor', bt_train, bt_test)
    else:
        print(f"\n[2/3] BrainTumor — skipped ({bt_train} not found)")

    # COVID19
    cv_train = os.path.join(dr, 'covid19', 'train')
    cv_test = os.path.join(dr, 'covid19', 'test')
    if os.path.isdir(cv_train):
        print("\n[3/3] COVID19")
        cache_image_folder(dr, 'COVID19', cv_train, cv_test)
    else:
        print(f"\n[3/3] COVID19 — skipped ({cv_train} not found)")

    print("\nDone! All cached datasets will be auto-loaded by fast_datasets.py")

if __name__ == "__main__":
    main()
