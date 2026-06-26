#!/usr/bin/env python3
"""
Fast cached dataset loaders. Monkey-patches mirage_lib.get_dataset()
to use pre-cached .pt files for slow datasets (ModelNet10, BrainTumor, COVID19).

Usage:
    import fast_datasets  # just import at top of script
    # Now mirage_lib.get_dataset() auto-uses caches when available
"""
import os, sys, torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix: Avoid /dev/shm exhaustion when multiple experiments run
# num_workers>0 in parallel. Use file_system instead of shared memory.
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import mirage_lib

class CachedModelNet10(Dataset):
    """Load ModelNet10 from pre-rendered .pt cache."""
    def __init__(self, root, train=True, transform=None, **kwargs):
        split = 'train' if train else 'test'
        cache_path = os.path.join(root, f'ModelNet10_{split}_cache.pt')
        data = torch.load(cache_path, weights_only=False)
        self.images = data['images']   # (N, 1, 32, 32)
        self.targets = data['labels'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]

class CachedImageDataset(Dataset):
    """Load BrainTumor/COVID19 from pre-cached .pt file.

    Cache stores images after Resize+ToTensor+Normalize.
    Training applies RandomHorizontalFlip at runtime (cheap tensor op).
    """
    def __init__(self, root, name, train=True, **kwargs):
        split = 'train' if train else 'test'
        cache_path = os.path.join(root, f'{name}_{split}_cache.pt')
        data = torch.load(cache_path, weights_only=False)
        self.images = data['images']   # (N, 3, 32, 32)
        self.targets = data['labels'].tolist()
        self.classes = data.get('classes', [])
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        # Apply random horizontal flip for training (same as original transform)
        if self.train and torch.rand(1).item() > 0.5:
            img = torch.flip(img, [2])  # flip width dimension
        return img, self.targets[idx]

# Monkey-patch get_dataset to use caches when available
_orig_get_dataset = mirage_lib.get_dataset

def _fast_get_dataset(name, data_root='./data_raw', **kwargs):
    if name == 'ModelNet10':
        cache_path = os.path.join(data_root, 'ModelNet10_train_cache.pt')
        if os.path.exists(cache_path):
            print(f"[fast_datasets] Using cached ModelNet10")
            train_ds = CachedModelNet10(data_root, train=True,
                                        transform=transforms.Normalize((0.5,), (0.5,)))
            test_ds = CachedModelNet10(data_root, train=False,
                                       transform=transforms.Normalize((0.5,), (0.5,)))
            num_classes = len(set(train_ds.targets))
            return train_ds, test_ds, num_classes, 1, 32

    elif name == 'BrainTumor':
        cache_path = os.path.join(data_root, 'BrainTumor_train_cache.pt')
        if os.path.exists(cache_path):
            print(f"[fast_datasets] Using cached BrainTumor")
            train_ds = CachedImageDataset(data_root, 'BrainTumor', train=True)
            test_ds = CachedImageDataset(data_root, 'BrainTumor', train=False)
            num_classes = len(train_ds.classes)
            return train_ds, test_ds, num_classes, 3, 32

    elif name == 'COVID19':
        cache_path = os.path.join(data_root, 'COVID19_train_cache.pt')
        if os.path.exists(cache_path):
            print(f"[fast_datasets] Using cached COVID19")
            train_ds = CachedImageDataset(data_root, 'COVID19', train=True)
            test_ds = CachedImageDataset(data_root, 'COVID19', train=False)
            num_classes = len(train_ds.classes)
            return train_ds, test_ds, num_classes, 3, 32

    return _orig_get_dataset(name, data_root, **kwargs)

mirage_lib.get_dataset = _fast_get_dataset
print("[fast_datasets] Patched get_dataset + set sharing_strategy='file_system'")
