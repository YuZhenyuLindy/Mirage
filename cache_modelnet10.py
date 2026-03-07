#!/usr/bin/env python3
"""
Pre-render ModelNet10 .off files to a cached .pt file.
This avoids repeated file I/O + Python rendering in DataLoader.

After running this, the cached file is automatically detected
by the patched ModelNet10Dataset.

Usage:
    python cache_modelnet10.py            # uses ./data_raw
    python cache_modelnet10.py --data-root /path/to/data
"""
import argparse, os, sys, time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mirage_lib import ModelNet10Dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data_raw")
    args = parser.parse_args()

    for split_name, train in [('train', True), ('test', False)]:
        print(f"Rendering ModelNet10 {split_name} split...")
        ds = ModelNet10Dataset(args.data_root, train=train)
        images, labels = [], []
        t0 = time.time()
        for i in range(len(ds)):
            img, lab = ds[i]
            images.append(img)
            labels.append(lab)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(ds)}")
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        cache_path = os.path.join(args.data_root, f'ModelNet10_{split_name}_cache.pt')
        torch.save({'images': images, 'labels': labels}, cache_path)
        print(f"  Saved {cache_path}: {images.shape}, {time.time()-t0:.1f}s")

    print("Done! Now ModelNet10 experiments will load from cache.")

if __name__ == "__main__":
    main()
