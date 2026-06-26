#!/usr/bin/env python3
"""
Download and prepare the 4 missing datasets for Mirage experiments.

Expected final structure under DATA_ROOT (default: ./data_raw):
  ModelNet10/          ← .off mesh files
  brain_tumor/Training/  brain_tumor/Testing/  ← 4-class MRI images
  covid19/train/         covid19/test/         ← 4-class X-ray images
  yahoo_answers/features_train.npz  features_test.npz  ← TF-IDF

Usage:
    pip install kaggle datasets scikit-learn  # if not installed
    python prepare_datasets.py [--data-root ./data_raw]
"""

import argparse, os, sys, shutil, subprocess, zipfile
import numpy as np

def run(cmd, **kwargs):
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kwargs)

# ModelNet10
def prepare_modelnet10(data_root):
    dest = os.path.join(data_root, 'ModelNet10')
    if os.path.isdir(dest) and len(os.listdir(dest)) >= 10:
        print("[ModelNet10] Already exists, skipping.")
        return

    print("[ModelNet10] Downloading...")
    url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    zip_path = os.path.join(data_root, 'ModelNet10.zip')

    # Try wget, then curl
    try:
        run(f'wget -q --show-progress -O "{zip_path}" "{url}"')
    except Exception:
        run(f'curl -L -o "{zip_path}" "{url}"')

    print("[ModelNet10] Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_root)
    os.remove(zip_path)

    if os.path.isdir(dest):
        print(f"[ModelNet10] OK — {len(os.listdir(dest))} classes")
    else:
        print("[ModelNet10] WARNING: extraction may have failed, check manually")

# Brain Tumor MRI
def prepare_brain_tumor(data_root):
    dest = os.path.join(data_root, 'brain_tumor')
    train_dir = os.path.join(dest, 'Training')
    test_dir = os.path.join(dest, 'Testing')

    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        print("[BrainTumor] Already exists, skipping.")
        return

    print("[BrainTumor] Downloading from Kaggle...")
    zip_path = os.path.join(data_root, 'brain-tumor-mri-dataset.zip')

    try:
        run(f'kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset '
            f'-p "{data_root}" --force')
    except Exception as e:
        print(f"[BrainTumor] kaggle CLI failed: {e}")
        print("  Please download manually from:")
        print("  https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
        print(f"  Extract to: {dest}/Training/ and {dest}/Testing/")
        return

    print("[BrainTumor] Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest)
    os.remove(zip_path)

    # The Kaggle dataset may extract with an extra nesting level
    # Check and fix if needed
    for sub in ['Training', 'Testing']:
        expected = os.path.join(dest, sub)
        if not os.path.isdir(expected):
            # Try to find it nested
            for root, dirs, files in os.walk(dest):
                if sub in dirs and root != dest:
                    src = os.path.join(root, sub)
                    shutil.move(src, expected)
                    print(f"  Moved {src} → {expected}")
                    break

    if os.path.isdir(train_dir):
        classes = [d for d in os.listdir(train_dir)
                   if os.path.isdir(os.path.join(train_dir, d))]
        print(f"[BrainTumor] OK — {len(classes)} classes: {classes}")
    else:
        print("[BrainTumor] WARNING: directory structure may need manual fixing")

# COVID-19 Radiography
def prepare_covid19(data_root):
    dest = os.path.join(data_root, 'covid19')
    train_dir = os.path.join(dest, 'train')
    test_dir = os.path.join(dest, 'test')

    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        print("[COVID19] Already exists, skipping.")
        return

    print("[COVID19] Downloading from Kaggle...")
    zip_path = os.path.join(data_root, 'covid19-radiography-database.zip')

    try:
        run(f'kaggle datasets download -d tawsifurrahman/covid19-radiography-database '
            f'-p "{data_root}" --force')
    except Exception as e:
        print(f"[COVID19] kaggle CLI failed: {e}")
        print("  Please download manually from:")
        print("  https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database")
        print(f"  Then run this script again.")
        return

    print("[COVID19] Extracting...")
    # Find the downloaded zip
    if not os.path.exists(zip_path):
        zips = [f for f in os.listdir(data_root) if 'covid' in f.lower() and f.endswith('.zip')]
        if zips:
            zip_path = os.path.join(data_root, zips[0])

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(os.path.join(data_root, '_covid_tmp'))
    os.remove(zip_path)

    # The Kaggle dataset has: COVID-19_Radiography_Dataset/{COVID,Normal,...}/images/
    # We need to restructure into train/test splits with 80/20 split
    print("[COVID19] Restructuring into train/test split...")
    src_base = os.path.join(data_root, '_covid_tmp')

    # Find the actual data directory
    actual_base = src_base
    for root, dirs, files in os.walk(src_base):
        if any(d in ['COVID', 'COVID-19', 'Normal'] for d in dirs):
            actual_base = root
            break

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    np.random.seed(42)
    for cls_name in sorted(os.listdir(actual_base)):
        cls_path = os.path.join(actual_base, cls_name)
        if not os.path.isdir(cls_path):
            continue

        # Find images (may be in images/ subfolder)
        img_dir = os.path.join(cls_path, 'images')
        if not os.path.isdir(img_dir):
            img_dir = cls_path

        imgs = [f for f in os.listdir(img_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not imgs:
            continue

        np.random.shuffle(imgs)
        split = int(len(imgs) * 0.8)
        train_imgs = imgs[:split]
        test_imgs = imgs[split:]

        # Sanitize class name
        safe_name = cls_name.replace(' ', '_')
        os.makedirs(os.path.join(train_dir, safe_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, safe_name), exist_ok=True)

        for img in train_imgs:
            shutil.copy2(os.path.join(img_dir, img),
                         os.path.join(train_dir, safe_name, img))
        for img in test_imgs:
            shutil.copy2(os.path.join(img_dir, img),
                         os.path.join(test_dir, safe_name, img))

        print(f"  {safe_name}: {len(train_imgs)} train, {len(test_imgs)} test")

    # Clean up
    shutil.rmtree(os.path.join(data_root, '_covid_tmp'), ignore_errors=True)

    classes = [d for d in os.listdir(train_dir)
               if os.path.isdir(os.path.join(train_dir, d))]
    print(f"[COVID19] OK — {len(classes)} classes: {classes}")

# Yahoo Answers (TF-IDF features)
def prepare_yahoo_answers(data_root):
    dest = os.path.join(data_root, 'yahoo_answers')
    train_npz = os.path.join(dest, 'features_train.npz')
    test_npz = os.path.join(dest, 'features_test.npz')

    if os.path.exists(train_npz) and os.path.exists(test_npz):
        print("[YahooAnswers] Already exists, skipping.")
        return

    print("[YahooAnswers] Preparing TF-IDF features...")

    # Try HuggingFace datasets first
    try:
        from datasets import load_dataset
        print("  Loading from HuggingFace...")
        ds = load_dataset("yahoo_answers_topics")
        train_texts = ds['train']['question_title']
        train_labels = ds['train']['topic']
        test_texts = ds['test']['question_title']
        test_labels = ds['test']['topic']
    except Exception as e1:
        print(f"  HuggingFace failed: {e1}")
        # Try alternative: yahoo_answers_csv from kaggle
        try:
            print("  Trying Kaggle...")
            import pandas as pd
            csv_dir = os.path.join(data_root, '_yahoo_tmp')
            os.makedirs(csv_dir, exist_ok=True)
            run(f'kaggle datasets download -d soumikrakshit/yahoo-answers-dataset '
                f'-p "{csv_dir}" --force')
            zips = [f for f in os.listdir(csv_dir) if f.endswith('.zip')]
            for z in zips:
                with zipfile.ZipFile(os.path.join(csv_dir, z), 'r') as zf:
                    zf.extractall(csv_dir)

            # Find CSV files
            train_csv = None
            test_csv = None
            for root, dirs, files in os.walk(csv_dir):
                for f in files:
                    if 'train' in f.lower() and f.endswith('.csv'):
                        train_csv = os.path.join(root, f)
                    elif 'test' in f.lower() and f.endswith('.csv'):
                        test_csv = os.path.join(root, f)

            df_train = pd.read_csv(train_csv, header=None)
            df_test = pd.read_csv(test_csv, header=None)
            # Format: label(1-indexed), title, content, answer
            train_labels = (df_train[0] - 1).tolist()
            train_texts = df_train[1].fillna('').tolist()
            test_labels = (df_test[0] - 1).tolist()
            test_texts = df_test[1].fillna('').tolist()
            shutil.rmtree(csv_dir, ignore_errors=True)
        except Exception as e2:
            print(f"  Kaggle also failed: {e2}")
            print("  Please install: pip install datasets")
            print("  Or: pip install kaggle && kaggle datasets download ...")
            return

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # Subsample: 5000 train + 2000 test per class (following the paper)
    n_train_per_class = 5000
    n_test_per_class = 2000
    num_classes = len(set(train_labels))

    print(f"  {num_classes} classes, subsampling {n_train_per_class} train / "
          f"{n_test_per_class} test per class...")

    np.random.seed(42)
    train_idx, test_idx = [], []
    for c in range(num_classes):
        tr_c = np.where(train_labels == c)[0]
        np.random.shuffle(tr_c)
        train_idx.extend(tr_c[:n_train_per_class].tolist())

        te_c = np.where(test_labels == c)[0]
        np.random.shuffle(te_c)
        test_idx.extend(te_c[:n_test_per_class].tolist())

    sub_train_texts = [train_texts[i] for i in train_idx]
    sub_train_labels = train_labels[train_idx]
    sub_test_texts = [test_texts[i] for i in test_idx]
    sub_test_labels = test_labels[test_idx]

    # TF-IDF vectorization
    from sklearn.feature_extraction.text import TfidfVectorizer

    print(f"  Computing TF-IDF (max_features=5000)...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english',
                            sublinear_tf=True)
    X_train = tfidf.fit_transform(sub_train_texts).toarray().astype(np.float32)
    X_test = tfidf.transform(sub_test_texts).toarray().astype(np.float32)

    os.makedirs(dest, exist_ok=True)
    np.savez(train_npz, X=X_train, y=sub_train_labels)
    np.savez(test_npz, X=X_test, y=sub_test_labels)

    print(f"[YahooAnswers] OK — train: {X_train.shape}, test: {X_test.shape}, "
          f"feature_dim={X_train.shape[1]}")

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='./data_raw')
    args = parser.parse_args()

    os.makedirs(args.data_root, exist_ok=True)
    print(f"Data root: {args.data_root}\n")

    prepare_modelnet10(args.data_root)
    print()
    prepare_brain_tumor(args.data_root)
    print()
    prepare_covid19(args.data_root)
    print()
    prepare_yahoo_answers(args.data_root)

    print("\n" + "=" * 50)
    print("Dataset preparation complete!")
    print("=" * 50)

if __name__ == '__main__':
    main()
