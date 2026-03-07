"""
Mirage: Exposing the Forgetting Illusion in VFL Unlearning
Complete library — models, datasets, training, unlearning, Mirage audit.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset, random_split
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import os, random, copy, time, warnings, gc
from PIL import Image
from glob import glob

warnings.filterwarnings('ignore')

# ============================================================
# Reproducibility
# ============================================================
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# Hyperparameters
# ============================================================
CONFIG = {
    'lr': 0.01,
    'batch_size': 128,
    'train_epochs': 50,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'unlearn_epochs': 5,
    'unlearn_lr': 0.01,
    'mixup_alpha': 1.0,
    'num_public_samples': 40,
    'num_passive_parties': 2,
    'probe_C': 1.0,
    'probe_max_iter': 1000,
    'cka_num_samples': 5000,
    'num_seeds': 3,
    'mia_shadow_epochs': 20,
}

# ============================================================
# 1. Custom Dataset Classes
# ============================================================

class VFLDataset(Dataset):
    """Wraps a dataset to split features for VFL parties."""
    def __init__(self, dataset, num_parties=2):
        self.dataset = dataset
        self.num_parties = num_parties

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if x.dim() == 3:  # C x H x W — split along width
            w = x.shape[2]
            chunk = w // self.num_parties
            parts = []
            for k in range(self.num_parties):
                s = k * chunk
                e = s + chunk if k < self.num_parties - 1 else w
                parts.append(x[:, :, s:e])
            return parts, y
        else:  # flat features — split along feature dim
            d = x.shape[0]
            chunk = d // self.num_parties
            parts = []
            for k in range(self.num_parties):
                s = k * chunk
                e = s + chunk if k < self.num_parties - 1 else d
                parts.append(x[s:e])
            return parts, y


class ModelNet10Dataset(Dataset):
    """Load ModelNet10 .off meshes and render as 32x32 depth images."""
    def __init__(self, root, train=True, transform=None, img_size=32):
        self.transform = transform
        self.img_size = img_size
        split = 'train' if train else 'test'
        base = os.path.join(root, 'ModelNet10')
        self.classes = sorted([d for d in os.listdir(base)
                               if os.path.isdir(os.path.join(base, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples, self.targets = [], []
        for cls in self.classes:
            cls_dir = os.path.join(base, cls, split)
            if not os.path.isdir(cls_dir):
                continue
            for f in sorted(os.listdir(cls_dir)):
                if f.endswith('.off'):
                    self.samples.append(os.path.join(cls_dir, f))
                    self.targets.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self._render(self.samples[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]

    def _render(self, path):
        verts = self._read_off(path)
        sz = self.img_size
        img = np.zeros((sz, sz), dtype=np.float32)
        if len(verts) == 0:
            return torch.FloatTensor(img).unsqueeze(0)
        verts = verts - verts.mean(0)
        mx = np.abs(verts).max()
        if mx > 1e-6:
            verts /= mx
        xs = np.clip(((verts[:, 0] + 1) / 2 * (sz - 1)).astype(int), 0, sz - 1)
        ys = np.clip(((verts[:, 1] + 1) / 2 * (sz - 1)).astype(int), 0, sz - 1)
        zs = (verts[:, 2] + 1) / 2
        for x, y, z in zip(xs, ys, zs):
            img[y, x] = max(img[y, x], z)
        return torch.FloatTensor(img).unsqueeze(0)  # (1, H, W)

    @staticmethod
    def _read_off(path):
        with open(path, 'r') as f:
            line = f.readline().strip()
            if line == 'OFF':
                parts = f.readline().strip().split()
            elif line.startswith('OFF'):
                parts = line[3:].strip().split()
            else:
                parts = line.split()
            n_verts = int(parts[0])
            verts = []
            for _ in range(n_verts):
                vals = f.readline().strip().split()
                if len(vals) >= 3:
                    verts.append([float(vals[0]), float(vals[1]), float(vals[2])])
        return np.array(verts) if verts else np.zeros((0, 3))


class ImageFolderFlat(Dataset):
    """ImageFolder-style dataset for BrainTumor / COVID19."""
    def __init__(self, root, transform=None):
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root)
                               if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples, self.targets = [], []
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            for f in sorted(os.listdir(cls_dir)):
                if os.path.splitext(f)[1].lower() in exts:
                    self.samples.append(os.path.join(cls_dir, f))
                    self.targets.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


class TensorDatasetWithTargets(Dataset):
    """Tensor dataset that exposes .targets for compatibility."""
    def __init__(self, features, labels):
        self.features = features if isinstance(features, torch.Tensor) else torch.FloatTensor(features)
        self.labels = labels if isinstance(labels, torch.Tensor) else torch.LongTensor(labels)
        self.targets = self.labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================
# 2. Dataset Loading
# ============================================================

def get_dataset(name='CIFAR10', data_root='./data_raw'):
    """Load dataset. Returns (train_ds, test_ds, num_classes, input_channels, img_size).

    For image datasets: input_channels=C, img_size=H (=W).
    For tabular (YahooAnswers): input_channels=0, img_size=feature_dim.
    """
    if name == 'MNIST':
        t = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
        train_ds = torchvision.datasets.MNIST(data_root, True, download=True, transform=t)
        test_ds  = torchvision.datasets.MNIST(data_root, False, download=True, transform=t)
        return train_ds, test_ds, 10, 1, 28

    elif name == 'CIFAR10':
        tr = transforms.Compose([transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
        te = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
        train_ds = torchvision.datasets.CIFAR10(data_root, True, download=True, transform=tr)
        test_ds  = torchvision.datasets.CIFAR10(data_root, False, download=True, transform=te)
        return train_ds, test_ds, 10, 3, 32

    elif name == 'CIFAR100':
        tr = transforms.Compose([transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))])
        te = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))])
        train_ds = torchvision.datasets.CIFAR100(data_root, True, download=True, transform=tr)
        test_ds  = torchvision.datasets.CIFAR100(data_root, False, download=True, transform=te)
        return train_ds, test_ds, 100, 3, 32

    elif name == 'ModelNet10':
        t_train = transforms.Normalize((0.5,), (0.5,))
        t_test  = transforms.Normalize((0.5,), (0.5,))
        train_ds = ModelNet10Dataset(data_root, train=True,  transform=t_train, img_size=32)
        test_ds  = ModelNet10Dataset(data_root, train=False, transform=t_test,  img_size=32)
        num_classes = len(train_ds.classes)
        return train_ds, test_ds, num_classes, 1, 32

    elif name == 'BrainTumor':
        tr = transforms.Compose([transforms.Resize((32, 32)), transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        te = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        train_dir = os.path.join(data_root, 'brain_tumor', 'Training')
        test_dir  = os.path.join(data_root, 'brain_tumor', 'Testing')
        train_ds = ImageFolderFlat(train_dir, transform=tr)
        test_ds  = ImageFolderFlat(test_dir,  transform=te)
        num_classes = len(train_ds.classes)
        return train_ds, test_ds, num_classes, 3, 32

    elif name == 'COVID19':
        tr = transforms.Compose([transforms.Resize((32, 32)), transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        te = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        base = os.path.join(data_root, 'covid19')
        train_dir = os.path.join(base, 'train')
        test_dir  = os.path.join(base, 'test')
        train_ds = ImageFolderFlat(train_dir, transform=tr)
        test_ds  = ImageFolderFlat(test_dir,  transform=te)
        num_classes = len(train_ds.classes)
        return train_ds, test_ds, num_classes, 3, 32

    elif name == 'YahooAnswers':
        feat_dir = os.path.join(data_root, 'yahoo_answers')
        train_npz = os.path.join(feat_dir, 'features_train.npz')
        test_npz  = os.path.join(feat_dir, 'features_test.npz')
        if not os.path.exists(train_npz):
            raise FileNotFoundError(
                f"{train_npz} not found. Run download_yahoo.py first to "
                "download and preprocess Yahoo Answers (TF-IDF).")
        d_train = np.load(train_npz)
        d_test  = np.load(test_npz)
        train_ds = TensorDatasetWithTargets(d_train['X'], d_train['y'])
        test_ds  = TensorDatasetWithTargets(d_test['X'],  d_test['y'])
        feature_dim = d_train['X'].shape[1]
        num_classes = int(d_train['y'].max()) + 1
        return train_ds, test_ds, num_classes, 0, feature_dim

    else:
        raise ValueError(f"Unsupported dataset: {name}")


def prepare_unlearning_data(train_ds, test_ds, unlearn_labels, num_public=40):
    """Split data into Dr, Du, public subsets."""
    if hasattr(train_ds, 'targets'):
        targets = np.array(train_ds.targets)
    elif hasattr(train_ds, 'labels'):
        targets = np.array(train_ds.labels)
    else:
        targets = np.array([y for _, y in train_ds])

    unlearn_mask = np.isin(targets, unlearn_labels)
    retain_idx = np.where(~unlearn_mask)[0].tolist()
    unlearn_idx = np.where(unlearn_mask)[0].tolist()

    Dr_train = Subset(train_ds, retain_idx)
    Du_train = Subset(train_ds, unlearn_idx)

    public_u_idx, public_r_idx = [], []
    for lab in unlearn_labels:
        li = np.where(targets == lab)[0]; np.random.shuffle(li)
        public_u_idx.extend(li[:num_public].tolist())
    retain_labels = sorted(set(range(int(targets.max()) + 1)) - set(unlearn_labels))
    for lab in retain_labels[:5]:
        li = np.where(targets == lab)[0]; np.random.shuffle(li)
        public_r_idx.extend(li[:num_public].tolist())

    Dp_u = Subset(train_ds, public_u_idx)
    Dp_r = Subset(train_ds, public_r_idx)

    if hasattr(test_ds, 'targets'):
        tt = np.array(test_ds.targets)
    else:
        tt = np.array([y for _, y in test_ds])
    tm = np.isin(tt, unlearn_labels)
    Dr_test = Subset(test_ds, np.where(~tm)[0].tolist())
    Du_test = Subset(test_ds, np.where(tm)[0].tolist())

    return Dr_train, Du_train, Dp_u, Dp_r, Dr_test, Du_test


# ============================================================
# 3. Model Definitions
# ============================================================

class ResNet18Bottom(nn.Module):
    def __init__(self, input_channels=3, input_width=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding_dim = 256
    def forward(self, x):
        return self.features(x).view(x.size(0), -1)


class VGG16Bottom(nn.Module):
    def __init__(self, input_channels=3, input_width=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding_dim = 256
    def forward(self, x):
        return self.features(x).view(x.size(0), -1)


class MLPBottom(nn.Module):
    """Bottom model for tabular / text features (e.g. TF-IDF)."""
    def __init__(self, input_dim=2500):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
        )
        self.embedding_dim = 256
    def forward(self, x):
        return self.features(x)


class ActiveModel(nn.Module):
    def __init__(self, embedding_dim, num_parties, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * num_parties, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(True),
            nn.Linear(256, num_classes),
        )
    def forward(self, embeddings):
        return self.classifier(embeddings)


class VFLModel(nn.Module):
    def __init__(self, passive_models, active_model):
        super().__init__()
        self.passive_models = nn.ModuleList(passive_models)
        self.active_model = active_model
        self._intermediate_features = {}

    def forward(self, party_inputs):
        embeddings = []
        for k, (m, x_k) in enumerate(zip(self.passive_models, party_inputs)):
            h = m(x_k)
            self._intermediate_features[f'passive_{k}'] = h
            embeddings.append(h)
        cat = torch.cat(embeddings, dim=1)
        self._intermediate_features['concatenated'] = cat
        return self.active_model(cat)

    def get_features(self, name='concatenated'):
        return self._intermediate_features.get(name, None)


def create_vfl_model(arch, input_channels, input_width, num_parties, num_classes):
    passives = []
    for _ in range(num_parties):
        if arch == 'resnet18':
            passives.append(ResNet18Bottom(input_channels, input_width))
        elif arch == 'vgg16':
            passives.append(VGG16Bottom(input_channels, input_width))
        elif arch == 'mlp':
            passives.append(MLPBottom(input_dim=input_width))
        else:
            raise ValueError(f"Unknown arch: {arch}")
    emb = passives[0].embedding_dim
    active = ActiveModel(emb, num_parties, num_classes)
    return VFLModel(passives, active)


# ============================================================
# 4. Training
# ============================================================

def train_vfl(model, train_loader, epochs, lr=0.01, device='cpu', verbose=True,
              use_adam=False):
    """Train VFL model. Set use_adam=True for MLP/tabular data."""
    model = model.to(device); model.train()
    if use_adam:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    for ep in range(epochs):
        total_loss, correct, total = 0, 0, 0
        it = tqdm(train_loader, desc=f'Epoch {ep+1}/{epochs}', leave=False) if verbose else train_loader
        for party_inputs, labels in it:
            party_inputs = [x.to(device) for x in party_inputs]
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(party_inputs)
            loss = criterion(out, labels)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += out.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        if verbose and (ep + 1) % 10 == 0:
            print(f'  Epoch {ep+1}: Loss={total_loss/total:.4f}, Acc={100.*correct/total:.2f}%')
    # Clear intermediate features to allow safe copy.deepcopy() later
    model._intermediate_features.clear()
    return model


# ============================================================
# 5. Unlearning Methods
# ============================================================

def retrain_from_scratch(arch, in_ch, in_w, n_parties, n_cls, Dr_loader, epochs, lr, device,
                         use_adam=False):
    m = create_vfl_model(arch, in_ch, in_w, n_parties, n_cls)
    return train_vfl(m, Dr_loader, epochs, lr, device, verbose=False, use_adam=use_adam)

def _make_opt(params, lr, use_adam=False):
    """Helper: Adam for tabular, SGD for images."""
    if use_adam:
        return optim.Adam(params, lr=lr, weight_decay=1e-4)
    return optim.SGD(params, lr=lr, momentum=0.9)

def _safe_copy(model, device):
    """Safely deepcopy a VFL model, clearing non-leaf intermediate features first."""
    if hasattr(model, '_intermediate_features'):
        model._intermediate_features.clear()
    m = copy.deepcopy(model).to(device)
    return m

def fine_tuning_unlearn(model, Dr_loader, epochs=5, lr=0.001, device='cpu', use_adam=False):
    m = _safe_copy(model, device); m.train()
    opt = _make_opt(m.parameters(), lr, use_adam)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for px, la in Dr_loader:
            px = [x.to(device) for x in px]; la = la.to(device)
            opt.zero_grad(); crit(m(px), la).backward(); opt.step()
    return m

def fisher_forgetting(model, Du_loader, Dr_loader, device='cpu', alpha=1.0):
    m = _safe_copy(model, device); m.eval()
    crit = nn.CrossEntropyLoss()
    fisher = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
    ns = 0
    for px, la in Du_loader:
        px = [x.to(device) for x in px]; la = la.to(device)
        m.zero_grad(); crit(m(px), la).backward()
        for n, p in m.named_parameters():
            if p.grad is not None: fisher[n] += p.grad.data ** 2
        ns += la.size(0)
    for n in fisher: fisher[n] /= max(ns, 1)
    for n, p in m.named_parameters():
        w = torch.clamp(1.0 / (fisher[n] + 1e-6), max=10.0)
        p.data += torch.randn_like(p) * alpha * w
        p.data = torch.clamp(p.data, -10.0, 10.0)
    return m

def amnesiac_unlearn(model, Du_loader, num_classes, epochs=5, lr=0.001, device='cpu',
                     use_adam=False):
    m = _safe_copy(model, device); m.train()
    opt = _make_opt(m.parameters(), lr, use_adam)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for px, la in Du_loader:
            px = [x.to(device) for x in px]
            rl = torch.randint(0, num_classes, la.shape).to(device)
            opt.zero_grad(); crit(m(px), rl).backward(); opt.step()
    return m

def unsir_unlearn(model, Du_loader, Dr_loader, epochs=5, lr=0.001, noise_scale=0.1,
                  device='cpu', use_adam=False):
    m = _safe_copy(model, device); m.train()
    crit = nn.CrossEntropyLoss()
    opt = _make_opt(m.parameters(), lr, use_adam)
    for _ in range(epochs):
        for px, la in Du_loader:
            px = [x.to(device) for x in px]; la = la.to(device)
            opt.zero_grad(); loss = -crit(m(px), la); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        with torch.no_grad():
            if any(torch.isnan(p).any() for p in m.parameters()): break
    with torch.no_grad():
        for p in m.parameters():
            p.data = torch.where(torch.isnan(p.data), torch.zeros_like(p.data), p.data)
            p.add_(torch.randn_like(p) * noise_scale)
    opt = _make_opt(m.parameters(), lr * 0.1, use_adam)
    for _ in range(epochs):
        for px, la in Dr_loader:
            px = [x.to(device) for x in px]; la = la.to(device)
            opt.zero_grad(); loss = crit(m(px), la)
            if torch.isnan(loss): continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m

def boundary_unlearn(model, Du_loader, Dr_loader, epochs=5, lr=0.001, device='cpu',
                     use_adam=False):
    m = _safe_copy(model, device); m.train()
    opt = _make_opt(m.parameters(), lr, use_adam)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for px, la in Du_loader:
            px = [x.to(device) for x in px]; la = la.to(device)
            opt.zero_grad(); out = m(px)
            with torch.no_grad():
                pr = F.softmax(out, 1); pr.scatter_(1, la.unsqueeze(1), 0)
                nl = pr.argmax(1)
            crit(out, nl).backward(); opt.step()
    return m

def ssd_unlearn(model, Du_loader, Dr_loader, device='cpu', damping=0.1):
    m = _safe_copy(model, device); m.eval()
    crit = nn.CrossEntropyLoss()
    imp_f = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
    imp_r = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
    for px, la in Du_loader:
        px = [x.to(device) for x in px]; la = la.to(device)
        m.zero_grad(); crit(m(px), la).backward()
        for n, p in m.named_parameters():
            if p.grad is not None: imp_f[n] += p.grad.data.abs()
    for px, la in Dr_loader:
        px = [x.to(device) for x in px]; la = la.to(device)
        m.zero_grad(); crit(m(px), la).backward()
        for n, p in m.named_parameters():
            if p.grad is not None: imp_r[n] += p.grad.data.abs()
    with torch.no_grad():
        for n, p in m.named_parameters():
            mask = (imp_f[n] / (imp_r[n] + 1e-8) > 1.0).float()
            p.data *= (1.0 - mask * damping)
    return m

def manifold_mixup_vfl_unlearn(model, Dp_u_loader, Dp_r_loader,
                                unlearn_epochs=5, lr=0.01, mixup_alpha=1.0, device='cpu',
                                use_adam=False):
    m = _safe_copy(model, device); m.train()
    crit = nn.CrossEntropyLoss()
    opt = _make_opt(m.parameters(), lr, use_adam)
    n_cls = m.active_model.classifier[-1].out_features
    for _ in range(unlearn_epochs):
        for px, la in Dp_u_loader:
            px = [x.to(device) for x in px]; la = la.to(device)
            embs = [m.passive_models[k](px[k]) for k in range(len(px))]
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            perm = torch.randperm(la.size(0)).to(device)
            mixed = [lam * h + (1 - lam) * h[perm] for h in embs]
            oh = F.one_hot(la, n_cls).float()
            ml = lam * oh + (1 - lam) * oh[perm]
            out = m.active_model(torch.cat(mixed, 1))
            lp = F.log_softmax(out, 1)
            loss = -(ml * lp).sum(1).mean()
            opt.zero_grad(); loss.backward()
            for p in m.parameters():
                if p.grad is not None: p.grad.data.neg_()
            opt.step()
        for px, la in Dp_r_loader:
            px = [x.to(device) for x in px]; la = la.to(device)
            opt.zero_grad(); crit(m(px), la).backward(); opt.step()
    return m


# ============================================================
# 6. Evaluation & Mirage Audit
# ============================================================

def evaluate_accuracy(model, loader, device='cpu'):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for px, la in loader:
            px = [x.to(device) for x in px]; la = la.to(device)
            correct += model(px).max(1)[1].eq(la).sum().item()
            total += la.size(0)
    return 100. * correct / total if total else 0.0

def measure_runtime(fn, *args, **kwargs):
    t0 = time.time(); r = fn(*args, **kwargs); return r, time.time() - t0

def extract_features(model, loader, device='cpu'):
    model.eval(); feats, labs = [], []
    with torch.no_grad():
        for px, la in loader:
            px = [x.to(device) for x in px]
            _ = model(px)
            feats.append(model.get_features('concatenated').cpu().numpy())
            labs.append(la.numpy())
    model._intermediate_features.clear()
    F_arr = np.concatenate(feats); L_arr = np.concatenate(labs)
    if not np.isfinite(F_arr).all():
        nc = np.isnan(F_arr).sum(); ic = np.isinf(F_arr).sum()
        print(f'    [warn] {nc} NaN, {ic} Inf in features — replacing with 0')
        F_arr = np.nan_to_num(F_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return F_arr, L_arr

def linear_probe_recovery(model, loader, unlearn_labels, device='cpu',
                          probe_type='linear', seed=42, data_fraction=1.0,
                          cached_features=None):
    if cached_features is not None:
        features, labels = cached_features
    else:
        features, labels = extract_features(model, loader, device)
    bl = np.isin(labels, unlearn_labels).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(features, bl, test_size=0.3,
                                           random_state=seed, stratify=bl)
    if data_fraction < 1.0:
        n = max(10, int(len(Xtr) * data_fraction))
        idx = np.random.RandomState(seed).choice(len(Xtr), n, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    if probe_type == 'linear':
        clf = LogisticRegression(C=CONFIG['probe_C'], max_iter=CONFIG['probe_max_iter'],
                                 random_state=seed, solver='lbfgs')
    else:
        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=CONFIG['probe_max_iter'],
                            random_state=seed, early_stopping=True)
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xte)
    # Use balanced accuracy to avoid majority-class trap
    # (e.g. forgetting 1-of-10 classes → 90% by always predicting "retained")
    from sklearn.metrics import balanced_accuracy_score
    acc = balanced_accuracy_score(yte, preds) * 100.
    try: auroc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    except: auroc = 0.5
    return acc, auroc

def linear_CKA(X, Y):
    n = np.linalg.norm
    num = n(Y.T @ X, 'fro') ** 2
    den = n(X.T @ X, 'fro') * n(Y.T @ Y, 'fro')
    if den < 1e-10: return 0.0
    r = float(num / den)
    return r if np.isfinite(r) else 0.0

def compute_cka_similarity(fu, fo, fr, num_samples=5000):
    n = min(num_samples, len(fu))
    idx = np.random.choice(len(fu), n, replace=False)
    return linear_CKA(fu[idx], fo[idx]), linear_CKA(fu[idx], fr[idx])

def feature_separability(features, labels, unlearn_labels):
    fm = np.isin(labels, unlearn_labels)
    ff, fr_ = features[fm], features[~fm]
    if len(ff) < 2 or len(fr_) < 2: return 0.0
    mu_u, mu_r = ff.mean(0), fr_.mean(0)
    vu = np.trace(np.cov(ff.T)) if ff.shape[1] <= ff.shape[0] else np.var(ff)
    vr = np.trace(np.cov(fr_.T)) if fr_.shape[1] <= fr_.shape[0] else np.var(fr_)
    s = float(np.linalg.norm(mu_u - mu_r) ** 2 / (vu + vr + 1e-10))
    return s if np.isfinite(s) else 0.0


# ============================================================
# 7. Experiment Runner
# ============================================================

def _free_model(model, device):
    model.cpu(); model._intermediate_features.clear()
    del model; gc.collect()
    if 'cuda' in str(device): torch.cuda.empty_cache()


def run_single_experiment(dataset_name, arch, unlearn_labels, device='cpu',
                          data_root='./data_raw', train_epochs=None):
    """Run complete experiment for one dataset+arch+labels setting."""
    if train_epochs is None:
        train_epochs = CONFIG['train_epochs']

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Arch: {arch} | Unlearn: {unlearn_labels}")
    print(f"{'='*60}")

    train_ds, test_ds, num_classes, in_ch, img_size = get_dataset(dataset_name, data_root)

    # Auto-select arch + optimizer for tabular data
    is_tabular = (in_ch == 0)
    if is_tabular:
        arch = 'mlp'
        input_width = img_size // CONFIG['num_passive_parties']
        train_lr = 1e-3      # Adam-friendly lr
        print(f"  [info] Tabular data -> arch=mlp, optimizer=Adam, lr={train_lr}")
    else:
        input_width = img_size // CONFIG['num_passive_parties']
        train_lr = CONFIG['lr']

    np_ = CONFIG['num_passive_parties']
    vfl_train = VFLDataset(train_ds, np_)
    vfl_test  = VFLDataset(test_ds, np_)
    Dr_train, Du_train, Dp_u, Dp_r, Dr_test, Du_test = prepare_unlearning_data(
        train_ds, test_ds, unlearn_labels, CONFIG['num_public_samples'])
    vfl_Dr = VFLDataset(Dr_train, np_); vfl_Du = VFLDataset(Du_train, np_)
    vfl_Dpu = VFLDataset(Dp_u, np_); vfl_Dpr = VFLDataset(Dp_r, np_)
    vfl_DrT = VFLDataset(Dr_test, np_); vfl_DuT = VFLDataset(Du_test, np_)

    bs = CONFIG['batch_size']
    train_loader   = DataLoader(vfl_train, bs, shuffle=True,  num_workers=0)
    Dr_loader      = DataLoader(vfl_Dr,    bs, shuffle=True,  num_workers=0)
    Du_loader      = DataLoader(vfl_Du,    bs, shuffle=True,  num_workers=0)
    Dp_u_loader    = DataLoader(vfl_Dpu, min(bs, len(Dp_u)), shuffle=True,  num_workers=0)
    Dp_r_loader    = DataLoader(vfl_Dpr, min(bs, len(Dp_r)), shuffle=True,  num_workers=0)
    Dr_test_loader = DataLoader(vfl_DrT,   bs, shuffle=False, num_workers=0)
    Du_test_loader = DataLoader(vfl_DuT,   bs, shuffle=False, num_workers=0)
    full_test_loader = DataLoader(vfl_test, bs, shuffle=False, num_workers=0)

    # 1. Train original
    print("\n[1/4] Training original VFL model...")
    orig = create_vfl_model(arch, in_ch, input_width, np_, num_classes)
    orig = train_vfl(orig, train_loader, train_epochs, train_lr, device,
                     use_adam=is_tabular)
    bdr = evaluate_accuracy(orig, Dr_test_loader, device)
    byu = evaluate_accuracy(orig, Du_test_loader, device)
    print(f"  Baseline: Dr={bdr:.2f}%, yu={byu:.2f}%")

    # 2. Retrain
    print("\n[2/4] Retraining from scratch...")
    retrained, rt_time = measure_runtime(
        retrain_from_scratch, arch, in_ch, input_width, np_, num_classes,
        Dr_loader, train_epochs, train_lr, device, use_adam=is_tabular)

    # 3. Pre-extract reference features
    print("\n[3/4] Extracting reference features...")
    feat_orig, lab_orig = extract_features(orig, full_test_loader, device)
    feat_retr, _ = extract_features(retrained, full_test_loader, device)

    # 4. Unlearning methods
    ul_lr = 1e-4 if is_tabular else 0.001  # smaller unlearning lr for Adam
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
          CONFIG['unlearn_epochs'], ul_lr * 10 if is_tabular else CONFIG['unlearn_lr'],
          CONFIG['mixup_alpha'], device, is_tabular), None),
    ]

    print("\n[4/4] Running unlearning methods...")
    results = {}
    for name, fn, args, pt in specs:
        print(f"  Running {name}...")
        if name == 'Retrain':
            mdl, rt = retrained, pt
        else:
            mdl, rt = measure_runtime(fn, *args)
        dr = evaluate_accuracy(mdl, Dr_test_loader, device)
        yu = evaluate_accuracy(mdl, Du_test_loader, device)
        fm, lm = extract_features(mdl, full_test_loader, device)
        lpr_acc, lpr_auc = linear_probe_recovery(
            mdl, full_test_loader, unlearn_labels, device, cached_features=(fm, lm))
        cka_o, cka_r = compute_cka_similarity(fm, feat_orig, feat_retr, CONFIG['cka_num_samples'])
        sep = feature_separability(fm, lm, unlearn_labels)
        results[name] = dict(dr_acc=dr, yu_acc=yu, lpr_acc=lpr_acc, lpr_auroc=lpr_auc,
                             cka_original=cka_o, cka_retrain=cka_r, separability=sep, runtime=rt)
        print(f"    Dr={dr:.2f}% | yu={yu:.2f}% | LPR={lpr_acc:.1f}% | "
              f"CKA_O={cka_o:.3f} | CKA_R={cka_r:.3f} | Sep={sep:.3f}")
        del fm, lm
        if name != 'Retrain': _free_model(mdl, device)

    _free_model(retrained, device); _free_model(orig, device)
    del feat_orig, feat_retr, lab_orig; gc.collect()
    if 'cuda' in str(device): torch.cuda.empty_cache()

    # Save results
    os.makedirs('results', exist_ok=True)
    key = f"{dataset_name}_{arch}_{'_'.join(map(str, unlearn_labels))}"
    rows = [{'dataset': dataset_name, 'architecture': arch, 'method': m, **v}
            for m, v in results.items()]
    df = pd.DataFrame(rows)
    csv_path = f'results/{key}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df.to_string(index=False))
    return results


# ============================================================
# 8. Sample-Level Unlearning
# ============================================================

def prepare_sample_unlearning_data(train_ds, test_ds, forget_ratio=0.05,
                                    seed=42, num_public=40):
    """Split data for sample-level unlearning (random sample removal).

    Returns Dr_train, Du_train, Dp_u, Dp_r, test_ds, forget_idx, retain_idx.
    The forget set is a random subset of training data, balanced across classes.
    """
    if hasattr(train_ds, 'targets'):
        targets = np.array(train_ds.targets)
    elif hasattr(train_ds, 'labels'):
        targets = np.array(train_ds.labels)
    else:
        targets = np.array([y for _, y in train_ds])

    rng = np.random.RandomState(seed)
    n = len(train_ds)
    classes = sorted(set(targets.tolist()))

    # Balanced sampling: forget_ratio% from each class
    forget_idx, retain_idx = [], []
    for c in classes:
        c_idx = np.where(targets == c)[0]
        rng.shuffle(c_idx)
        n_forget = max(1, int(len(c_idx) * forget_ratio))
        forget_idx.extend(c_idx[:n_forget].tolist())
        retain_idx.extend(c_idx[n_forget:].tolist())

    Dr_train = Subset(train_ds, retain_idx)
    Du_train = Subset(train_ds, forget_idx)

    # Public samples
    public_u = forget_idx[:num_public]
    public_r = retain_idx[:num_public]
    Dp_u = Subset(train_ds, public_u)
    Dp_r = Subset(train_ds, public_r)

    return Dr_train, Du_train, Dp_u, Dp_r, test_ds, forget_idx, retain_idx


def sample_level_lpr(model, train_ds, forget_idx, retain_idx,
                     num_parties=2, device='cpu', seed=42, max_samples=5000):
    """LPR for sample-level: can a probe distinguish forgotten vs retained samples?

    Returns (accuracy, auroc).
    """
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.RandomState(seed)
    n_per = min(max_samples // 2, len(forget_idx), len(retain_idx))
    f_sub = rng.choice(forget_idx, n_per, replace=False).tolist()
    r_sub = rng.choice(retain_idx, n_per, replace=False).tolist()

    all_idx = f_sub + r_sub
    labels = np.array([1]*n_per + [0]*n_per)  # 1=forgotten, 0=retained

    vfl_ds = VFLDataset(Subset(train_ds, all_idx), num_parties)
    loader = DataLoader(vfl_ds, batch_size=128, shuffle=False, num_workers=0)
    features, _ = extract_features(model, loader, device)

    Xtr, Xte, ytr, yte = train_test_split(
        features, labels, test_size=0.3, random_state=seed, stratify=labels)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, solver='lbfgs')
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xte)
    acc = balanced_accuracy_score(yte, preds) * 100.
    try:
        auroc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    except:
        auroc = 0.5
    return acc, auroc


def run_sample_experiment(dataset_name, arch, forget_ratio=0.05, device='cpu',
                           data_root='./data_raw', train_epochs=None, seed=42):
    """Run sample-level unlearning experiment."""
    if train_epochs is None:
        train_epochs = CONFIG['train_epochs']

    print(f"\n{'='*60}")
    print(f"SAMPLE-LEVEL: {dataset_name} | Arch: {arch} | Forget: {forget_ratio*100:.0f}%")
    print(f"{'='*60}")

    train_ds, test_ds, num_classes, in_ch, img_size = get_dataset(dataset_name, data_root)
    is_tabular = (in_ch == 0)
    if is_tabular:
        arch = 'mlp'
        input_width = img_size // CONFIG['num_passive_parties']
        train_lr = 1e-3
    else:
        input_width = img_size // CONFIG['num_passive_parties']
        train_lr = CONFIG['lr']

    np_ = CONFIG['num_passive_parties']

    # Prepare sample-level data splits
    Dr_train, Du_train, Dp_u, Dp_r, test_full, forget_idx, retain_idx = \
        prepare_sample_unlearning_data(train_ds, test_ds, forget_ratio, seed)

    vfl_train = VFLDataset(train_ds, np_)
    vfl_Dr = VFLDataset(Dr_train, np_); vfl_Du = VFLDataset(Du_train, np_)
    vfl_Dpu = VFLDataset(Dp_u, np_); vfl_Dpr = VFLDataset(Dp_r, np_)
    vfl_test = VFLDataset(test_full, np_)

    bs = CONFIG['batch_size']
    train_loader = DataLoader(vfl_train, bs, shuffle=True, num_workers=0)
    Dr_loader    = DataLoader(vfl_Dr, bs, shuffle=True, num_workers=0)
    Du_loader    = DataLoader(vfl_Du, bs, shuffle=True, num_workers=0)
    Dp_u_loader  = DataLoader(vfl_Dpu, min(bs, len(Dp_u)), shuffle=True, num_workers=0)
    Dp_r_loader  = DataLoader(vfl_Dpr, min(bs, len(Dp_r)), shuffle=True, num_workers=0)
    test_loader  = DataLoader(vfl_test, bs, shuffle=False, num_workers=0)

    # Train original
    print("\n[1/3] Training original VFL model...")
    orig = create_vfl_model(arch, in_ch, input_width, np_, num_classes)
    orig = train_vfl(orig, train_loader, train_epochs, train_lr, device,
                     use_adam=is_tabular)
    base_acc = evaluate_accuracy(orig, test_loader, device)
    print(f"  Baseline test accuracy: {base_acc:.2f}%")

    # Retrain without forgotten samples
    print("\n[2/3] Retraining from scratch (without forgotten samples)...")
    retrained = create_vfl_model(arch, in_ch, input_width, np_, num_classes)
    retrained = train_vfl(retrained, Dr_loader, train_epochs, train_lr, device,
                          use_adam=is_tabular, verbose=False)

    # Unlearning methods
    ul_lr = 1e-4 if is_tabular else 0.001
    specs = [
        ('Retrain', None, None),
        ('FT', fine_tuning_unlearn,
         (orig, Dr_loader, 5, ul_lr, device, is_tabular)),
        ('Amnesiac', amnesiac_unlearn,
         (orig, Du_loader, num_classes, 5, ul_lr, device, is_tabular)),
        ('UNSIR', unsir_unlearn,
         (orig, Du_loader, Dr_loader, 5, ul_lr, 0.1, device, is_tabular)),
        ('BU', boundary_unlearn,
         (orig, Du_loader, Dr_loader, 5, ul_lr, device, is_tabular)),
        ('Target', manifold_mixup_vfl_unlearn,
         (orig, Dp_u_loader, Dp_r_loader,
          CONFIG['unlearn_epochs'], ul_lr * 10 if is_tabular else CONFIG['unlearn_lr'],
          CONFIG['mixup_alpha'], device, is_tabular)),
    ]

    print("\n[3/3] Running unlearning + sample-level LPR...")
    results = {}
    for name, fn, args in specs:
        print(f"  {name}...", end=' ')
        if name == 'Retrain':
            mdl = retrained
        else:
            mdl = fn(*args)
        dr_acc = evaluate_accuracy(mdl, test_loader, device)
        lpr_acc, lpr_auc = sample_level_lpr(
            mdl, train_ds, forget_idx, retain_idx, np_, device, seed)
        results[name] = dict(dr_acc=dr_acc, lpr_acc=lpr_acc, lpr_auroc=lpr_auc)
        print(f"Dr={dr_acc:.2f}% | LPR={lpr_acc:.1f}% | AUROC={lpr_auc:.3f}")
        if name != 'Retrain':
            _free_model(mdl, device)

    _free_model(retrained, device); _free_model(orig, device)
    gc.collect()
    if 'cuda' in str(device): torch.cuda.empty_cache()

    # Save
    os.makedirs('results', exist_ok=True)
    key = f"sample_{dataset_name}_{arch}_{int(forget_ratio*100)}pct"
    rows = [{'dataset': dataset_name, 'architecture': arch,
             'forget_ratio': forget_ratio, 'method': m, **v}
            for m, v in results.items()]
    df = pd.DataFrame(rows)
    csv_path = f'results/{key}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df.to_string(index=False))
    return results


# ============================================================
# 9. K-Party Experiments
# ============================================================

def run_kparty_experiment(dataset_name, arch, unlearn_labels, K_values=(2, 4, 8),
                           device='cpu', data_root='./data_raw', train_epochs=None):
    """Run forgetting gap analysis across different numbers of VFL parties."""
    if train_epochs is None:
        train_epochs = CONFIG['train_epochs']

    train_ds, test_ds, num_classes, in_ch, img_size = get_dataset(dataset_name, data_root)
    is_tabular = (in_ch == 0)
    if is_tabular:
        arch = 'mlp'
        train_lr = 1e-3
    else:
        train_lr = CONFIG['lr']

    all_results = {}

    for K in K_values:
        print(f"\n{'='*60}")
        print(f"K-PARTY: {dataset_name} | K={K} | Unlearn: {unlearn_labels}")
        print(f"{'='*60}")

        if is_tabular:
            in_w = img_size // K
        else:
            in_w = img_size // K

        vfl_train = VFLDataset(train_ds, K)
        vfl_test  = VFLDataset(test_ds, K)
        Dr_train, Du_train, Dp_u, Dp_r, Dr_test, Du_test = prepare_unlearning_data(
            train_ds, test_ds, unlearn_labels, CONFIG['num_public_samples'])
        vfl_Dr = VFLDataset(Dr_train, K); vfl_Du = VFLDataset(Du_train, K)
        vfl_Dpu = VFLDataset(Dp_u, K); vfl_Dpr = VFLDataset(Dp_r, K)
        vfl_DrT = VFLDataset(Dr_test, K); vfl_DuT = VFLDataset(Du_test, K)

        bs = CONFIG['batch_size']
        train_loader   = DataLoader(vfl_train, bs, shuffle=True, num_workers=0)
        Dr_loader      = DataLoader(vfl_Dr, bs, shuffle=True, num_workers=0)
        Du_loader      = DataLoader(vfl_Du, bs, shuffle=True, num_workers=0)
        Dp_u_loader    = DataLoader(vfl_Dpu, min(bs, len(Dp_u)), shuffle=True, num_workers=0)
        Dp_r_loader    = DataLoader(vfl_Dpr, min(bs, len(Dp_r)), shuffle=True, num_workers=0)
        Dr_test_loader = DataLoader(vfl_DrT, bs, shuffle=False, num_workers=0)
        Du_test_loader = DataLoader(vfl_DuT, bs, shuffle=False, num_workers=0)
        full_test_loader = DataLoader(vfl_test, bs, shuffle=False, num_workers=0)

        # Train
        print(f"  Training with K={K} parties...")
        orig = create_vfl_model(arch, in_ch, in_w, K, num_classes)
        orig = train_vfl(orig, train_loader, train_epochs, train_lr, device,
                         use_adam=is_tabular)
        bdr = evaluate_accuracy(orig, Dr_test_loader, device)
        byu = evaluate_accuracy(orig, Du_test_loader, device)
        print(f"  Baseline: Dr={bdr:.2f}%, yu={byu:.2f}%")

        # Retrain
        retrained = create_vfl_model(arch, in_ch, in_w, K, num_classes)
        retrained = train_vfl(retrained, Dr_loader, train_epochs, train_lr, device,
                              use_adam=is_tabular, verbose=False)

        feat_orig, lab_orig = extract_features(orig, full_test_loader, device)
        feat_retr, _ = extract_features(retrained, full_test_loader, device)

        # Target method + Retrain for comparison
        ul_lr = 1e-4 if is_tabular else 0.001
        methods = {
            'Retrain': retrained,
            'Target': manifold_mixup_vfl_unlearn(
                orig, Dp_u_loader, Dp_r_loader,
                CONFIG['unlearn_epochs'],
                ul_lr * 10 if is_tabular else CONFIG['unlearn_lr'],
                CONFIG['mixup_alpha'], device, is_tabular),
            'FT': fine_tuning_unlearn(orig, Dr_loader, 5, ul_lr, device, is_tabular),
            'BU': boundary_unlearn(orig, Du_loader, Dr_loader, 5, ul_lr, device, is_tabular),
        }

        k_results = {}
        for name, mdl in methods.items():
            dr = evaluate_accuracy(mdl, Dr_test_loader, device)
            yu = evaluate_accuracy(mdl, Du_test_loader, device)
            fm, lm = extract_features(mdl, full_test_loader, device)
            lpr_acc, lpr_auc = linear_probe_recovery(
                mdl, full_test_loader, unlearn_labels, device,
                cached_features=(fm, lm))
            cka_o, cka_r = compute_cka_similarity(fm, feat_orig, feat_retr,
                                                   CONFIG['cka_num_samples'])
            sep = feature_separability(fm, lm, unlearn_labels)
            k_results[name] = dict(K=K, dr_acc=dr, yu_acc=yu, lpr_acc=lpr_acc,
                                   cka_original=cka_o, cka_retrain=cka_r,
                                   separability=sep)
            print(f"  K={K} {name}: Dr={dr:.2f}% yu={yu:.2f}% "
                  f"LPR={lpr_acc:.1f}% CKA_O={cka_o:.3f} Sep={sep:.3f}")
            del fm, lm

        all_results[K] = k_results

        for name, mdl in methods.items():
            _free_model(mdl, device)
        _free_model(orig, device)
        del feat_orig, feat_retr; gc.collect()
        if 'cuda' in str(device): torch.cuda.empty_cache()

    # Save
    os.makedirs('results', exist_ok=True)
    rows = []
    for K, kr in all_results.items():
        for m, v in kr.items():
            rows.append({'dataset': dataset_name, 'architecture': arch,
                         'method': m, **v})
    df = pd.DataFrame(rows)
    csv_path = f'results/kparty_{dataset_name}_{arch}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nK-party results saved to {csv_path}")
    print(df.to_string(index=False))
    return all_results


# ============================================================
# 10. t-SNE Visualization
# ============================================================

def visualize_tsne(model_dict, loader, unlearn_labels, device='cpu',
                   save_path=None, max_samples=3000, seed=42):
    """Generate t-SNE visualization comparing methods.

    Args:
        model_dict: dict of {method_name: vfl_model}
        loader: VFL DataLoader for full test set
        unlearn_labels: list of forgotten label indices
        save_path: path to save figure (e.g., 'figures/tsne_visualization.pdf')
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib or sklearn not available for t-SNE.")
        return

    methods = list(model_dict.keys())
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4.5))
    if n_methods == 1:
        axes = [axes]

    for i, name in enumerate(methods):
        mdl = model_dict[name]
        feats, labs = extract_features(mdl, loader, device)

        # Subsample for speed
        rng = np.random.RandomState(seed)
        n = min(max_samples, len(feats))
        idx = rng.choice(len(feats), n, replace=False)
        feats_sub = feats[idx]
        labs_sub = labs[idx]

        is_forget = np.isin(labs_sub, unlearn_labels)

        try:
            tsne = TSNE(n_components=2, random_state=seed, perplexity=30,
                         max_iter=1000, init='pca', learning_rate='auto')
        except TypeError:
            tsne = TSNE(n_components=2, random_state=seed, perplexity=30,
                         n_iter=1000, init='pca', learning_rate='auto')
        emb = tsne.fit_transform(feats_sub)

        ax = axes[i]
        # Plot retained samples
        ax.scatter(emb[~is_forget, 0], emb[~is_forget, 1],
                   c='#4A90D9', alpha=0.25, s=4, label='Retained', rasterized=True)
        # Plot forgotten samples
        ax.scatter(emb[is_forget, 0], emb[is_forget, 1],
                   c='#E74C3C', alpha=0.6, s=12, label='Forgotten', rasterized=True)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.legend(loc='lower right', fontsize=9, framealpha=0.8)

        del feats, labs, feats_sub, emb

    plt.tight_layout(pad=1.0)
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                     exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  t-SNE figure saved to {save_path}")
    plt.close(fig)
    return fig
