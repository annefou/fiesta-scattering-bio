# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Step 01 — Extract scattering features
#
# Computes 246-dim Cross Scattering Transform features via
# [FOSCAT](https://github.com/jmdelouis/FOSCAT) on three splits defined by
# Decrop et al. 2025:
#
# - **train (balanced)** — up to 100 images per class from their `train.txt`
#   (≈ 9,444 images total). The balanced subset lets a scattering-only LR
#   classifier give each class equal weight, exploiting scattering's lack of
#   training-frequency bias.
# - **val** — their full `val.txt` (33,829 images), used as meta-training
#   data for the stacking classifier in step 03.
# - **test** — their full `test.txt` (33,718 images), used for final
#   evaluation.
#
# Configuration: `NORIENT=8`, `KERNELSZ=3`, RGB per-channel, 64×64 input.
# Feature dimension per image: 3 channels × (2 + 40 + 40) = 246.
#
# ## Data dependencies
#
# On first run this notebook downloads:
#
# 1. The LifeWatch FlowCam plankton dataset from
#    [Zenodo 10.5281/zenodo.10554845](https://doi.org/10.5281/zenodo.10554845)
#    (~650 MB, 337,567 images, 95 classes).
# 2. Decrop et al.'s own train/val/test split files, which are distributed
#    with their pretrained model on
#    [Zenodo 10.5281/zenodo.15269453](https://doi.org/10.5281/zenodo.15269453).
#    We extract only the `dataset_files/` subfolder (~3 MB) from that
#    tarball — the 47 MB model itself is not needed for scattering.

# %%
import os
import subprocess
import tarfile
import time
import urllib.request
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
import foscat.scat_cov as sc

# %%
PROJECT = Path.cwd()
DATA_DIR = PROJECT / 'data'
IMAGES_DIR = DATA_DIR / 'images_DS'
SPLIT_DIR = PROJECT / 'splits'                       # dataset_files from Zenodo 15269453
RESULTS = PROJECT / 'results'
RESULTS.mkdir(exist_ok=True)

TARGET_SIZE = 64
NORIENT = 8
KERNELSZ = 3
MAX_PER_CLASS = 100     # balanced training cap
SEED = 42

CI_MODE = os.environ.get('CI', '').lower() in ('true', '1')
CI_LIMIT = 300 if CI_MODE else None                  # per split

print(f'CI_MODE = {CI_MODE}')

# %% [markdown]
# ## 1. Download dataset

# %%
if not IMAGES_DIR.exists():
    print('Downloading LifeWatch FlowCam dataset (Zenodo 10554845, ~650 MB)...')
    archive = DATA_DIR / 'phytoplankton.7z'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        url = ('https://zenodo.org/records/10554845/files/'
               'phytoplankton_images_and_datasplit.7z?download=1')
        urllib.request.urlretrieve(url, str(archive))
        print(f'  Downloaded {archive.stat().st_size / 1e6:.0f} MB')
    print('  Extracting with 7z...')
    subprocess.run(['7z', 'x', str(archive), f'-o{archive.parent}', '-y'],
                   check=True)
else:
    print(f'Dataset present at {IMAGES_DIR}')

# %% [markdown]
# ## 2. Download Decrop's split files

# %%
if not (SPLIT_DIR / 'train.txt').exists():
    print('Downloading Decrop\'s split files (from pretrained model, Zenodo 15269453)...')
    tar_path = PROJECT / 'Phytoplankton_EfficientNetV2B0.tar.gz'
    if not tar_path.exists():
        url = ('https://zenodo.org/records/15269453/files/'
               'Phytoplankton_EfficientNetV2B0.tar.gz?download=1')
        urllib.request.urlretrieve(url, str(tar_path))
    # Extract only dataset_files/
    SPLIT_DIR.mkdir(exist_ok=True)
    with tarfile.open(tar_path, 'r:gz') as tf:
        for member in tf.getmembers():
            if 'dataset_files/' in member.name and member.isfile():
                name = member.name.split('dataset_files/')[-1]
                if not name:
                    continue
                dest = SPLIT_DIR / name
                extracted = tf.extractfile(member)
                if extracted is not None:
                    dest.write_bytes(extracted.read())
    print(f'  Extracted split files to {SPLIT_DIR}')
else:
    print(f'Split files present at {SPLIT_DIR}')

# %% [markdown]
# ## 3. Parse splits, build balanced training set

# %%
with open(SPLIT_DIR / 'classes.txt') as f:
    class_names = [ln.strip() for ln in f if ln.strip()]
N_CLASSES = len(class_names)
print(f'Classes: {N_CLASSES}')

def load_split(path):
    paths, labels = [], []
    with open(path) as f:
        for line in f:
            rel, lab = line.strip().rsplit(' ', 1)
            full = IMAGES_DIR / rel
            if full.exists():
                paths.append(str(full))
                labels.append(int(lab))
    return np.array(paths), np.array(labels)

val_paths,  val_labels  = load_split(SPLIT_DIR / 'val.txt')
test_paths, test_labels = load_split(SPLIT_DIR / 'test.txt')

# Balanced train: group train.txt by class, cap at MAX_PER_CLASS
train_by_class = defaultdict(list)
with open(SPLIT_DIR / 'train.txt') as f:
    for line in f:
        rel, lab = line.strip().rsplit(' ', 1)
        train_by_class[int(lab)].append(rel)

rng = np.random.default_rng(SEED)
tr_paths, tr_labels = [], []
for cls in range(N_CLASSES):
    files = train_by_class.get(cls, [])
    if not files:
        continue
    k = min(MAX_PER_CLASS, len(files))
    chosen = rng.choice(files, size=k, replace=False) if len(files) > k else files
    for rel in chosen:
        full = IMAGES_DIR / rel
        if full.exists():
            tr_paths.append(str(full))
            tr_labels.append(cls)
tr_paths = np.array(tr_paths)
tr_labels = np.array(tr_labels)

if CI_LIMIT is not None:
    tr_paths,   tr_labels   = tr_paths[:CI_LIMIT],   tr_labels[:CI_LIMIT]
    val_paths,  val_labels  = val_paths[:CI_LIMIT],  val_labels[:CI_LIMIT]
    test_paths, test_labels = test_paths[:CI_LIMIT], test_labels[:CI_LIMIT]

print(f'Balanced train: {len(tr_paths)}  val: {len(val_paths)}  test: {len(test_paths)}')

# %% [markdown]
# ## 4. Feature extraction (per-channel RGB, NORIENT=8)

# %%
scat = sc.funct(NORIENT=NORIENT, KERNELSZ=KERNELSZ,
                all_type='float32', silent=True, use_2D=True)
print(f'FOSCAT device: {scat.backend.device}')


def extract(paths, tag):
    t0 = time.time()
    out, kept = [], []
    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert('RGB').resize((TARGET_SIZE, TARGET_SIZE))
            arr = np.array(im, dtype=np.float32) / 255.0
        except Exception:
            continue
        ch_feats = []
        for c in range(3):
            f = scat.eval(arr[..., c].reshape(1, TARGET_SIZE, TARGET_SIZE))
            parts = []
            for attr in ('S0', 'S1', 'S2', 'P00', 'P11', 'P01'):
                if hasattr(f, attr):
                    parts.append(scat.backend.to_numpy(getattr(f, attr)).ravel())
            ch_feats.append(np.concatenate(parts))
        out.append(np.concatenate(ch_feats))
        kept.append(i)
        if (i + 1) % 2000 == 0 or i + 1 == len(paths):
            rate = (i + 1) / (time.time() - t0)
            eta  = (len(paths) - i - 1) / rate
            print(f'  [{tag}] {i+1}/{len(paths)}  {rate:.1f} img/s  ETA {eta/60:.1f} min')
    return np.array(out), np.array(kept)

# %%
suffix = '_smoke' if CI_MODE else ''

for name, paths, labels in [
    ('train_balanced', tr_paths,  tr_labels),
    ('val',            val_paths, val_labels),
    ('test',           test_paths, test_labels),
]:
    out_path = RESULTS / f'features_{name}{suffix}.npz'
    if out_path.exists():
        d = np.load(out_path)
        print(f'[{name}] cached: {d["X"].shape}  -> {out_path.name}')
        continue
    print(f'Extracting [{name}] ({len(paths)} images)...')
    X, keep = extract(paths, name)
    y = labels[keep]
    np.savez_compressed(out_path, X=X, y=y, paths=paths[keep])
    print(f'  saved {out_path.name}: {X.shape}')

print('\nStep 01 complete.')
