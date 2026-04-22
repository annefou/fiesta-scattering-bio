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
# # Step 02 — CNN predictions on val + test
#
# Runs [Decrop et al. 2025](https://doi.org/10.3389/fmars.2025.1699781)'s
# pretrained EfficientNetV2-B0 on the **val** and **test** splits.
# Saves full 95-class softmax matrices to `results/cnn_predictions_*.npz`
# for the stacking step (03).
#
# This step needs TensorFlow 2.19 and the
# [`planktonclas`](https://github.com/lifewatch/planktonclas) toolkit, which
# conflict with the FOSCAT PyTorch stack used by steps 01 and 03. Two
# equivalent ways to run this notebook:
#
# **Option A — reuse the reproduction repo's environment (recommended).**
# The notebook is a near-duplicate of scripts in
# [fiesta-decrop-reproduction](https://github.com/annefou/fiesta-decrop-reproduction).
# Clone that repo, activate its environment, run 01 and 02 there, and copy
# (or symlink) these outputs into this repo's `results/`:
#
# ```bash
# # from fiesta-decrop-reproduction clone with its env active
# python 01_reproduce_decrop.py    # produces cnn_predictions for test.txt
# python 02_cnn_val_predictions.py # produces cnn_predictions for val.txt
# ```
#
# Then link into this repo:
#
# ```bash
# cd fiesta-scattering-bio/results
# ln -s ../../fiesta-decrop-reproduction/results/reproduce_decrop_predictions.npz cnn_predictions_test.npz
# ln -s ../../fiesta-decrop-reproduction/results/cnn_predictions_val.npz             cnn_predictions_val.npz
# ```
#
# **Option B — dedicated CNN venv inside this repo.** See README.
#
# This notebook *assumes* a usable planktonclas environment is active and
# the pretrained model has been downloaded to `models/Phytoplankton_EfficientNetV2B0/`.

# %%
import json
import os
import sys
import tarfile
import time
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np

PROJECT = Path.cwd()
IMAGES_DIR = PROJECT / 'data/images_DS'
SPLIT_DIR = PROJECT / 'splits'
MODELS_DIR = PROJECT / 'models'
TIMESTAMP = 'Phytoplankton_EfficientNetV2B0'
MODEL_DIR = MODELS_DIR / TIMESTAMP
MODEL_NAME = 'final_model.h5'
RESULTS = PROJECT / 'results'
RESULTS.mkdir(exist_ok=True)

CI_MODE = os.environ.get('CI', '').lower() in ('true', '1')
CI_LIMIT = 200 if CI_MODE else None

# %% [markdown]
# ## 1. Ensure pretrained model is present

# %%
if not (MODEL_DIR / 'ckpts' / MODEL_NAME).exists():
    print('Downloading pretrained EfficientNetV2-B0 (Zenodo 15269453, 47 MB)...')
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    tar_path = MODELS_DIR / f'{TIMESTAMP}.tar.gz'
    if not tar_path.exists():
        url = ('https://zenodo.org/records/15269453/files/'
               f'{TIMESTAMP}.tar.gz?download=1')
        urllib.request.urlretrieve(url, str(tar_path))
    with tarfile.open(tar_path, 'r:gz') as tf:
        tf.extractall(MODELS_DIR)
    if not (MODEL_DIR / 'ckpts' / MODEL_NAME).exists():
        candidates = [p for p in MODELS_DIR.iterdir()
                      if p.is_dir() and (p / 'ckpts' / MODEL_NAME).exists()]
        if candidates and candidates[0] != MODEL_DIR:
            candidates[0].rename(MODEL_DIR)
print(f'Model: {MODEL_DIR}')

# %% [markdown]
# ## 2. Load model via planktonclas

# %%
from planktonclas import config, paths, utils
from planktonclas.test_utils import predict
from tensorflow.keras.models import load_model

# planktonclas expects config.yaml at project root; provide a minimal one if absent
conf_path = PROJECT / 'config.yaml'
if not conf_path.exists():
    # Fall back to the copy distributed with fiesta-decrop-reproduction
    raise FileNotFoundError(
        'config.yaml missing. Copy it from fiesta-decrop-reproduction '
        'into this repo root: '
        'cp ../fiesta-decrop-reproduction/config.yaml .'
    )

config.set_config_path(str(conf_path))
paths.CONF = config.get_conf_dict()
paths.timestamp = TIMESTAMP

with open(os.path.join(paths.get_conf_dir(), 'conf.json')) as f:
    conf = json.load(f)

ckpt = os.path.join(paths.get_checkpoints_dir(), MODEL_NAME)
model = load_model(ckpt, custom_objects=utils.get_custom_objects())

DS_DIR = MODEL_DIR / 'dataset_files'
with open(DS_DIR / 'classes.txt') as f:
    class_names = [ln.strip() for ln in f if ln.strip()]
N_CLASSES = len(class_names)
print(f'Model loaded: input {model.input_shape}, output {model.output_shape}')

# %% [markdown]
# ## 3. Predict val and test splits

# %%
def load_split_paths(txt_path):
    paths_, labels = [], []
    with open(txt_path) as f:
        for line in f:
            rel, lab = line.strip().rsplit(' ', 1)
            full = IMAGES_DIR / rel
            if full.exists():
                paths_.append(str(full))
                labels.append(int(lab))
    return paths_, np.array(labels)

SPLIT_FILE = SPLIT_DIR if (SPLIT_DIR / 'val.txt').exists() else DS_DIR

for split in ['val', 'test']:
    out = RESULTS / f'cnn_predictions_{split}{"_smoke" if CI_MODE else ""}.npz'
    if out.exists():
        print(f'[{split}] cached: {out.name}')
        continue

    split_paths, split_labels = load_split_paths(SPLIT_FILE / f'{split}.txt')
    if CI_LIMIT is not None:
        split_paths = split_paths[:CI_LIMIT]
        split_labels = split_labels[:CI_LIMIT]
    print(f'\nPredicting {split} ({len(split_paths)} images, 10-crop TTA)...')

    t0 = time.time()
    pred_lab, pred_prob = predict(
        model, split_paths, conf, top_K=N_CLASSES, filemode='local',
    )
    elapsed = time.time() - t0
    print(f'  inference: {elapsed/60:.1f} min')

    N = len(split_paths)
    full_probs = np.zeros((N, N_CLASSES), dtype=np.float32)
    for i in range(N):
        full_probs[i, pred_lab[i]] = pred_prob[i]

    np.savez_compressed(
        out,
        y_true=split_labels,
        y_pred=full_probs.argmax(axis=1),
        full_probs=full_probs.astype(np.float16),
        paths=np.array(split_paths),
    )
    print(f'  saved {out.name}')

print('\nStep 02 complete.')
