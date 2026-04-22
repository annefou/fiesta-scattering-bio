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
# # Plankton Classification Using Scattering Transform Texture Features
#
# ## What this notebook does
#
# This notebook applies the **Cross Scattering Transform** — originally
# developed for astrophysics ([Delouis et al. 2022](https://doi.org/10.1051/0004-6361/202244566))
# — to classify marine plankton species from FlowCam microscopy images.
#
# Instead of using deep learning (which requires large labeled datasets
# and GPU training), we extract compact **scattering texture features**
# from each plankton image and classify with a simple logistic regression.
# The scattering transform captures multi-scale texture patterns
# (spines, chains, circular structures) that distinguish species.
#
# ## The cross-domain story
#
# | Domain | Data | Application | Tool |
# |--------|------|-------------|------|
# | **Astrophysics** | Planck dust maps | Denoising / synthesis | FOSCAT |
# | **Environmental sciences** | Copernicus SST | Cloud gap-filling | FOSCAT |
# | **Biodiversity / Bioimaging** | LifeWatch FlowCam plankton | Texture classification | FOSCAT |
#
# The same mathematical method, same software — applied to three
# completely different scientific domains. This is the
# [FIESTA-OSCARS](https://oscars-project.eu) cross-domain vision.
#
# ## Data
#
# [LifeWatch observatory phytoplankton training set](https://doi.org/10.5281/zenodo.10554845)
# — 337,613 FlowCam images from the Belgian Part of the North Sea,
# 95 classes, CC-BY license. Contributed by LifeWatch Belgium (VLIZ).
#
# ## Credits
#
# - **Method**: Jean-Marc Delouis, LOPS/CNRS (FOSCAT)
# - **Plankton data**: LifeWatch Belgium / VLIZ
# - **This application**: Anne Fouilloux, LifeWatch ERIC (FIESTA-OSCARS)

# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
import foscat.scat_cov as sc
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

# %% [markdown]
# ## Configuration

# %%
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

TARGET_SIZE = 64
NORIENT = 4

CI_MODE = os.environ.get("CI", "").lower() in ("true", "1")
N_PER_CLASS = 50 if CI_MODE else 300

CLASSES = [
    'Actinoptychus',
    'Asterionella',
    'Chaetoceros',
    'Ditylum_brightwellii',
    'Guinardia_flaccida',
    'Rhizosolenia',
    'Pseudo-nitzschia',
    'Bellerochea',
    'Eucampia',
    'Zygoceros',
]

# %% [markdown]
# ## 1. Download and load plankton images
#
# The dataset is downloaded from Zenodo on first run (~360 MB).
# Each image is a Region of Interest (ROI) extracted from a FlowCam
# acquisition — a single plankton organism or colony. Images vary in
# size (55–115 px); we resize all to 64×64 for consistent feature
# extraction.

# %%
import subprocess
import urllib.request

DATA_DIR = Path("data/images_DS")

if not DATA_DIR.exists():
    print("Downloading LifeWatch FlowCam plankton dataset from Zenodo...")
    archive = Path("data/phytoplankton.7z")
    archive.parent.mkdir(parents=True, exist_ok=True)

    url = "https://zenodo.org/records/10554845/files/phytoplankton_images_and_datasplit.7z?download=1"
    urllib.request.urlretrieve(url, str(archive))
    print(f"  Downloaded: {archive.stat().st_size / 1e6:.0f} MB")

    print("  Extracting...")
    subprocess.run(["7z", "x", str(archive), f"-o{archive.parent}", "-y"],
                   capture_output=True, check=True)
    print(f"  Done: {sum(1 for _ in DATA_DIR.rglob('*.jpg')):,} images")
else:
    print(f"Data already exists: {DATA_DIR}")

# %%
print(f"Loading {N_PER_CLASS} images per class from {len(CLASSES)} species...")
images = []
labels = []
class_names = []

for cls_name in CLASSES:
    cls_dir = Path(f"data/images_DS/{cls_name}")
    if not cls_dir.exists():
        print(f"  {cls_name}: NOT FOUND, skipping")
        continue
    files = list(cls_dir.iterdir())[:N_PER_CLASS]
    loaded = 0
    for f in files:
        try:
            img = Image.open(f).convert('L')
            img = img.resize((TARGET_SIZE, TARGET_SIZE))
            images.append(np.array(img, dtype=np.float32) / 255.0)
            labels.append(len(class_names))
            loaded += 1
        except:
            pass
    if loaded > 0:
        class_names.append(cls_name)
        print(f"  {cls_name}: {loaded}")

images = np.array(images)
labels = np.array(labels)
n_classes = len(class_names)
print(f"\nTotal: {len(images)} images, {n_classes} classes")

# %% [markdown]
# ## 2. Compute scattering features
#
# The scattering transform applies multi-scale wavelet filters to each
# image and computes statistics of the filter responses. This captures:
#
# - **S1**: first-order coefficients — energy at each scale and orientation
#   (captures edges, elongation, symmetry)
# - **S2**: second-order coefficients — correlations between scales
#   (captures texture complexity, hierarchical structure)
#
# Together, S1+S2 provide a compact descriptor (~42 features) that
# characterises the visual texture of each plankton organism.

# %%
print(f"Computing scattering features (NORIENT={NORIENT})...")
t0 = time.time()

scat = sc.funct(NORIENT=NORIENT, KERNELSZ=3, all_type='float32',
                silent=True, use_2D=True)
print(f"  Device: {scat.backend.device}")

features = []
for i in range(len(images)):
    feat = scat.eval(images[i].reshape(1, TARGET_SIZE, TARGET_SIZE))
    all_coeffs = []
    for attr in ['S0', 'S1', 'S2', 'P00', 'P11', 'P01']:
        if hasattr(feat, attr):
            val = scat.backend.to_numpy(getattr(feat, attr)).ravel()
            all_coeffs.append(val)
    features.append(np.concatenate(all_coeffs))
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(images)}...")

features = np.array(features)
elapsed_scat = time.time() - t0
print(f"  {features.shape[1]} features per image, computed in {elapsed_scat:.1f}s")

# %% [markdown]
# ## 3. Train and evaluate classifiers
#
# We compare three approaches, all using logistic regression:
#
# 1. **Raw pixels** (4,096 features) — baseline, no feature engineering
# 2. **Scattering S1+S2** (42 features) — texture features from FOSCAT
# 3. **Combined** — pixels + scattering together

# %%
X_train_idx, X_test_idx, y_train, y_test = train_test_split(
    range(len(images)), labels, test_size=0.3, random_state=42, stratify=labels
)

results_dict = {}

# Pixel baseline
scaler_px = StandardScaler()
X_px_tr = scaler_px.fit_transform(images[X_train_idx].reshape(len(X_train_idx), -1))
X_px_te = scaler_px.transform(images[X_test_idx].reshape(len(X_test_idx), -1))
clf_px = LogisticRegression(max_iter=2000, random_state=42, C=0.1).fit(X_px_tr, y_train)
acc_px = accuracy_score(y_test, clf_px.predict(X_px_te))
results_dict['raw_pixels'] = {'features': int(X_px_tr.shape[1]), 'accuracy': float(acc_px)}

# Scattering
scaler_sc = StandardScaler()
X_sc_tr = scaler_sc.fit_transform(features[X_train_idx])
X_sc_te = scaler_sc.transform(features[X_test_idx])
clf_sc = LogisticRegression(max_iter=2000, random_state=42, C=0.1).fit(X_sc_tr, y_train)
acc_sc = accuracy_score(y_test, clf_sc.predict(X_sc_te))
results_dict['scattering'] = {'features': int(features.shape[1]), 'accuracy': float(acc_sc)}

# Combined
X_cb_tr = np.hstack([X_px_tr, X_sc_tr])
X_cb_te = np.hstack([X_px_te, X_sc_te])
clf_cb = LogisticRegression(max_iter=2000, random_state=42, C=0.1).fit(X_cb_tr, y_train)
acc_cb = accuracy_score(y_test, clf_cb.predict(X_cb_te))
results_dict['combined'] = {'features': int(X_cb_tr.shape[1]), 'accuracy': float(acc_cb)}

print("=== RESULTS ===")
print(f"  Classes: {n_classes}")
print(f"  Random baseline:                    {1/n_classes:.1%}")
print(f"  Raw pixels ({X_px_tr.shape[1]} features):        {acc_px:.1%}")
print(f"  Scattering S1+S2 ({features.shape[1]} features):   {acc_sc:.1%}")
print(f"  Combined ({X_cb_tr.shape[1]} features):          {acc_cb:.1%}")
print(f"  Scattering advantage over pixels:   {(acc_sc - acc_px)*100:+.1f} pp")

# %% [markdown]
# ## 4. Per-class analysis

# %%
print("\nPer-class accuracy (scattering features):")
y_pred = clf_sc.predict(X_sc_te)
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
for cls_name in class_names:
    if cls_name in report:
        r = report[cls_name]
        print(f"  {cls_name:30s}: {r['f1-score']:.1%} F1 ({r['support']:.0f} samples)")

# %% [markdown]
# ## 5. Save results

# %%
results = {
    "method": "Cross Scattering Transform (FOSCAT) + Logistic Regression",
    "original_paper": "Delouis et al. 2022, A&A 668, A122",
    "original_paper_doi": "10.1051/0004-6361/202244566",
    "dataset": "LifeWatch FlowCam phytoplankton (Zenodo 10.5281/zenodo.10554845)",
    "dataset_contributor": "LifeWatch Belgium / VLIZ",
    "n_classes": n_classes,
    "classes": class_names,
    "n_per_class": N_PER_CLASS,
    "image_size": TARGET_SIZE,
    "norient": NORIENT,
    "device": str(scat.backend.device),
    "scattering_time_seconds": elapsed_scat,
    "results": results_dict,
    "random_baseline": 1 / n_classes,
}

with open(RESULTS / "plankton_classification_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {RESULTS / 'plankton_classification_results.json'}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel A: accuracy comparison
ax = axes[0]
methods = ['Raw pixels\n(4096)', f'Scattering\n({features.shape[1]})', f'Combined\n({X_cb_tr.shape[1]})']
accs = [acc_px, acc_sc, acc_cb]
colors = ['lightcoral', 'steelblue', 'mediumpurple']
bars = ax.bar(methods, accs, color=colors)
ax.axhline(1/n_classes, color='gray', linestyle='--', label=f'Random ({1/n_classes:.0%})')
ax.set_ylabel('Test accuracy')
ax.set_title('Classification accuracy')
ax.set_ylim(0, 1)
ax.legend()
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{acc:.1%}', ha='center', fontweight='bold')

# Panel B: per-class F1 scores
ax = axes[1]
f1_scores = [report[cls]['f1-score'] for cls in class_names if cls in report]
short_names = [c.split('_')[0][:12] for c in class_names]
ax.barh(short_names, f1_scores, color='steelblue')
ax.set_xlabel('F1 score')
ax.set_title('Per-class F1 (scattering features)')
ax.set_xlim(0, 1)

# Panel C: sample images
ax = axes[2]
n_show = min(5, n_classes)
for i in range(n_show):
    idx = np.where(labels == i)[0][0]
    ax_img = fig.add_axes([0.68 + (i % 5) * 0.06, 0.55 - (i // 5) * 0.35, 0.05, 0.15])
    ax_img.imshow(images[idx], cmap='gray')
    ax_img.set_title(class_names[i].split('_')[0][:8], fontsize=7)
    ax_img.axis('off')
ax.axis('off')
ax.set_title('Sample plankton images')

fig.suptitle('Plankton Classification with Scattering Transform (FOSCAT)',
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(RESULTS / 'plankton_classification.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {RESULTS / 'plankton_classification.png'}")

# %% [markdown]
# ## 6. What does this mean?
#
# The scattering transform — originally developed for characterising
# Planck dust polarisation maps — also captures discriminative texture
# features in microscopy images of marine plankton. With just **42
# scattering features** and a simple logistic regression, we achieve
# **61% accuracy** on 10 plankton species — outperforming raw pixel
# classification (44%) by 17 percentage points while using **100× fewer
# features**.
#
# This demonstrates that the mathematical framework of scattering
# transforms is truly **domain-agnostic**: the same texture statistics
# that describe cosmic dust structure also describe plankton morphology.
#
# ## Limitations
#
# - Only RGB converted to grayscale — colour information discarded
# - 64×64 resize loses fine morphological detail
# - Only 10 of 95 available classes tested
# - Simple logistic regression — a non-linear classifier could do better
# - No comparison with CNN baseline (which would likely outperform)
#
# The point is not to beat deep learning — it's to show that the
# **same compact features** work across astrophysics, oceanography,
# and marine biology.
#
# ## Companion projects
#
# - **Astrophysics**: [fiesta-scattering-astro](https://github.com/annefou/fiesta-scattering-astro)
# - **Environmental sciences**: [fiesta-scattering-sst](https://github.com/annefou/fiesta-scattering-sst)
#
# ## Replication context
#
# Part of [FIESTA-OSCARS](https://oscars-project.eu) cross-domain
# FAIR image analysis. Published as FORRT nanopublications on
# [Science Live](https://platform.sciencelive4all.org).
