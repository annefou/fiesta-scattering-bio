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
# # Step 03 — Stacked CNN + scattering meta-classifier
#
# Combines the CNN probabilities from step 02 with scattering features
# from step 01 via a stacked logistic regression trained on Decrop's
# val split, then evaluated on their test split.
#
# ## Pipeline
#
# 1. Fit a per-class-balanced LR on the 246-dim scattering features of the
#    balanced training set → emits scattering softmax probabilities for val
#    and test.
# 2. Load CNN softmax probabilities on val + test (produced by step 02).
# 3. Build meta-features = concatenate `[CNN probs (95), scattering probs (95)]`
#    → 190 dimensions.
# 4. Train a class-weighted logistic regression meta-classifier on val,
#    evaluate on test.
# 5. Report top-1, top-5, rare-class mean recall, and per-class comparison
#    to CNN alone, to a 50/50 naive ensemble, and to the hard-switch oracle.

# %%
from pathlib import Path
from collections import defaultdict
import json
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, top_k_accuracy_score, precision_recall_fscore_support,
)

PROJECT = Path.cwd()
RESULTS = PROJECT / 'results'
SPLIT_DIR = PROJECT / 'splits'

# %% [markdown]
# ## 1. Load artefacts from steps 01 and 02

# %%
CI_MODE = __import__('os').environ.get('CI', '').lower() in ('true', '1')
suffix = '_smoke' if CI_MODE else ''

sc_train = np.load(RESULTS / f'features_train_balanced{suffix}.npz', allow_pickle=True)
sc_val   = np.load(RESULTS / f'features_val{suffix}.npz',            allow_pickle=True)
sc_test  = np.load(RESULTS / f'features_test{suffix}.npz',           allow_pickle=True)

cnn_val  = np.load(RESULTS / f'cnn_predictions_val{suffix}.npz',  allow_pickle=True)
cnn_test = np.load(RESULTS / f'cnn_predictions_test{suffix}.npz', allow_pickle=True)

with open(SPLIT_DIR / 'classes.txt') as f:
    class_names = [ln.strip() for ln in f if ln.strip()]
N_CLASSES = len(class_names)

train_counts = defaultdict(int)
with open(SPLIT_DIR / 'train.txt') as f:
    for line in f:
        _, lab = line.strip().rsplit(' ', 1)
        train_counts[int(lab)] += 1
rare_idx = np.array([i for i in range(N_CLASSES) if train_counts.get(i, 0) < 200])
print(f'Rare classes (train < 200): {len(rare_idx)}')

# %% [markdown]
# ## 2. Scattering LR — fit on balanced train, predict on val + test

# %%
scaler = StandardScaler().fit(sc_train['X'])
clf_sc = LogisticRegression(
    max_iter=3000, C=1.0, solver='lbfgs',
    class_weight='balanced', n_jobs=-1,
).fit(scaler.transform(sc_train['X']), sc_train['y'])

def scatter_probs(X):
    proba = np.zeros((len(X), N_CLASSES), dtype=np.float32)
    sp = clf_sc.predict_proba(scaler.transform(X))
    for j, c in enumerate(clf_sc.classes_):
        proba[:, c] = sp[:, j]
    return proba

proba_sc_val  = scatter_probs(sc_val['X'])
proba_sc_test = scatter_probs(sc_test['X'])

# %% [markdown]
# ## 3. Align CNN probabilities to the scattering test / val ordering

# %%
def align(scat_paths, cnn_paths, cnn_probs):
    p2i = {p: i for i, p in enumerate(cnn_paths)}
    aligned = np.zeros((len(scat_paths), N_CLASSES), dtype=np.float32)
    hits = 0
    for i, p in enumerate(scat_paths):
        j = p2i.get(p)
        if j is not None:
            aligned[i] = cnn_probs[j]
            hits += 1
    return aligned, hits

# CNN val keys: `val_paths` (from 02 inside this repo) or `paths` (from reproduction repo artefact)
val_key = 'val_paths' if 'val_paths' in cnn_val.files else 'paths'
test_key = 'test_paths' if 'test_paths' in cnn_test.files else 'paths'

proba_cnn_val,  hv = align(sc_val['paths'],  cnn_val[val_key],  cnn_val['full_probs'].astype(np.float32))
proba_cnn_test, ht = align(sc_test['paths'], cnn_test[test_key], cnn_test['full_probs'].astype(np.float32))
print(f'Aligned CNN val: {hv}/{len(sc_val["paths"])}  test: {ht}/{len(sc_test["paths"])}')

y_val  = sc_val['y']
y_test = sc_test['y']

# %% [markdown]
# ## 4. Stacking meta-classifier: train on val, evaluate on test

# %%
meta_val  = np.concatenate([proba_cnn_val,  proba_sc_val],  axis=1).astype(np.float32)
meta_test = np.concatenate([proba_cnn_test, proba_sc_test], axis=1).astype(np.float32)

print('Training stacking LR on val...')
meta_clf = LogisticRegression(
    max_iter=2000, C=1.0, solver='lbfgs',
    class_weight='balanced', n_jobs=-1,
).fit(meta_val, y_val)

proba_stack = np.zeros_like(proba_cnn_test)
sp = meta_clf.predict_proba(meta_test)
for j, c in enumerate(meta_clf.classes_):
    proba_stack[:, c] = sp[:, j]

# %% [markdown]
# ## 5. Evaluate: CNN vs scattering vs 50/50 ensemble vs stacked LR vs oracle

# %%
def metrics(probs):
    pred = probs.argmax(axis=1)
    top1 = accuracy_score(y_test, pred)
    top5 = top_k_accuracy_score(y_test, probs, k=5, labels=np.arange(N_CLASSES))
    _, rec, _, _ = precision_recall_fscore_support(
        y_test, pred, labels=np.arange(N_CLASSES), zero_division=0)
    return top1, top5, rec, pred

t1_cnn,  t5_cnn,  rec_cnn,  _ = metrics(proba_cnn_test)
t1_sc,   t5_sc,   rec_sc,   _ = metrics(proba_sc_test)
t1_e,    t5_e,    rec_e,    _ = metrics(0.5 * proba_cnn_test + 0.5 * proba_sc_test)
t1_s,    t5_s,    rec_s,    _ = metrics(proba_stack)

# Oracle: take whichever of CNN / scattering has the correct argmax
cnn_pred = proba_cnn_test.argmax(1); sc_pred = proba_sc_test.argmax(1)
cnn_right = cnn_pred == y_test;      sc_right = sc_pred == y_test
oracle = np.where(cnn_right, cnn_pred,
                   np.where(sc_right, sc_pred, cnn_pred))
t1_o = accuracy_score(y_test, oracle)
_, rec_o, _, _ = precision_recall_fscore_support(
    y_test, oracle, labels=np.arange(N_CLASSES), zero_division=0)

print('\n=== Final results on held-out test.txt ===')
print(f'{"method":<35} {"top-1":>8} {"top-5":>8} {"rare recall":>13}')
for name, t1, t5, rec in [
    ('CNN alone (Decrop 2025)',     t1_cnn,  t5_cnn,  rec_cnn),
    ('Scattering + LR alone',       t1_sc,   t5_sc,   rec_sc),
    ('50/50 probability ensemble',  t1_e,    t5_e,    rec_e),
    ('Stacked LR (val-trained)',    t1_s,    t5_s,    rec_s),
]:
    print(f'  {name:<33} {t1:>8.4f} {t5:>8.4f} {rec[rare_idx].mean():>13.3f}')
print(f'  {"Oracle (hard-switch ceiling)":<33} {t1_o:>8.4f}               {rec_o[rare_idx].mean():>13.3f}')

print('\n=== Rare classes — per-class recall ===')
print(f'{"class":<38} {"train":>6} {"CNN":>7} {"50/50":>7} {"Stack":>7} {"Oracle":>7}')
for i in rare_idx:
    print(f'{class_names[i]:<38} {train_counts.get(i,0):>6} '
          f'{rec_cnn[i]:>7.2%} {rec_e[i]:>7.2%} {rec_s[i]:>7.2%} {rec_o[i]:>7.2%}')

# How many of the 95 classes benefit from stacking?
deltas = rec_s - rec_cnn
n_better = int((deltas > 0.01).sum())
n_worse  = int((deltas < -0.01).sum())
print(f'\nStacked LR vs CNN across 95 classes: {n_better} better, {n_worse} worse')

# %% [markdown]
# ## 6. Save artefact for downstream consumption

# %%
out = {
    'cnn':        {'top1': float(t1_cnn), 'top5': float(t5_cnn),
                   'rare_recall': float(rec_cnn[rare_idx].mean())},
    'scattering': {'top1': float(t1_sc),  'top5': float(t5_sc),
                   'rare_recall': float(rec_sc[rare_idx].mean())},
    'ens_50_50':  {'top1': float(t1_e),   'top5': float(t5_e),
                   'rare_recall': float(rec_e[rare_idx].mean())},
    'stacked_val':{'top1': float(t1_s),   'top5': float(t5_s),
                   'rare_recall': float(rec_s[rare_idx].mean())},
    'oracle':     {'top1': float(t1_o),
                   'rare_recall': float(rec_o[rare_idx].mean())},
    'per_rare_class': {
        class_names[i]: {
            'train_count':   int(train_counts.get(i, 0)),
            'cnn':           float(rec_cnn[i]),
            'ens_50_50':     float(rec_e[i]),
            'stacked_val':   float(rec_s[i]),
            'oracle':        float(rec_o[i]),
        } for i in rare_idx
    },
    'all_classes_delta_stack_vs_cnn': {
        class_names[i]: {
            'cnn':           float(rec_cnn[i]),
            'stacked_val':   float(rec_s[i]),
            'delta':         float(rec_s[i] - rec_cnn[i]),
        } for i in range(N_CLASSES)
    },
    'n_classes_better_stacked':  n_better,
    'n_classes_worse_stacked':   n_worse,
}

path = RESULTS / f'stacking_val_trained_results{suffix}.json'
path.write_text(json.dumps(out, indent=2))
print(f'\nSaved {path}')
