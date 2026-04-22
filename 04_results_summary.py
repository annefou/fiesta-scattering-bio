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
# # Step 04 — Results summary
#
# Renders the headline tables and figures from the canonical result file
# [`results/stacking_val_trained_results.json`](results/stacking_val_trained_results.json)
# produced by a full end-to-end run of steps 01 → 02 → 03 on Decrop et al.
# 2025's held-out `test.txt` (33,718 images, 95 classes).
#
# This notebook is intentionally lightweight — it reads a committed JSON and
# reproduces the figures, so the Jupyter Book build surfaces actual numbers
# without having to execute the ~2 hour feature + CNN pipeline.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path('results') if Path('results').exists() else Path.cwd() / 'results'
with open(RESULTS / 'stacking_val_trained_results.json') as f:
    R = json.load(f)

# %% [markdown]
# ## Headline table (Decrop et al. 2025's `test.txt`, 33,718 images)

# %%
rows = [
    ('CNN alone (Decrop 2025)', R['cnn']),
    ('Scattering + LR alone',  R['scattering']),
    ('50/50 probability ensemble', R['ens_50_50']),
    ('Stacked LR (val-trained)',   R['stacked_val']),
    ('Oracle (hard-switch ceiling)', R['oracle']),
]

print(f"{'Method':<32} {'Top-1':>9} {'Top-5':>9} {'Rare recall':>13}")
print('-' * 67)
for name, m in rows:
    top5 = f"{m['top5']:>8.2%}" if 'top5' in m else f"{'—':>9}"
    print(f"{name:<32} {m['top1']:>9.2%} {top5} {m['rare_recall']:>13.2%}")

# %% [markdown]
# ### Reading the table
#
# - **CNN alone** is Decrop et al. 2025's pretrained EfficientNetV2-B0 reproduced exactly.
# - **Scattering alone** (26.9 % top-1) is a linear LR on 246-dim scattering features — an order of magnitude behind CNN because it's a 42× smaller feature space.
# - **50/50 ensemble** averages the two softmax vectors — a +2.6 pp rare-recall lift at essentially no top-1 cost.
# - **Stacked LR (val-trained)** trains a class-weighted linear meta-classifier on `val.txt` and evaluates on `test.txt` — **+8.4 pp** rare recall, **−0.72 pp** overall top-1.
# - **Oracle** is the ceiling achievable by any per-sample hard-switch between CNN and scattering — **+16.9 pp** rare recall. Stacking captures about half of this.

# %% [markdown]
# ## Per-rare-class breakdown
#
# The 13 classes with fewer than 200 training images — where CNN is most
# disadvantaged by class imbalance.

# %%
rare = R['per_rare_class']
print(f"{'class':<40} {'train':>6} {'CNN':>7} {'50/50':>8} {'Stack':>8} {'Oracle':>8}")
print('-' * 85)
rare_mean = {'cnn': 0., 'ens_50_50': 0., 'stacked_val': 0., 'oracle': 0.}
for cls, d in sorted(rare.items(), key=lambda kv: -kv[1]['cnn']):
    print(f"{cls:<40} {d.get('train_count', ''):>6} "
          f"{d['cnn']:>7.1%} {d['ens_50_50']:>8.1%} "
          f"{d['stacked_val']:>8.1%} {d['oracle']:>8.1%}")
    for k in rare_mean:
        rare_mean[k] += d[k] / len(rare)
print('-' * 85)
print(f"{'mean (13 rare classes)':<40} {'':>6} "
      f"{rare_mean['cnn']:>7.1%} {rare_mean['ens_50_50']:>8.1%} "
      f"{rare_mean['stacked_val']:>8.1%} {rare_mean['oracle']:>8.1%}")

# %% [markdown]
# ## Where does stacking beat CNN? All 95 classes

# %%
deltas = R['all_classes_delta_stack_vs_cnn']
delta_values = sorted(
    [(c, v['cnn'], v['stacked_val'], v['delta']) for c, v in deltas.items()],
    key=lambda t: -t[3],
)

n_better = sum(1 for _,_,_,d in delta_values if d > 0.01)
n_worse  = sum(1 for _,_,_,d in delta_values if d < -0.01)
n_tie    = sum(1 for _,_,_,d in delta_values if abs(d) <= 0.01)

print(f'Stacked LR vs CNN across {len(delta_values)} classes: '
      f'{n_better} better (>1 pp), {n_worse} worse (>1 pp), {n_tie} within 1 pp')

print('\nTop 10 per-class wins:')
for c, a, b, d in delta_values[:10]:
    print(f'  {c:<38} CNN={a:.2%}  Stack={b:.2%}  Δ={d:+.2%}')

print('\nBottom 5 (losses):')
for c, a, b, d in delta_values[-5:]:
    print(f'  {c:<38} CNN={a:.2%}  Stack={b:.2%}  Δ={d:+.2%}')

# %% [markdown]
# ## Figure — per-class delta (stacked − CNN)

# %%
fig, ax = plt.subplots(figsize=(12, 3.5))
deltas_pp = np.array([d for _,_,_,d in delta_values]) * 100
colors = ['tab:green' if d > 1 else ('tab:red' if d < -1 else 'lightgray')
          for d in deltas_pp]
ax.bar(range(len(deltas_pp)), deltas_pp, color=colors, width=1.0)
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlim(-1, len(deltas_pp))
ax.set_xlabel('Class (sorted by Δ, most improved left → most degraded right)')
ax.set_ylabel('Δ recall (pp): stacked − CNN')
ax.set_title(f'Stacked LR vs CNN on 95 plankton classes '
             f'({n_better} better, {n_tie} ≈ tied, {n_worse} worse)')
ax.grid(axis='y', linewidth=0.3, alpha=0.5)
plt.tight_layout()
out_fig = RESULTS / 'per_class_delta.png'
plt.savefig(out_fig, dpi=120)
plt.show()
print(f'Saved {out_fig}')

# %% [markdown]
# ## Figure — headline numbers

# %%
fig, ax = plt.subplots(figsize=(9, 4))
methods = ['CNN\nalone', 'Scattering\nalone', '50/50\nensemble',
           'Stacked LR\n(val-trained)', 'Oracle\nceiling']
top1s = [R['cnn']['top1'], R['scattering']['top1'], R['ens_50_50']['top1'],
         R['stacked_val']['top1'], R['oracle']['top1']]
rare = [R['cnn']['rare_recall'], R['scattering']['rare_recall'],
        R['ens_50_50']['rare_recall'], R['stacked_val']['rare_recall'],
        R['oracle']['rare_recall']]

x = np.arange(len(methods))
w = 0.38
ax.bar(x - w/2, [100*v for v in top1s], w,
       label='Top-1 accuracy', color='tab:blue', alpha=0.85)
ax.bar(x + w/2, [100*v for v in rare],  w,
       label='Rare-class mean recall', color='tab:orange', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(methods)
ax.set_ylabel('%')
ax.set_ylim(0, 100)
ax.set_title('Stacking lifts rare-class recall while preserving overall accuracy')
ax.legend(loc='lower right')
ax.grid(axis='y', linewidth=0.3, alpha=0.5)
for xi, t, r in zip(x, top1s, rare):
    ax.text(xi - w/2, 100*t + 1, f'{100*t:.1f}', ha='center', fontsize=8)
    ax.text(xi + w/2, 100*r + 1, f'{100*r:.1f}', ha='center', fontsize=8)
plt.tight_layout()
out_fig = RESULTS / 'headline_numbers.png'
plt.savefig(out_fig, dpi=120)
plt.show()
print(f'Saved {out_fig}')
