# FIESTA Scattering Bio — scattering transforms complement a plankton CNN

This repository shows that the **Cross Scattering Transform** — originally
developed for astrophysics
([Delouis et al. 2022](https://doi.org/10.1051/0004-6361/202244566)) — carries
information **complementary** to a fine-tuned ImageNet CNN on plankton image
classification.

A stacked logistic-regression meta-classifier trained on Decrop et al. 2025's
open `val.txt` split and evaluated on their held-out `test.txt` improves
per-class recall on **33 of 95 plankton taxa**, lifts rare-class mean recall
by **+8.4 pp** (47.7 % → 56.1 %), and costs only **0.72 pp in overall top-1
accuracy** (86.34 % → 85.62 %). The gain captures about half of the oracle
ceiling.

The method baseline we compare against is [Decrop et al. 2025](https://doi.org/10.3389/fmars.2025.1699781)'s
EfficientNetV2-B0. We re-run their pretrained model on their own held-out
test split and reproduce their published numbers exactly
(86.3426 % top-1, paper: 86.34 %) — see
[fiesta-decrop-reproduction](https://github.com/annefou/fiesta-decrop-reproduction)
for the standalone reproduction.

## Headline numbers

| Method | Top-1 | Top-5 | Rare-class mean recall |
|---|---:|---:|---:|
| CNN alone (Decrop et al. 2025) | 86.34 % | 98.70 % | 47.7 % |
| Scattering alone + LR | 26.93 % | 60.43 % | 43.0 % |
| 50/50 probability ensemble | 86.28 % | 94.82 % | 50.3 % |
| **Stacked LR (val-trained)** | **85.62 %** | **95.39 %** | **56.1 %** |
| Oracle ceiling (hard switch) | 87.68 % | — | 64.6 % |

Full per-class details in [`results/stacking_val_trained_results.json`](results/stacking_val_trained_results.json).

## Why this matters scientifically

CNNs for microscopy classification are biased by training frequency: they
underperform on rare taxa. Scattering coefficients are deterministic wavelet
statistics with **no learned parameters**, so they do not inherit this bias.
The two predictors err on different samples — scattering alone is uniquely
correct on **1.3 %** of test images — and those disagreements concentrate on
rare classes. A small linear stacking classifier captures a large share of
this complementarity.

The same scattering transform is used in three other FIESTA-OSCARS
repositories (astrophysical map synthesis, SST gap-filling, and SST on a
WGS84 ellipsoid). This bio repository is the one domain demonstrating
scattering as a **complementary layer** on top of an existing deep model —
the pattern for deploying scattering alongside deep learning in any field
where a pretrained model already exists but has calibration gaps on rare
categories.

## Pipeline

Three numbered steps:

| Step | Script | Env | Approx CPU time | Output |
|---|---|---|---|---|
| 01 | `01_scattering_features.py` | `environment.yml` (FOSCAT) | ~60 min for all three splits | `results/features_*.npz` |
| 02 | `02_cnn_predictions.py` | see below | ~56 min for val + test | `results/cnn_predictions_*.npz` |
| 03 | `03_stacking.py` | `environment.yml` | < 2 min | `results/stacking_val_trained_results.json` |

### Step 02 — CNN baseline

Step 02 reuses [Decrop et al. 2025's pretrained EfficientNetV2-B0](https://doi.org/10.5281/zenodo.15269453)
via the [`planktonclas`](https://github.com/lifewatch/planktonclas) toolkit.
This needs TensorFlow 2.19 + planktonclas, which conflict with the FOSCAT
PyTorch stack used by steps 01 and 03. Two options:

1. **Use the reproduction repo's env.** [fiesta-decrop-reproduction](https://github.com/annefou/fiesta-decrop-reproduction)
   already publishes exactly this environment — clone it and run
   `02_cnn_predictions.py` from the reproduction repo's venv, then copy the
   predictions (`cnn_predictions_val.npz`, `cnn_predictions_test.npz`) into
   this repo's `results/` directory. This is the recommended path.
2. **Create a dedicated CNN environment locally.** If you prefer to stay in
   one repository:
   ```bash
   python3.11 -m venv cnn-venv
   cnn-venv/bin/pip install tensorflow==2.19.0 "planktonclas==0.2.0" zenodo-get pillow scikit-learn pyyaml
   cnn-venv/bin/python 02_cnn_predictions.py
   ```

Either way, steps 01 and 03 run in this repo's main `environment.yml`.

## How to reproduce end-to-end

```bash
# 1. Scattering features (train-balanced, val, test) — ~60 min CPU
micromamba create -f environment.yml
micromamba activate fiesta-scattering-bio
python 01_scattering_features.py        # downloads 650 MB dataset on first run

# 2. CNN predictions on val + test — ~56 min CPU; see "Step 02" above
#    (run via fiesta-decrop-reproduction's env, copy .npz files into results/)

# 3. Stacking — < 2 min
python 03_stacking.py                    # outputs stacking_val_trained_results.json
```

Or with Snakemake:

```bash
snakemake --cores all
```

## Data and model provenance

- **Images**: [LifeWatch FlowCam phytoplankton training set](https://doi.org/10.5281/zenodo.10554845)
  (VLIZ / LifeWatch Belgium, CC-BY 4.0). 337,567 images, 95 classes.
  Used with Decrop et al. 2025's own `train.txt` / `val.txt` / `test.txt`
  split files (distributed with the pretrained model, Zenodo 15269453).
- **Pretrained model**: [Phytoplankton_EfficientNetV2B0](https://doi.org/10.5281/zenodo.15269453)
  (CC-BY 4.0).
- **FOSCAT**: [github.com/jmdelouis/FOSCAT](https://github.com/jmdelouis/FOSCAT),
  CPU-compatible fork [annefou/FOSCAT @ v0.1.0-cpu](https://github.com/annefou/FOSCAT)
  (upstream PR [#40](https://github.com/jmdelouis/FOSCAT/pull/40)).

## FIESTA-OSCARS cross-domain context

This repository is part of a four-domain demonstration of scattering
transforms built on FOSCAT. The same mathematical framework and software
core are used across:

| Domain | Repository | What scattering does |
|---|---|---|
| Astrophysics | [fiesta-scattering-astro](https://github.com/annefou/fiesta-scattering-astro) | LSS map synthesis (map generation matching scattering statistics) |
| Environmental sciences | [fiesta-scattering-sst](https://github.com/annefou/fiesta-scattering-sst) | SST gap-filling on HEALPix sphere |
| Environmental sciences | [fiesta-scattering-sst-healpix-geo](https://github.com/annefou/fiesta-scattering-sst-healpix-geo) | SST gap-filling on WGS84 ellipsoid |
| Biodiversity / Bioimaging | **this repo** | Complementary features to CNN on plankton classification |
| CNN reproduction baseline | [fiesta-decrop-reproduction](https://github.com/annefou/fiesta-decrop-reproduction) | Independent reproduction of Decrop et al. 2025 |

The natural **Galaxy tool factoring** emerges from this landscape:

- `foscat-synthesis` — scattering-based gap-filling / map synthesis (3 repos)
- `foscat-features` — feature extraction (this repo)
- `scattering-stacking` — CNN + scattering meta-classifier (this repo)

All three tools share the same FOSCAT core and are composable across domains.

## Credits

- **Method** — Jean-Marc Delouis, LOPS/CNRS ([FOSCAT](https://github.com/jmdelouis/FOSCAT)); Allys et al., Boulanger et al. on scattering transforms for cosmology.
- **CNN baseline and dataset** — Decrop et al. 2025 ([Frontiers in Marine Science](https://doi.org/10.3389/fmars.2025.1699781)), LifeWatch Belgium / VLIZ.
- **This work** — Anne Fouilloux, LifeWatch ERIC (FIESTA-OSCARS).

## License

Code MIT. Data and pretrained model CC-BY 4.0 — cite Decrop et al. 2025 and
the Zenodo dataset + model records when re-using.
