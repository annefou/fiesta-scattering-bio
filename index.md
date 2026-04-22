# FIESTA Scattering Bio — scattering transforms complement a plankton CNN

## Why this matters

Phytoplankton form the base of the marine food web and produce roughly half of
Earth's oxygen. The **FlowCam** imaging system generates hundreds of thousands
of microscopy images per monitoring survey, too many for manual identification.
Automated classifiers are essential, but deep models trained on class-imbalanced
corpora systematically underperform on **rare taxa** — exactly the classes
biodiversity monitoring most needs to see.

This repository demonstrates that the **Cross Scattering Transform** — a
deterministic, wavelet-based feature extractor with no learned parameters —
carries information **complementary** to a fine-tuned ImageNet CNN on this
problem, and that a small linear meta-classifier can exploit this
complementarity.

## Headline result

Using Decrop et al. 2025's pretrained EfficientNetV2-B0 as the CNN baseline and
stacking with 246-dim RGB scattering features:

| Method | Top-1 | Top-5 | Rare-class mean recall |
|---|---:|---:|---:|
| CNN alone | 86.34 % | 98.70 % | 47.7 % |
| 50/50 probability ensemble | 86.28 % | 94.82 % | 50.3 % |
| **Stacked LR (val-trained)** | **85.62 %** | **95.39 %** | **56.1 %** |
| Oracle ceiling | 87.68 % | — | 64.6 % |

**33 of 95 classes improve by >1 pp** under stacking; only 16 degrade. The
+8.4 pp lift on rare-class recall captures about half of the oracle ceiling,
at a cost of 0.72 pp in overall top-1 accuracy.

## Why it works

CNNs are biased by training frequency: a class with 100 training images gets
orders of magnitude less gradient signal than one with 10,000. Scattering
coefficients are **deterministic wavelet statistics** — they do not learn from
frequency at all.

Coverage analysis on the 33,718-image test set:

| Outcome | % of images |
|---|---:|
| Both CNN and scattering correct | 25.6 % |
| CNN only correct | 60.8 % |
| **Scattering only correct** | **1.3 %** |
| Neither correct | 12.3 % |

The 1.3 % is small overall but **heavily enriched in rare taxa**, which is why
stacking lifts rare-class recall specifically while barely touching overall
top-1 accuracy. Confidence-based gating *cannot* capture this gain: CNN is
pathologically overconfident (median max-softmax = 1.000, and 45.7 % of its
errors happen at max-prob ≥ 0.9), so CNN confidence does not correlate with
correctness.

## The FIESTA-OSCARS cross-domain story

This repository is part of a **four-domain demonstration** of scattering
transforms built on [FOSCAT](https://github.com/jmdelouis/FOSCAT):

| Domain | Repository | Role of scattering |
|---|---|---|
| Astrophysics | [fiesta-scattering-astro](https://github.com/annefou/fiesta-scattering-astro) | LSS map synthesis |
| Environmental sciences | [fiesta-scattering-sst](https://github.com/annefou/fiesta-scattering-sst) | SST gap-filling (sphere) |
| Environmental sciences | [fiesta-scattering-sst-healpix-geo](https://github.com/annefou/fiesta-scattering-sst-healpix-geo) | SST gap-filling (WGS84) |
| **Biodiversity** | **this repo** | **Complementary features to a deep classifier** |

Plus a dedicated CNN reproduction repository, which is where the baseline we
stack onto comes from:

| Repository | Role |
|---|---|
| [fiesta-decrop-reproduction](https://github.com/annefou/fiesta-decrop-reproduction) | Independent reproduction of Decrop et al. 2025 + CNN prediction artefacts consumed here |

The same FOSCAT core supports three reusable Galaxy tools:

- `foscat-synthesis` — gap-filling / map synthesis (astro, sst, sst-healpix-geo)
- `foscat-features` — feature extraction (this repo)
- `scattering-stacking` — meta-classifier composing CNN + scattering (this repo)

## Pipeline

See the [README](https://github.com/annefou/fiesta-scattering-bio/blob/main/README.md)
for full run instructions. Three steps:

1. **`01_scattering_features.py`** — extract 246-dim RGB scattering features
   on balanced-train, val, and test splits. (~60 min CPU for 77k images.)
2. **`02_cnn_predictions.py`** — run Decrop et al.'s pretrained EfficientNetV2-B0
   on val and test. (~56 min CPU. Needs a separate TF/planktonclas environment —
   recommended: reuse artefacts from `fiesta-decrop-reproduction`.)
3. **`03_stacking.py`** — fit scattering LR, build meta-features, train stacking
   LR on val, evaluate on test. (< 2 min.)

## Data and model provenance

- **Images**: [LifeWatch FlowCam phytoplankton training set](https://doi.org/10.5281/zenodo.10554845)
  (VLIZ / LifeWatch Belgium, CC-BY 4.0).
- **Splits**: Decrop et al. 2025's own `train.txt` / `val.txt` / `test.txt` as
  distributed with their [pretrained model](https://doi.org/10.5281/zenodo.15269453) (CC-BY 4.0).
- **FOSCAT**: [github.com/jmdelouis/FOSCAT](https://github.com/jmdelouis/FOSCAT),
  CPU-compatible fork at [annefou/FOSCAT@v0.1.0-cpu](https://github.com/annefou/FOSCAT).

## Credits

- Method — Jean-Marc Delouis, LOPS/CNRS (FOSCAT); scattering transforms for cosmology and Earth observation.
- CNN baseline + dataset — Decrop et al. 2025 ([Frontiers in Marine Science](https://doi.org/10.3389/fmars.2025.1699781)), LifeWatch Belgium / VLIZ.
- This work — Anne Fouilloux, LifeWatch ERIC (FIESTA-OSCARS).
