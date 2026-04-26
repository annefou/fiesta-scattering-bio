# FIESTA Scattering Bio — scattering transforms complement a plankton CNN

[![Source DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19687112.svg)](https://doi.org/10.5281/zenodo.19687112)
[![Docker image DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19701138.svg)](https://doi.org/10.5281/zenodo.19701138)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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

## FORRT nanopublication chain

The full provenance of this work is recorded as a six-step FORRT
nanopublication chain on the
[Science Live](https://platform.sciencelive4all.org) platform. Each step is
independently citable and machine-readable; together they form the FAIR
provenance receipt for this study.

> **Headline assertion — machine-readable:**
> [**This work `cito:extends` Decrop et al. 2025, `cito:usesMethodIn` Delouis et al. 2022, AND `cito:citesAsDataSource` `fiesta-decrop-reproduction`**](https://w3id.org/sciencelive/np/RA8BSfq4Cbs3A4chU6fvafk0px-yDZwsYKFIO9cnzkyx4)
>
> The CiTO citation nanopublication encodes three relationships at once:
> this work extends Decrop et al. 2025's CNN classifier with stacking
> (`cito:extends`); the underlying multi-scale scattering-feature method
> comes from Delouis et al. 2022 (`cito:usesMethodIn`); and the
> [`fiesta-decrop-reproduction`](https://github.com/annefou/fiesta-decrop-reproduction)
> repository is cited as the data source for the CNN val and test
> softmax probabilities our stacking consumes (`cito:citesAsDataSource`).
> Discovery tools (Scholia, Wikidata pipelines, SPARQL endpoints) can
> follow this single citation to find all three relationships.

The five preceding nanopubs build the provenance ladder up to that citation:

| Step | Type | Asserts | Nanopub URI |
|---|---|---|---|
| 1 | Quote-with-comment (Annotate a paper quotation) | Verbatim quote of Decrop et al. 2025's acknowledgement of the rare-species class-imbalance limitation, with personal comment framing the extension | [`RAH71…`](https://w3id.org/sciencelive/np/RAH71F0FaKHQ2OZ-k_mGaO0GB501kDHo_ZqY6ScBbtep4) |
| 2 | AIDA sentence *(Nanodash namespace)* | Atomic, declarative restatement: stacking scattering features on a CNN's softmax probabilities lifts mean rare-class recall by 8.4 percentage points at 0.72-percentage-point top-1 cost. *(Published via Nanodash because of the Science Live AIDA-form bug with combined datasets+publications fields.)* | [`RAT-P…`](https://w3id.org/np/RAT-PGqwhe4Y2hALiOhgEm4ErYVK9vfk0YbQpV2qdN1Ck) |
| 3 | FORRT Claim (model performance) | The stacking-improvement claim, typed as a FORRT model-performance claim | [`RAeI2…`](https://w3id.org/sciencelive/np/RAeI2cspabQBUMHiA94jOKjiBypsIFSPPDwH_f5l1sJu4) |
| 4 | FORRT Replication Study | Replication with different methodology — CNN + scattering meta-features + class-weighted LR stacking, on Decrop's exact `val.txt` and `test.txt` splits | [`RAauQ…`](https://w3id.org/sciencelive/np/RAauQR3eY4NGILF2ei5a6YLbZjT7JDK_tQGYyl2R4DcMk) |
| 5 | FORRT Replication Outcome (Validated, High) | Stacked LR top-1 85.62%, top-5 95.39%, mean rare-class recall 56.08% — vs CNN alone 86.34% / 98.70% / 47.70%; 33 of 95 classes improve, 16 worsen | [`RAIqy…`](https://w3id.org/sciencelive/np/RAIqyPwP8PzMKSKCMKb04vzR1CpttIsAYqIRy6RkaBPZk) |
| 6 | **CiTO citation — `cito:extends` Decrop 2025 + `cito:usesMethodIn` Delouis 2022 + `cito:citesAsDataSource` `fiesta-decrop-reproduction`** | The headline triple assertion above | [**`RA8BS…`**](https://w3id.org/sciencelive/np/RA8BSfq4Cbs3A4chU6fvafk0px-yDZwsYKFIO9cnzkyx4) |

The chain runs: paper → quote → atomic claim → FORRT claim → study (this
repo) → outcome (the metrics in the Headline numbers table) → CiTO
citations to the three relationships above.

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

## Container image

A Docker container for this repository is built on every release, pushed to
GitHub Container Registry, and archived to Zenodo.

```bash
docker pull ghcr.io/annefou/fiesta-scattering-bio:latest
docker run --rm -v "$PWD/results:/app/results" \
    ghcr.io/annefou/fiesta-scattering-bio:latest
```

Zenodo-archived image tarballs via the
[Docker image concept DOI 10.5281/zenodo.19701138](https://doi.org/10.5281/zenodo.19701138).

**Note:** the container runs step 01 (scattering features) by default. Step 02
(CNN inference) needs a separate TF-based image — reuse
[fiesta-decrop-reproduction](https://github.com/annefou/fiesta-decrop-reproduction)'s
container for that.

## How to cite

If you use this repository, please cite it via its Zenodo DOI together
with the CNN baseline paper (Decrop et al. 2025) and the scattering-transform
method paper (Delouis et al. 2022).

```
Fouilloux, A. (2026). FIESTA Scattering Bio — scattering transforms
complement a plankton CNN (v0.3.0). Zenodo.
https://doi.org/10.5281/zenodo.19687112
```

BibTeX:

```bibtex
@software{fouilloux_fiesta_scattering_bio,
  author    = {Fouilloux, Anne},
  title     = {FIESTA Scattering Bio — scattering transforms complement a plankton CNN},
  year      = {2026},
  version   = {0.3.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19687112},
  url       = {https://doi.org/10.5281/zenodo.19687112}
}
```

The DOI above is the **concept DOI** — it always resolves to the latest
release. Specific version DOIs are available on the
[Zenodo record page](https://doi.org/10.5281/zenodo.19687112).

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata
and the full reference list (Decrop et al. 2025, Delouis et al. 2022,
LifeWatch dataset, pretrained model).

## Credits

- **Method** — Jean-Marc Delouis, LOPS/CNRS ([FOSCAT](https://github.com/jmdelouis/FOSCAT)); Allys et al., Boulanger et al. on scattering transforms for cosmology.
- **CNN baseline and dataset** — Decrop et al. 2025 ([Frontiers in Marine Science](https://doi.org/10.3389/fmars.2025.1699781)), LifeWatch Belgium / VLIZ.
- **This work** — Anne Fouilloux, LifeWatch ERIC (FIESTA-OSCARS).

## License

Code MIT. Data and pretrained model CC-BY 4.0 — cite Decrop et al. 2025 and
the Zenodo dataset + model records when re-using.
