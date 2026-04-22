# FIESTA Scattering Bio: Plankton Classification with Scattering Transform

This repository applies the **Cross Scattering Transform** -- originally
developed for astrophysics
([Delouis et al. 2022](https://doi.org/10.1051/0004-6361/202244566),
*Astronomy & Astrophysics*) -- to classify marine plankton species from
FlowCam microscopy images.

Instead of training a deep neural network, we extract compact **scattering
texture features** with [FOSCAT](https://github.com/jmdelouis/FOSCAT) and
classify with a simple logistic regression. The scattering transform captures
multi-scale texture patterns (spines, chains, circular structures) that
distinguish plankton species -- the same mathematics that describes cosmic dust
polarisation also describes plankton morphology.

## Results

| Approach | Features | Accuracy (10 classes) |
|---|---|---|
| Raw pixels | 4,096 | **44 %** |
| Scattering S1+S2 | 42 | **61 %** |
| Combined | 4,138 | ~62 % |

With just **42 scattering features**, the method outperforms raw pixel
classification by 17 percentage points while using **100x fewer features**.

Full figures and JSON results are written to `results/`.

## Data

The plankton images come from the
[LifeWatch observatory phytoplankton training set](https://doi.org/10.5281/zenodo.10554845)
(Zenodo) -- 337,613 FlowCam images from the Belgian Part of the North Sea,
95 classes, CC-BY license. Contributed by **LifeWatch Belgium / VLIZ**.

The data (~650 MB) must be downloaded from Zenodo and extracted into `data/`
before running the notebook.

## Credits

- **Method**: Jean-Marc Delouis, LOPS/CNRS ([FOSCAT](https://github.com/jmdelouis/FOSCAT))
- **Plankton data**: LifeWatch Belgium / VLIZ
- **This application**: Anne Fouilloux, LifeWatch ERIC (FIESTA-OSCARS)

## Companion repositories

This work is part of a **cross-domain demonstration** within the
FIESTA-OSCARS project. The same scattering transform methodology is applied to
three different scientific domains:

- **Astrophysics**: [fiesta-scattering-astro](https://github.com/annefou/fiesta-scattering-astro) -- LSS map synthesis
- **Earth observation**: [fiesta-scattering-sst](https://github.com/annefou/fiesta-scattering-sst) -- SST gap-filling
- **Biodiversity / Bioimaging** (this repo) -- plankton texture classification

## Quick start

```bash
# Install dependencies
pip install foscat scikit-learn numpy matplotlib Pillow healpy jupytext

# Download data from Zenodo into data/
# https://doi.org/10.5281/zenodo.10554845

# Run the classification
python 01_plankton_classification.py
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate fiesta-scattering-bio
jupytext --to notebook 01_plankton_classification.py
jupyter execute 01_plankton_classification.ipynb
```

## Note on FOSCAT and GPU/CPU support

The [FOSCAT](https://github.com/jmdelouis/FOSCAT) package (as of v2026.2.7 on
PyPI) has several hardcoded `device='cuda'` defaults, which means it **only
works on machines with an NVIDIA GPU** out of the box. On CPU-only machines
(Apple Silicon Macs, CI runners, etc.) it will crash with a CUDA device error.

We have submitted a fix upstream:
[jmdelouis/FOSCAT#40](https://github.com/jmdelouis/FOSCAT/pull/40)
([commit](https://github.com/annefou/FOSCAT/commit/04244ed)).

Until the fix is merged and released, you can install FOSCAT from our fork:

```bash
pip install git+https://github.com/annefou/FOSCAT.git@fix/cpu-device-fallback
```

The fix is fully backwards compatible: on CUDA machines the behaviour is
identical to the original. It simply adds auto-detection so that CPU is used as
a fallback when CUDA is not available.

## FIESTA-OSCARS

FIESTA (FAIR Interoperable Experimental Scattering Transform Analysis) is part
of the [OSCARS](https://oscars-project.eu/) project, exploring reproducible
research workflows that bridge disciplines through shared mathematical methods.

## Author

**Anne Fouilloux** -- LifeWatch ERIC
ORCID [0000-0002-1784-2920](https://orcid.org/0000-0002-1784-2920)

## License

[MIT](LICENSE)
