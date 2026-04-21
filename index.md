# Scattering Transform: Plankton Classification

## Why plankton monitoring matters

Phytoplankton form the base of the marine food web and produce roughly half of
Earth's oxygen. Monitoring their species composition is essential for
understanding marine biodiversity, detecting harmful algal blooms, and tracking
the effects of climate change on ocean ecosystems.

Automated instruments like the **FlowCam** generate hundreds of thousands of
microscopy images per survey -- far too many for manual identification.
Automated classification is essential, but training deep learning models
requires large labeled datasets and significant compute resources.

## Scattering transform: texture features without deep learning

The **scattering transform** is a wavelet-based method that extracts
multi-scale texture descriptors from images. Unlike a CNN, it requires no
training: the wavelet filters are fixed, and the resulting coefficients are
provably stable to small deformations.

Originally developed for characterising non-Gaussian structure in astrophysical
maps ([Delouis et al. 2022](https://doi.org/10.1051/0004-6361/202244566)), we
show here that the same coefficients capture discriminative morphological
features in plankton microscopy images -- spines, chains, circular structures,
and surface textures that distinguish species.

## The FIESTA cross-domain story

This repository is part of a **cross-domain demonstration** within the
FIESTA-OSCARS project. We apply the same scattering transform methodology --
the same mathematics, the same software
([FOSCAT](https://github.com/jmdelouis/FOSCAT)) -- to three completely
different scientific domains:

| Domain | Repository | Application |
|---|---|---|
| **Astrophysics** | [fiesta-scattering-astro](https://github.com/annefou/fiesta-scattering-astro) | LSS map synthesis |
| **Earth observation** | [fiesta-scattering-sst](https://github.com/annefou/fiesta-scattering-sst) | SST gap-filling |
| **Biodiversity / Bioimaging** | this repo | Plankton texture classification |

The mathematical framework is identical; only the input data and physical
interpretation change. This transferability is a concrete example of how FAIR,
reproducible research workflows can bridge scientific disciplines.

## Results summary

| Approach | Features | Accuracy (10 classes) |
|---|---|---|
| Raw pixels | 4,096 | **44 %** |
| Scattering S1+S2 | 42 | **61 %** |
| Combined | 4,138 | ~62 % |

With just **42 scattering features** and a simple logistic regression, the
method outperforms raw pixel classification by 17 percentage points while
using **100x fewer features**.

## Data source

Plankton images are from the
[LifeWatch observatory phytoplankton training set](https://doi.org/10.5281/zenodo.10554845)
(Zenodo) -- 337,613 FlowCam images from the Belgian Part of the North Sea,
95 classes, CC-BY license. Contributed by LifeWatch Belgium / VLIZ.

## Companion repositories

- [fiesta-scattering-astro](https://github.com/annefou/fiesta-scattering-astro) -- Astrophysics map synthesis
- [fiesta-scattering-sst](https://github.com/annefou/fiesta-scattering-sst) -- Sea Surface Temperature gap-filling
- [FOSCAT](https://github.com/jmdelouis/FOSCAT) -- the scattering transform library by Jean-Marc Delouis
