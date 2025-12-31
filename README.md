# RawRefinery

[![PyPI version](https://img.shields.io/pypi/v/rawforge.svg)](https://pypi.org/project/rawforge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python version](https://img.shields.io/pypi/pyversions/rawforge.svg)](https://pypi.org/project/rawforge/)

**RawForge** is an open-source command line application for **raw image quality refinement and denoising**.

Currently in **alpha release**.

---

## Install from pip:
```bash
pip install rawforge
```

---
## Example command line syntax:

```bash
rawforge TreeNetDenoiseHeavy test.CR2 test_heavy.dng --cfa 
```


```bash 
usage: rawforge [-h] [--conditioning CONDITIONING] [--dims x0 x1 y0 y1] [--cfa] [--device DEVICE] [--disable_tqdm] [--tile_size TILE_SIZE] [--lumi LUMI] [--chroma CHROMA] model in_file out_file

A command line utility for processing raw images.

positional arguments:
  model                 The name of the model to use.
  in_file               The name of the file to open.
  out_file              The name of the file to save.

options:
  -h, --help            show this help message and exit
  --conditioning CONDITIONING
                        Conditioning array to feed model.
  --dims x0 x1 y0 y1    Optional crop dimensions.
  --cfa                 Save the image as a CFA image (default: False).
  --device DEVICE       Set device backend (cuda, cpu, mps).
  --disable_tqdm        Disable the progress bar.
  --tile_size TILE_SIZE
                        Set tile size. (default: 256)
  --lumi LUMI           Lumi noise (0-1).
  --chroma CHROMA       Chroma noise (0-1).
```

----

## Acknowledgments

With thanks to:


> Brummer, Benoit; De Vleeschouwer, Christophe. (2025).
> *Raw Natural Image Noise Dataset.*
> [https://doi.org/10.14428/DVN/DEQCIM](https://doi.org/10.14428/DVN/DEQCIM), Open Data @ UCLouvain, V1.

> Chen, Liangyu; Chu, Xiaojie; Zhang, Xiangyu; Chen, Jianhao. (2022).
> *NAFNet: Simple Baselines for Image Restoration.*
> [https://doi.org/10.48550/arXiv.2208.04677](https://doi.org/10.48550/arXiv.2208.04677), arXiv, V1.
