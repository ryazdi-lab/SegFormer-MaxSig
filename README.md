# Lightweight Order-Statistic Gate for Transformer-Based Breast Ultrasound Lesion Segmentation

Shayan Ghasembeiky, Soheila Karbasi, Reza Yazdi
Department of Computer Engineering, Golestan University, Gorgan, Iran

## Overview

MaxSig is a small order-statistic layer that contrasts the centre and median of each local patch against its mean. It was originally validated on small convolutional networks. This work tests it on a self-attention backbone: a zero-initialised **MaxSig Gate** is inserted into SegFormer's Mix Transformer (MiT) encoder after stages 1 and 2.

```
y_out = x + γ · Post(MaxSig(x)),    γ initialised to 0
```

Here `x` is the feature map output by an encoder stage, and `Post` is BatchNorm followed by LeakyReLU. The gate starts as an identity map, so the network learns how much MaxSig signal to use.

## Setup

- **Backbones:** SegFormer MiT-B0 to B5 (3.7M to 84.6M parameters), re-implemented from scratch and trained from random initialisation
- **Datasets (7):** BUS-UCLM, BUS-Synth, BUSI-WHU, BUSI, BUSBRA, BrEaST-USG, BLUI
- **Comparisons:** 42 matched baseline vs. +MaxSig runs (6 backbones × 7 datasets)
- **Training:** AdamW, lr 1e-4, batch size 8, 100 epochs, 256×256 inputs, no augmentation, seed 42

## Results

| Metric (mean over 42 runs) | Change with MaxSig | Runs improved |
|---|---|---|
| Test Dice | +1.17 points | 21 / 42 |
| Test IoU | +1.36 points | 21 / 42 |
| HD95 | −2.26 px (better) | 27 / 42 |

- Parameter overhead is **0.81%** at B0 and falls to **0.14%** at B5.
- Five of seven datasets show a positive average Dice change.
- The median Dice change is only +0.03, and two runs swing by more than 17 points in opposite directions. We attribute these to single-seed instability in from-scratch transformer training. **Multi-seed evaluation is needed before treating MaxSig as a drop-in improvement.**

## Code

Code and full per-run results are provided.

## Citation

```bibtex
@inproceedings{yazdi2026maxsig,
  title  = {Lightweight Order-Statistic Gate for Transformer-Based Breast Ultrasound Lesion Segmentation},
  author = {Ghasembeiky, Shayan and Karbasi, Soheila and Yazdi, Reza},
  booktitle={The 33rd National and the 11th International Conference on Biomedical Engineering of Iran},
  year   = {2026}
}
```

## Contact

Reza Yazdi: r.yazdi@gu.ac.ir
