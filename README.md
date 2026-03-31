# LDP-Slicing: Local Differential Privacy for Images via Randomized Bit-Plane Slicing [CVPR'26]

<p align="center">
    <a href="https://scholar.google.com/citations?user=oMFszPAAAAAJ&hl=en">  Yuanming Cao </a> ·
    <a href="https://scholar.google.com/citations?user=lvzvu-cAAAAJ&hl=zh-CN">Chengqi Li </a> ·
    <a href="https://www.cas.mcmaster.ca/~hew11/"> Wenbo He </a>
</p>

<div align="center">
  <a href="https://arxiv.org/abs/2603.03711" target="_blank"><img src=https://img.shields.io/badge/paper-arxiv-red.svg height=22px></a>
  <a href=https://hidethedandi.github.io/codacola/LDP-Slicing/index.html target="_blank"><img src=https://img.shields.io/badge/Project-Page-bb8a2e.svg?logo=github height=22px></a>
</div>

## TL;DR
- We propose **LDP-Slicing**, a training-free image privatization pipeline with pixel-level local differential privacy.
- We apply randomized response in a **bit-plane representation** and combine it with optional perceptual obfuscation.

## Overview

> Local Differential Privacy (LDP) is a strong trust model but is often considered impractical for images due to high-dimensional pixel space. LDP-Slicing addresses this mismatch by converting pixels to bit planes and applying LDP directly at bit level, with perceptual obfuscation (DWT-based) and optimized privacy budget allocation.

<p align="center">
    <img src="assets/pipline.jpg" alt="Pipeline" width="80%" />
</p>

## Environment

```bash
pip install -r requirements.txt
```

`requirements.txt` lists **core** libs for `ldp_slicing.py` (PyTorch + `pytorch-wavelets`) plus **common** libs for training PPFR/PPIC.

## Quick Start 

### 1) Load epsilon allocation from the budget table: 
`privacy_budgets.json` (repo root by default, full derivations in supp of the main paper.  
Use **`get_privacy_budget`** to pick a **color weights** (`411` / `211` / `111`) and a **total budget** \(\varepsilon_{\mathrm{tot}}\). It returns per–bit-plane tuples for Y and chroma channels. 

```python
from ldp_slicing import get_privacy_budget

# color_weight: "411" -> 4:1:1 (default), "211" -> 2:1:1, "111" -> 1:1:1
epsilon_y, epsilon_c, total_eps, budget_key = get_privacy_budget("411", 20.0)
```


### 2) Apply privatization algorithm
```python
import torch
from ldp_slicing import dp_slicing_dwt

device = "cuda" if torch.cuda.is_available() else "cpu"
img = torch.rand(1, 3, 224, 224, device=device)  # [0,1]

x_priv = dp_slicing_dwt(    #Default ldp-slicing + LL pruning
        img,
        wavelet="haar",
        level=1,
        remove_ll=True,
        ll_scale=0.0,
        epsilon_y=epsilon_y, # You can also set custom eps values for both y and c channels.
        epsilon_c=epsilon_c,
        device=device,
)
```

### 3) Add with a PyTorch DataLoader (Recommended)
Apply LDP-Slicing with a torch dataloader:

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ldp_slicing import dp_slicing_dwt, get_privacy_budget

device = "cuda" if torch.cuda.is_available() else "cpu"

epsilon_y, epsilon_c, _, _ = get_privacy_budget("411", 20.0)
train_set = datasets.CIFAR10( # use cifar10 for example
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),  # outputs [0,1]
)
loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

model = ...  # your network
model = model.to(device)
optimizer = ... # ...

for images, labels in loader:
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    # apply dp-slicing on GPU
    with torch.no_grad():
        images_priv = dp_slicing_dwt(
            images,
            wavelet="haar",
            level=1,
            remove_ll=True,
            ll_scale=0.0,
            epsilon_y=epsilon_y,
            epsilon_c=epsilon_c,
            device=device,
        )

    # Optional: normalization after ldp-slicing
    # images_priv = normalize(images_priv)

    logits = model(images_priv)
    loss = ...

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

## Training Privacy-Preserving Image Classification/Face Recognition 

For Image Classification: `experiment/train_resnet56_ppic.py`.
For Face Recognition: `experiment/train_arcface_ppfr.py`. 

### Run the training scripts

**Image classification (CIFAR / ResNet-56):**

```bash
bash ./script/train_ppic.sh
```

**Face recognition (ArcFace / IR-50):** set `DATA_ROOT`, `FILE_LIST`, and optionally `PRETRAINED_PATH`, `CUDA_VISIBLE_DEVICES`, `WORLD_SIZE`, then:

```bash
bash ./script/train_ppfr.sh
```
Change the path of pairlist and pretrain model:
`DATA_ROOT=/your/ms1m FILE_LIST=/your/list.txt PRETRAINED_PATH=/your/IR50.pth bash ./script/train_ppfr.sh`

Notes:
- **PPIC:** checkpoints and logs under `./checkpoint/`; CIFAR data under `./data/`.
- **PPFR:** checkpoints under `./checkpoint/arcface_*`; see `experiment/train_arcface_ppfr.py` for arguments (`--file_list` is required: lines of `relative_path label`).
- For PPFR data prep, see: https://github.com/yakhyo/face-recognition  
- IR-50 pretrained backbone: https://drive.google.com/file/d/1ik8tzE9Scxhs4RJMZ6m0WW938BWX305k/view?usp=share_link


## Main Results

### Privacy-preserving face recognition:
<p align="center">
    <img src="assets/result.png" alt="result1" width="55%" />
</p>

### Privacy-preserving image classification:
<p align="center">
    <img src="assets/cifar10.png" alt="result2" width="24%" />
    <img src="assets/cifar100.png" alt="result3" width="24%" />
</p>


## Citation
If you find this project useful, please cite:

```bibtex
@inproceedings{cao2026ldpslicing,
    title={LDP-Slicing: Local Differential Privacy for Images via Randomized Bit-Plane Slicing},
    author={Yuanming Cao and Chengqi Li and Wenbo He},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2026}
}
```

## License
**Apache License 2.0**.
- Full details: [`LICENSE.txt`](LICENSE.txt)
