# LDP-Slicing: Local Differential Privacy for Images via Randomized Bit-Plane Slicing

<p align="center">
    <a href="https://scholar.google.com/citations?user=oMFszPAAAAAJ&hl=en"> <b>Yuanming Cao</b> </a> ·
    <a href="https://scholar.google.com/citations?user=lvzvu-cAAAAJ&hl=zh-CN"> <b>Chengqi Li</b> </a> ·
    <a href="https://www.cas.mcmaster.ca/~hew11/"> <b> Wenbo He </b> </a>
</p>
<p align="center">
    <b>CVPR 2026</b><br/>
    <a href="https://arxiv.org/abs/2603.03711"> <img src=https://img.shields.io/badge/paper-arxiv-red.svg height=22px> </a> 
    <a href="https://github.com/HideTheDandi/LDP-Slicing"> <img src=https://img.shields.io/badge/GitHub-Repository-181717.svg?logo=github height=22px></a> 
    <a href="https://hidethedandi.github.io/coda_webpage/LDP-Slicing/index.html"> <img src= https://img.shields.io/badge/Project-Page-bb8a2e.svg?logo=github height=22px></a> 

</p>

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

## Quick Start 

### 1) Load epsilon allocation from table (Optional)
```python
from ldp_slicing import get_epsilon_value

epsilon_y, epsilon_c = get_epsilon_value(20.0) #Load eps value at startup to avoid bottleneck
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
from ldp_slicing import dp_slicing_dwt, get_epsilon_value

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load once, avoid bottleneck
epsilon_y, epsilon_c = get_epsilon_value(20.0)
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


## Privacy Budget Table

The precomputed budget schedules are provided in `privacy_budgets.json` for total budgets (Full derivation is in Appendix of the main paper): 

```
1.0, 2.4, 5.2, 12.0, 20.0, 32.0, 58.0
```

Each entry contains:
- `epsilon_y`: 8-value tuple for Y channel bit-planes
- `epsilon_c`: 8-value tuple for Cb/Cr channel bit-planes

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
