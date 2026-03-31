import numpy as np
import torch


class Cutout:
    """Randomly mask out square regions in an image tensor.

    Expected input: a float tensor in shape [C, H, W].
    """

    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim != 3:
            raise ValueError(f"Cutout expects [C,H,W], got {tuple(img.shape)}")

        _, h, w = img.shape
        mask = np.ones((h, w), np.float32)

        half = self.length // 2
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = max(0, y - half)
            y2 = min(h, y + half)
            x1 = max(0, x - half)
            x2 = min(w, x + half)
            mask[y1:y2, x1:x2] = 0.0

        mask_t = torch.from_numpy(mask).to(device=img.device, dtype=img.dtype)
        mask_t = mask_t.expand_as(img[0]).unsqueeze(0)  # [1, H, W]
        return img * mask_t

