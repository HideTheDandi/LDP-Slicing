
# DWT + LDP-Slicing:
# - Inputs are expected in [N, C, H,W] format.
# - dp_slicing_dwt returns float images in [0, 1].

import json
from pathlib import Path
import torch
from pytorch_wavelets import DWTForward, DWTInverse


def to_ycbcr(x: torch.Tensor, max_value: float = 255.0) -> torch.Tensor:
   
    # x: Input tensor with shape [N, 3, H, W]. Values are typically in [0, max_value].
    # max_value: Dynamic range maximum (255 for 8-bit full range).

    # reuturs YCbCr tensor with shape [N, 3, H, W].
    if x.ndim != 4 or x.size(1) != 3:
        raise ValueError(f"x must have shape [N, 3, H, W], got {tuple(x.shape)}")

    delta = max_value / 2.0
    r = x[:, 0:1, :, :]
    g = x[:, 1:2, :, :]
    b = x[:, 2:3, :, :]

    y = 0.299000 * r + 0.587000 * g + 0.114000 * b
    cb = -0.168736 * r - 0.331264 * g + 0.500000 * b + delta
    cr = 0.500000 * r - 0.418688 * g - 0.081312 * b + delta
    return torch.cat([y, cb, cr], dim=1)


def to_rgb(ycbcr: torch.Tensor, max_value: float = 255.0) -> torch.Tensor:

    # ycbcr: Input tensor with shape [N, 3, H, W].
    # max_value: Dynamic range maximum (255 for 8-bit full range).

    # returns RGB tensor with shape [N, 3, H, W].
    if ycbcr.ndim != 4 or ycbcr.size(1) != 3:
        raise ValueError(f"ycbcr must have shape [N, 3, H, W], got {tuple(ycbcr.shape)}")

    delta = max_value / 2.0
    y = ycbcr[:, 0:1, :, :]
    cb = ycbcr[:, 1:2, :, :] - delta
    cr = ycbcr[:, 2:3, :, :] - delta

    r = y + 1.402000 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772000 * cb
    return torch.cat([r, g, b], dim=1)


def _epsilon_to_per_bit(epsilon, device) -> torch.Tensor:

    # assgin eps if not using  utility-aware privacy budget optimization (naive approach)
    if isinstance(epsilon, (int, float)):
        values = [float(epsilon)] * 8
    elif hasattr(epsilon, "__len__") and len(epsilon) == 8:
        values = [float(v) for v in epsilon]
    elif hasattr(epsilon, "__len__") and len(epsilon) == 2:
        lsb_eps, msb_eps = epsilon
        values = [float(lsb_eps)] * 4 + [float(msb_eps)] * 4
    else:
        raise ValueError("epsilon must be a float, length-2 sequence, or length-8 sequence")

    return torch.tensor(values, device=device, dtype=torch.float32)


def load_budgets_table(json_path=None):
    
    #load epsilon allocation table from JSON (solved by lagrangian, default is None).
    if json_path is None:
        json_path = Path(__file__).with_name("privacy_budgets.json")
    else:
        json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_epsilon_value(total_epsilon, budgets=None):

    # read table and return (epsilon_y, epsilon_c) 
    if budgets is None:
        budgets = load_budgets_table()

    key = f"{float(total_epsilon):.1f}"
    if key not in budgets:
        available = ", ".join(sorted(budgets.keys(), key=lambda x: float(x)))
        raise KeyError(f"total_epsilon={total_epsilon} not found. Available: {available}")

    schedule = budgets[key]
    return tuple(schedule["epsilon_y"]), tuple(schedule["epsilon_c"])


def bit_plane_slicing(channel_data: torch.Tensor, epsilon, device) -> torch.Tensor:
    # apply randomized reponse to each bit-plane (per channel).

    # channel_data: Tensor with shape [N, 1, H, W], values interpreted in [0, 255].
    # epsilon: Privacy budget 
    #         - float: same epsilon for all 8 bits
    #         - length-8 sequence: per-bit epsilon
    #         - length-2 sequence: [eps_lsb, eps_msb] mapped as 4 LSB + 4 MSB

    # returns perturbed channel with shape [N, 1, H, W] in [0, 255] float domain.
    if channel_data.ndim != 4 or channel_data.size(1) != 1:
        raise ValueError(f"channel_data must have shape [N, 1, H, W], got {tuple(channel_data.shape)}")

    img_byte = channel_data.clamp(0, 255).to(torch.uint8)
    shifter = torch.arange(8, device=device, dtype=torch.uint8).view(1, 1, 8, 1, 1)
    bit_planes = ((img_byte.unsqueeze(2) >> shifter) & 1).to(torch.float32)

    eps_per_bit = _epsilon_to_per_bit(epsilon, device)
    p_truth = torch.sigmoid(eps_per_bit).view(1, 1, 8, 1, 1)

    random_mask = torch.rand_like(bit_planes)
    noisy_bit_planes = torch.where(random_mask < p_truth, bit_planes, 1.0 - bit_planes)

    weights = (2 ** torch.arange(8, device=device, dtype=torch.float32)).view(1, 1, 8, 1, 1)
    return (noisy_bit_planes * weights).sum(dim=2)


#### Cached DWT/IDWT modules.##########
_dwt_cache = {}
_idwt_cache = {}
def get_dwt(wavelet: str, level: int, device, mode: str = "periodization") -> DWTForward:

    key = (wavelet, level, str(device), mode)
    if key not in _dwt_cache:
        _dwt_cache[key] = DWTForward(J=level, wave=wavelet, mode=mode).to(device)
    return _dwt_cache[key]


def get_idwt(wavelet: str, device, mode: str = "periodization") -> DWTInverse:
    key = (wavelet, str(device), mode)
    if key not in _idwt_cache:
        _idwt_cache[key] = DWTInverse(wave=wavelet, mode=mode).to(device)
    return _idwt_cache[key]
######################################

def dp_slicing_dwt(
    x: torch.Tensor,
    wavelet: str = "haar",
    level: int = 1,
    remove_ll: bool = False,
    ll_scale: float = 0.0,
    epsilon_y=1.0,
    epsilon_c=1.0,
    device="cuda",
    mode: str = "periodization",
) -> torch.Tensor:
    # LL Pruning + LDP slicing.

    # x: Input RGB tensor [N, 3, H, W], either in [0, 1] or [0, 255].
    # wavelet: e.g., "haar", "db2".
    # level: DWT decomposition level.
    # remove_ll: Pruning LL coefficients.
    # ll_scale: Scale factor applied to LL when `remove_ll=True`.
    # epsilon_y: DP epsilon specification for Y channel.
    # epsilon_c: DP epsilon specification for Cb/Cr channels.
    # device: device. 
    # mode: DWT boundary mode (default: "periodization").

    # returns obfuscated RGB tensor [N, 3, H, W] in [0, 1] with ldp guarantee.
    if x.ndim != 4 or x.size(1) != 3:
        raise ValueError(f"x must have shape [N, 3, H, W], got {tuple(x.shape)}")

    x = x.to(device)
    dwt = get_dwt(wavelet=wavelet, level=level, device=device, mode=mode)
    idwt = get_idwt(wavelet=wavelet, device=device, mode=mode)

    x_255 = x * 255.0 if x.max().item() <= 1.0 else x

    ycbcr = to_ycbcr(x_255, max_value=255.0)
    ycbcr_centered = ycbcr - 128.0

    y_cent, cb_cent, cr_cent = torch.chunk(ycbcr_centered, 3, dim=1)
    y_ll, y_h = dwt(y_cent)
    cb_ll, cb_h = dwt(cb_cent)
    cr_ll, cr_h = dwt(cr_cent)

    if remove_ll:
        y_ll = y_ll * ll_scale
        cb_ll = cb_ll * ll_scale
        cr_ll = cr_ll * ll_scale

    y_obf = (idwt((y_ll, y_h)) + 128.0).clamp(0, 255)
    cb_obf = (idwt((cb_ll, cb_h)) + 128.0).clamp(0, 255)
    cr_obf = (idwt((cr_ll, cr_h)) + 128.0).clamp(0, 255)

    y_priv = bit_plane_slicing(y_obf, epsilon=epsilon_y, device=device)
    cb_priv = bit_plane_slicing(cb_obf, epsilon=epsilon_c, device=device)
    cr_priv = bit_plane_slicing(cr_obf, epsilon=epsilon_c, device=device)

    ycbcr_priv = torch.cat([y_priv, cb_priv, cr_priv], dim=1).clamp(0, 255)
    rgb_priv_255 = to_rgb(ycbcr_priv, max_value=255.0)
    return rgb_priv_255.clamp(0.0, 255.0) / 255.0

if __name__ == "__main__":
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epsilon_y, epsilon_c = get_epsilon_value(20.0)
    dummy_input = torch.rand(1, 3, 225, 225).to(device)  # [0, 1] range
    obfuscated = dp_slicing_dwt(
        dummy_input,
        epsilon_y=epsilon_y,
        epsilon_c=epsilon_c,
        device=device,
    )
    print(obfuscated.shape)