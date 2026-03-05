import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ldp_slicing import dp_slicing_dwt, get_epsilon_value, load_budgets_table


def load_image(path: Path, device: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # H, W, C in [0, 1]
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1, 3, H, W
    return x.to(device)


def save_image(t: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = t.clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()  # H, W, C
    arr = (t * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def generate_for_image(
    img_path: Path,
    out_dir: Path,
    total_epsilons,
    device: str,
    budgets: dict,
) -> None:
    x = load_image(img_path, device)
    stem = img_path.stem

    for total_eps in total_epsilons:
        eps_y, eps_c = get_epsilon_value(total_eps, budgets=budgets)
        x_priv = dp_slicing_dwt(
            x,
            wavelet="haar",
            level=1,
            remove_ll=True,
            ll_scale=0.0,
            epsilon_y=eps_y,
            epsilon_c=eps_c,
            device=device,
        )

        eps_key = f"{float(total_eps):.1f}".replace(".", "_")
        out_name = f"{stem}_eps{eps_key}.png"
        out_path = out_dir / "priv" / out_name
        print(f"[{stem}] ε={total_eps} -> {out_path}")
        save_image(x_priv, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate privatized comparison images for GitHub Pages."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing raw comparison images (e.g. coda_webpage/LDP-Slicing/assets/cmp_imgs).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Directory to store privatized images (default: same as input_dir).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        nargs="*",
        default=None,
        help="List of total epsilons (e.g. --eps 1.0 5.2 20.0). "
        "If omitted, use all keys from privacy_budgets.json.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir
    )

    if not input_dir.is_dir():
        raise SystemExit(f"input_dir does not exist: {input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    budgets = load_budgets_table()

    if args.eps is None:
        total_epsilons = sorted(float(k) for k in budgets.keys())
    else:
        total_epsilons = args.eps

    img_paths = [
        p
        for p in input_dir.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and "_eps" not in p.stem  # avoid re-processing outputs
    ]
    if not img_paths:
        raise SystemExit(f"No input images found under {input_dir}")

    for p in img_paths:
        generate_for_image(p, output_dir, total_epsilons, device, budgets)


if __name__ == "__main__":
    main()

