#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from inference.pipeline import InpaintPipeline
from models.gan import SpatialNoiseGenerator
from utils.config import load_config
from utils.io import ensure_dir
from utils.seeds import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--gopt_bundle", type=str, required=True, help="Path to gopt_bundle.pt")
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--out", type=str, default=str(ROOT / "outputs" / "zopt_result.pt"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    pipe = InpaintPipeline(cfg, device)

    img_np = pipe.pre.load_image(args.image)
    img_np = pipe.pre.process_numpy(img_np)
    t = pipe.pre.to_tensor(img_np).unsqueeze(0).to(device)

    bundle = torch.load(args.gopt_bundle, map_location=device, weights_only=False)
    G = SpatialNoiseGenerator(
        out_channels=int(cfg.get("gan", {}).get("out_channels", 1)),
        z_channels=int(cfg.get("gan", {}).get("z_channels", 1)),
        base=64,
        scale_factor=int(bundle.get("scale_factor", cfg.get("gopt", {}).get("scale_factor", 8))),
        num_upsample=1,
    ).to(device)
    G.load_state_dict(bundle["G"], strict=True)
    G.eval()

    hole = torch.zeros_like(t[:, :1])
    h, w = t.shape[-2], t.shape[-1]
    hs = int(bundle.get("hole", cfg.get("gopt", {}).get("hole_size", 64)))
    y0, x0 = (h - hs) // 2, (w - hs) // 2
    hole[:, :, y0 : y0 + hs, x0 : x0 + hs] = 1.0

    z_init = bundle.get("z_fixed")
    if z_init is None:
        ss = int(bundle.get("seed_side", 10))
        z_init = torch.randn(1, 1, ss, ss)
    z, fused = pipe.z_opt_refinement(G, t, hole, z_init=z_init.to(device))

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    torch.save({"z": z.cpu(), "fused": fused.cpu(), "image": t.cpu(), "hole": hole.cpu()}, out_path)
    print("saved", out_path)


if __name__ == "__main__":
    main()
