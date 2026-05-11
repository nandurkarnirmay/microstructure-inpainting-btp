#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torchvision.utils import save_image

from inference.pipeline import InpaintPipeline
from models.gan import MaskGuidedGenerator, SpatialNoiseGenerator
from training.cyclic import cyclic_inpaint_round
from utils.config import load_config
from utils.io import ensure_dir
from utils.seeds import set_seed
from visualization.viz import save_comparison_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--unet_ckpt", type=str, default=None)
    ap.add_argument("--gan_ckpt", type=str, default=None)
    ap.add_argument("--gopt_bundle", type=str, default=None)
    ap.add_argument("--run_zopt", action="store_true", help="Run z-opt using gopt_bundle (requires --gopt_bundle)")
    ap.add_argument("--use_cyclic", action="store_true")
    ap.add_argument("--out_dir", type=str, default=str(ROOT / "outputs" / "infer"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(Path(args.out_dir))
    pipe = InpaintPipeline(cfg, device)
    pre = pipe.pre
    blend = cfg.get("blending", {})

    img_np = pre.load_image(args.image)
    img_np = pre.process_numpy(img_np)
    t = pre.to_tensor(img_np).unsqueeze(0).to(device)

    unet = None
    if args.unet_ckpt:
        unet = pipe.load_unet(
            args.unet_ckpt,
            in_channels=int(cfg.get("unet", {}).get("in_channels", 1)),
            base_channels=int(cfg.get("unet", {}).get("base_channels", 32)),
            num_levels=int(cfg.get("unet", {}).get("num_levels", 4)),
        )
        m = pipe.segment_defects(unet, t)
        m = pipe.refine_mask(m)
        save_image(m, out_dir / "mask.png")
    else:
        m = torch.zeros_like(t[:, :1])
        h, w = t.shape[-2], t.shape[-1]
        m[..., h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0

    preds = []
    if args.gan_ckpt:
        G = pipe.load_gan(
            args.gan_ckpt,
            in_channels=int(cfg.get("gan", {}).get("in_channels", 2)),
            out_channels=int(cfg.get("gan", {}).get("out_channels", 1)),
            base=int(cfg.get("gan", {}).get("base_channels", 48)),
        )
        if args.use_cyclic and unet is not None:
            pred, _, rounds = cyclic_inpaint_round(
                t,
                m,
                G,
                None,
                unet,
                max_rounds=int(cfg.get("cyclic", {}).get("max_rounds", 2)),
                defect_threshold=float(cfg.get("cyclic", {}).get("defect_threshold", 0.15)),
                border=int(cfg.get("gopt", {}).get("annulus_width", 8)),
                blend_mode=str(blend.get("mode", "feather")),
                feather_kernel=int(blend.get("feather_kernel", 9)),
                feather_sigma=float(blend.get("feather_sigma", 2.0)),
            )
            print("cyclic rounds", rounds)
        else:
            pred = pipe.inpaint_masked_gan(G, t, m)
        preds.append(("gan", pred))
        save_image(pred, out_dir / "inpaint_gan.png")

    if args.gopt_bundle:
        bundle = torch.load(args.gopt_bundle, map_location=device, weights_only=False)
        Gs = SpatialNoiseGenerator(
            out_channels=int(cfg.get("gan", {}).get("out_channels", 1)),
            z_channels=int(cfg.get("gan", {}).get("z_channels", 1)),
            base=64,
            scale_factor=int(bundle.get("scale_factor", cfg.get("gopt", {}).get("scale_factor", 8))),
            num_upsample=1,
        ).to(device)
        Gs.load_state_dict(bundle["G"], strict=True)
        Gs.eval()
        zf = bundle["z_fixed"].to(device)
        pred_s = pipe.inpaint_spatial_gopt(Gs, t, m, zf, seed_side=int(bundle.get("seed_side", zf.shape[-1])))
        preds.append(("gopt", pred_s))
        save_image(pred_s, out_dir / "inpaint_gopt.png")

        if args.run_zopt:
            z, fused = pipe.z_opt_refinement(Gs, t, m, z_init=zf)
            preds.append(("zopt", fused))
            save_image(fused, out_dir / "inpaint_zopt.png")
            torch.save({"z": z.cpu(), "fused": fused.cpu()}, out_dir / "zopt_bundle.pt")

    if preds:
        save_comparison_grid(
            [t, m, preds[-1][1]],
            ["input", "mask", preds[-1][0]],
            out_dir / "grid.png",
        )
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
