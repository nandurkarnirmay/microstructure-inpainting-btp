#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from datasets.microscopy import SyntheticInpaintDataset
from models.gan import MaskGuidedGenerator, PatchDiscriminator
from training.gan_trainer import train_gan_loop
from utils.config import load_config
from utils.io import ensure_dir
from utils.seeds import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--images", type=str, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    gan_cfg = cfg.get("gan", {})
    pre_cfg = cfg.get("preprocessing", {})
    ds = SyntheticInpaintDataset(
        args.images,
        hole_min=int(gan_cfg.get("hole_min", 32)),
        hole_max=int(gan_cfg.get("hole_max", 96)),
        image_size=int(pre_cfg.get("image_size", 256)),
        grayscale=bool(pre_cfg.get("grayscale", True)),
        seed=int(cfg.get("seed", 42)),
    )
    loader = DataLoader(ds, batch_size=int(gan_cfg.get("batch_size", 4)), shuffle=True, num_workers=0)
    in_ch = int(gan_cfg.get("in_channels", 2))
    out_ch = int(gan_cfg.get("out_channels", 1))
    G = MaskGuidedGenerator(in_channels=in_ch, out_channels=out_ch, base=int(gan_cfg.get("base_channels", 48))).to(device)
    D = PatchDiscriminator(in_channels=out_ch + 1, base=64).to(device)
    ckpt_dir = ensure_dir(Path(cfg.get("paths", {}).get("checkpoints", ROOT / "checkpoints")) / "gan")
    log_dir = ensure_dir(Path(cfg.get("paths", {}).get("logs", ROOT / "outputs" / "logs")) / "gan")
    train_gan_loop(G, D, loader, gan_cfg, device, ckpt_dir, log_dir)


if __name__ == "__main__":
    main()
