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

from datasets.microscopy import MicroscopyFolderDataset
from models.gan import PatchDiscriminator, SpatialNoiseGenerator
from preprocessing.pipeline import MicroscopyPreprocessor, PreprocessConfig, build_train_transforms
from training.gopt_trainer import train_gopt_loop
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
    gopt_cfg = cfg.get("gopt", {})
    pre_cfg = cfg.get("preprocessing", {})
    pre = MicroscopyPreprocessor(
        PreprocessConfig(
            image_size=int(pre_cfg.get("image_size", 256)),
            grayscale=bool(pre_cfg.get("grayscale", True)),
            normalize=pre_cfg.get("normalize", "minmax"),
            clahe_clip=float(pre_cfg.get("clahe_clip", 2.0)),
            clahe_grid=int(pre_cfg.get("clahe_grid", 8)),
            denoise_h=float(pre_cfg.get("denoise_h", 0.0)),
            augment=bool(pre_cfg.get("augment", False)),
        )
    )
    tfm = build_train_transforms(bool(pre_cfg.get("augment", False)))
    ds = MicroscopyFolderDataset(args.images, masks_dir=None, preprocess=pre, transform=tfm)
    loader = DataLoader(ds, batch_size=int(cfg.get("gan", {}).get("batch_size", 4)), shuffle=True, num_workers=0)
    out_ch = int(cfg.get("gan", {}).get("out_channels", 1))
    z_ch = int(cfg.get("gan", {}).get("z_channels", 1))
    G = SpatialNoiseGenerator(
        out_channels=out_ch,
        z_channels=z_ch,
        base=64,
        scale_factor=int(gopt_cfg.get("scale_factor", 8)),
        num_upsample=1,
    ).to(device)
    D = PatchDiscriminator(in_channels=out_ch + 1, base=64).to(device)
    ckpt_dir = ensure_dir(Path(cfg.get("paths", {}).get("checkpoints", ROOT / "checkpoints")) / "gopt")
    log_dir = ensure_dir(Path(cfg.get("paths", {}).get("logs", ROOT / "outputs" / "logs")) / "gopt")
    train_gopt_loop(G, D, loader, gopt_cfg, device, ckpt_dir, log_dir)


if __name__ == "__main__":
    main()
