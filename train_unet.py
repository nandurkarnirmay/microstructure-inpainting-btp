#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, random_split

from datasets.microscopy import MicroscopyFolderDataset
from models.unet import UNet
from preprocessing.pipeline import MicroscopyPreprocessor, PreprocessConfig, build_train_transforms
from training.unet_trainer import train_unet_loop
from utils.config import load_config
from utils.io import ensure_dir
from utils.seeds import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--images", type=str, required=True, help="Folder of microscopy images")
    ap.add_argument("--masks", type=str, default=None, help="Folder of binary masks (same stem as images)")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    pre_cfg = cfg.get("preprocessing", {})
    unet_cfg = cfg.get("unet", {})
    pre = MicroscopyPreprocessor(
        PreprocessConfig(
            image_size=pre_cfg.get("image_size"),
            grayscale=bool(pre_cfg.get("grayscale", True)),
            normalize=pre_cfg.get("normalize", "minmax"),
            clahe_clip=float(pre_cfg.get("clahe_clip", 2.0)),
            clahe_grid=int(pre_cfg.get("clahe_grid", 8)),
            denoise_h=float(pre_cfg.get("denoise_h", 0.0)),
            augment=bool(pre_cfg.get("augment", False)),
        )
    )
    tfm = build_train_transforms(bool(pre_cfg.get("augment", False)))
    ds = MicroscopyFolderDataset(args.images, masks_dir=args.masks, preprocess=pre, transform=tfm)
    n_val = max(1, int(len(ds) * args.val_ratio))
    n_tr = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(int(cfg.get("seed", 42))))

    train_loader = DataLoader(train_ds, batch_size=int(unet_cfg.get("batch_size", 4)), shuffle=True, num_workers=0, collate_fn=None)
    val_loader = DataLoader(val_ds, batch_size=int(unet_cfg.get("batch_size", 4)), shuffle=False, num_workers=0)

    model = UNet(
        in_channels=int(unet_cfg.get("in_channels", 1)),
        out_channels=1,
        base_channels=int(unet_cfg.get("base_channels", 32)),
        num_levels=int(unet_cfg.get("num_levels", 4)),
    ).to(device)

    ckpt_dir = ensure_dir(Path(cfg.get("paths", {}).get("checkpoints", ROOT / "checkpoints")) / "unet")
    log_dir = ensure_dir(Path(cfg.get("paths", {}).get("logs", ROOT / "outputs" / "logs")) / "unet")
    train_unet_loop(model, train_loader, val_loader, unet_cfg, device, ckpt_dir, log_dir)


if __name__ == "__main__":
    main()
