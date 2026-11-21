#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
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
    ap.add_argument("--out_dir", type=str, default=None,
                    help="checkpoint folder, e.g. checkpoints\\gan_checkpoints\\gan_spinodal "
                         "(default: checkpoints\\gan, the old behaviour)")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow training into a folder that already holds gan_weights.pt")
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
    if args.out_dir:
        ckpt_dir = Path(args.out_dir)
    else:
        ckpt_dir = Path(cfg.get("paths", {}).get("checkpoints", ROOT / "checkpoints")) / "gan"
    if (ckpt_dir / "gan_weights.pt").exists() and not args.overwrite:
        raise SystemExit(f"{ckpt_dir / 'gan_weights.pt'} already exists. Choose another --out_dir, "
                         f"or add --overwrite if you really mean to replace it.")
    ckpt_dir = ensure_dir(ckpt_dir)
    log_dir = ensure_dir(Path(cfg.get("paths", {}).get("logs", ROOT / "outputs" / "logs")) / "gan" / ckpt_dir.name)

    # record what this checkpoint is, so its history never has to be rebuilt from file dates
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        commit = "unknown"
    names = [Path(p).name for p in ds.base.paths]
    kinds = {}
    for n in names:
        k = n.split("_")[0]
        kinds[k] = kinds.get(k, 0) + 1
    info = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "finished": None,
        "images": str(Path(args.images).resolve()),
        "n_images": len(names),
        "image_kinds": kinds,
        "epochs_planned": int(gan_cfg.get("epochs", 0)),
        "hole_min": int(gan_cfg.get("hole_min", 32)),
        "hole_max": int(gan_cfg.get("hole_max", 96)),
        "config": str(Path(args.config).resolve()),
        "gan_config": gan_cfg,
        "git_commit": commit,
    }
    info_path = ckpt_dir / "training_info.json"
    info_path.write_text(json.dumps(info, indent=2))
    print(f"training {len(names)} images {kinds} -> {ckpt_dir}")

    train_gan_loop(G, D, loader, gan_cfg, device, ckpt_dir, log_dir)

    info["finished"] = datetime.now().isoformat(timespec="seconds")
    info_path.write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
