#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from evaluation.metrics import compute_full_metrics, compute_metrics
from preprocessing.pipeline import MicroscopyPreprocessor, PreprocessConfig
from utils.config import load_config


def _load_tensor(pre: MicroscopyPreprocessor, path: str) -> torch.Tensor:
    img_np = pre.load_image(path)
    img_np = pre.process_numpy(img_np)
    return pre.to_tensor(img_np).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--pred", type=str, required=True, help="Restored / predicted image")
    ap.add_argument("--gt", type=str, required=True, help="Ground-truth clean image")
    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument("--fid_real_dir", type=str, default=None, help="Folder of real patches for FID")
    ap.add_argument("--fid_gen_dir", type=str, default=None, help="Folder of generated patches for FID")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pre_cfg = cfg.get("preprocessing", {})
    pre = MicroscopyPreprocessor(
        PreprocessConfig(
            image_size=pre_cfg.get("image_size"),
            grayscale=bool(pre_cfg.get("grayscale", True)),
            normalize=pre_cfg.get("normalize", "minmax"),
            clahe_clip=float(pre_cfg.get("clahe_clip", 2.0)),
            clahe_grid=int(pre_cfg.get("clahe_grid", 8)),
            denoise_h=float(pre_cfg.get("denoise_h", 0.0)),
        )
    )
    pred = _load_tensor(pre, args.pred)
    gt = _load_tensor(pre, args.gt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.fid_real_dir and args.fid_gen_dir:
        rep = compute_full_metrics(
            pred,
            gt,
            fid_real_dir=args.fid_real_dir,
            fid_gen_dir=args.fid_gen_dir,
            device=device,
            s2_max_r=int(cfg.get("evaluation", {}).get("s2_max_r", 64)),
            phase_threshold=float(cfg.get("evaluation", {}).get("phase_threshold", 0.5)),
        )
    else:
        rep = compute_metrics(
            pred,
            gt,
            include_microstructure=True,
            s2_max_r=int(cfg.get("evaluation", {}).get("s2_max_r", 64)),
            phase_threshold=float(cfg.get("evaluation", {}).get("phase_threshold", 0.5)),
        )
    d = rep.as_dict()
    print(json.dumps(d, indent=2))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)


if __name__ == "__main__":
    main()
