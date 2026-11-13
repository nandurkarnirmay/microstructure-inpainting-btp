r"""One-minute check, BEFORE retraining: do the training images and masks line up?

  python check_align.py
  python check_align.py --images data\unet_voronoi_images --masks data\unet_voronoi_masks

Loads samples exactly as train_unet.py does (same config, augmentation on)
through the loader train_unet.py imports (add --compare_original to also run
datasets/microscopy_original.py, the pre-fix loader, if you still have it). Damage is painted
flat, so inside a correctly placed mask the image has almost no texture.
  ratio = std(image inside mask) / std(image outside mask)
Aligned  -> ratio near 0.   Misaligned -> ratio near 1 (mask sits on real texture).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from preprocessing.pipeline import MicroscopyPreprocessor, PreprocessConfig, build_train_transforms
from utils.config import load_config

N = 200


def run(module_name, IMAGES, MASKS):
    mod = __import__(f"datasets.{module_name}", fromlist=["MicroscopyFolderDataset"])
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    p = cfg.get("preprocessing", {})
    pre = MicroscopyPreprocessor(PreprocessConfig(
        image_size=p.get("image_size"),
        grayscale=bool(p.get("grayscale", True)),
        normalize=p.get("normalize", "minmax"),
        clahe_clip=float(p.get("clahe_clip", 2.0)),
        clahe_grid=int(p.get("clahe_grid", 8)),
        denoise_h=float(p.get("denoise_h", 0.0)),
        augment=bool(p.get("augment", False)),
    ))
    augment = bool(p.get("augment", False))
    ds = mod.MicroscopyFolderDataset(IMAGES, masks_dir=MASKS, preprocess=pre,
                                     transform=build_train_transforms(augment))
    torch.manual_seed(0)
    ratios = []
    for i in range(min(N, len(ds))):
        s = ds[i]
        img = s["image"][0].numpy()
        m = s["mask"][0].numpy() > 0.5
        if m.sum() < 50 or (~m).sum() < 50:
            continue
        out_std = img[~m].std()
        if out_std > 1e-6:
            ratios.append(img[m].std() / out_std)
    r = np.array(ratios)
    print(f"{module_name:22s} augment={augment}  n={len(r)}  "
          f"median ratio {np.median(r):.3f}   share aligned (<0.3) {100*(r < 0.3).mean():.0f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "data" / "unet_images"))
    ap.add_argument("--masks", default=str(ROOT / "data" / "unet_masks"))
    ap.add_argument("--compare_original", action="store_true",
                    help="also run datasets/microscopy_original.py (the pre-fix loader), if present")
    a = ap.parse_args()
    if a.compare_original:
        run("microscopy_original", Path(a.images), Path(a.masks))
    run("microscopy", Path(a.images), Path(a.masks))   # what train_unet.py imports
