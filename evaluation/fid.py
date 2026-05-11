from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


def _list_images(folder: Path, exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")) -> List[Path]:
    paths: List[Path] = []
    for e in exts:
        paths.extend(folder.glob(f"*{e}"))
        paths.extend(folder.glob(f"*{e.upper()}"))
    return sorted(set(paths))


class _InceptionFeats(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights

        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False, aux_logits=False)
        net.fc = nn.Identity()
        net.dropout = nn.Identity()
        self.net = net.to(device).eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.net(x)


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
        sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def compute_fid_between_folders(
    real_dir: str | Path,
    gen_dir: str | Path,
    device: torch.device,
    batch_size: int = 16,
    max_images: int = 500,
) -> float:
    real_paths = _list_images(Path(real_dir))[:max_images]
    gen_paths = _list_images(Path(gen_dir))[:max_images]
    if len(real_paths) < 2 or len(gen_paths) < 2:
        raise ValueError("FID requires at least 2 images in each folder.")
    extractor = _InceptionFeats(device)

    def feats(paths: List[Path]) -> np.ndarray:
        from PIL import Image

        outs: List[torch.Tensor] = []
        for i in range(0, len(paths), batch_size):
            chunk = paths[i : i + batch_size]
            batch = []
            for p in chunk:
                arr = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
                batch.append(torch.from_numpy(arr).unsqueeze(0))
            x = torch.stack(batch, dim=0).to(device)
            f = extractor(x)
            outs.append(f.cpu())
        return torch.cat(outs, dim=0).numpy()

    r = feats(real_paths)
    g = feats(gen_paths)
    mu1, mu2 = r.mean(axis=0), g.mean(axis=0)
    if r.shape[0] > 1:
        sigma1 = np.cov(r, rowvar=False)
    else:
        sigma1 = np.zeros((r.shape[1], r.shape[1]))
    if g.shape[0] > 1:
        sigma2 = np.cov(g, rowvar=False)
    else:
        sigma2 = np.zeros((g.shape[1], g.shape[1]))
    return _frechet_distance(mu1, sigma1, mu2, sigma2)


def compute_fid_optional(
    real_dir: Optional[str | Path],
    gen_dir: Optional[str | Path],
    device: torch.device,
) -> Optional[float]:
    if real_dir is None or gen_dir is None:
        return None
    try:
        return compute_fid_between_folders(real_dir, gen_dir, device=device)
    except Exception:
        return None
