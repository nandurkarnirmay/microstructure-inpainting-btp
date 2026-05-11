from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


@dataclass
class PreprocessConfig:
    image_size: Optional[int] = 256
    grayscale: bool = True
    normalize: str = "minmax"
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    denoise_h: float = 0.0
    patch_size: int = 128
    patch_stride: int = 64
    augment: bool = False


class MicroscopyPreprocessor:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg

    def load_image(self, path: str) -> np.ndarray:
        img = np.array(Image.open(path).convert("RGB"))
        if self.cfg.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img[..., None]
        return img.astype(np.float32)

    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        if self.cfg.clahe_clip <= 0:
            return img
        lab = img
        if lab.ndim == 2:
            lab = lab[..., None]
        out = np.empty_like(lab)
        clahe = cv2.createCLAHE(self.cfg.clahe_clip, (self.cfg.clahe_grid, self.cfg.clahe_grid))
        for c in range(lab.shape[2]):
            ch = (lab[..., c] * 255.0).clip(0, 255).astype(np.uint8)
            out[..., c] = clahe.apply(ch).astype(np.float32) / 255.0
        return out if out.shape[2] > 1 else out[..., 0]

    def denoise(self, img: np.ndarray) -> np.ndarray:
        if self.cfg.denoise_h <= 0:
            return img
        if img.ndim == 2:
            u8 = (img * 255).clip(0, 255).astype(np.uint8)
            dn = cv2.fastNlMeansDenoising(u8, h=self.cfg.denoise_h)
            return dn.astype(np.float32) / 255.0
        u8 = (img * 255).clip(0, 255).astype(np.uint8)
        dn = cv2.fastNlMeansDenoisingColored(u8, None, self.cfg.denoise_h, self.cfg.denoise_h, 7, 21)
        return dn.astype(np.float32) / 255.0

    def normalize(self, img: np.ndarray) -> np.ndarray:
        if self.cfg.normalize == "minmax":
            lo, hi = float(img.min()), float(img.max())
            if hi - lo < 1e-8:
                return np.zeros_like(img)
            return (img - lo) / (hi - lo)
        if self.cfg.normalize == "none":
            return img
        raise ValueError(f"Unknown normalize: {self.cfg.normalize}")

    def resize(self, img: np.ndarray, size: Optional[int] = None) -> np.ndarray:
        size = size or self.cfg.image_size
        if size is None:
            return img
        if img.ndim == 2:
            return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    def process_numpy(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32)
        if x.max() > 1.5:
            x = x / 255.0
        x = self.apply_clahe(x)
        x = self.denoise(x)
        x = self.normalize(x)
        x = self.resize(x)
        return x

    def to_tensor(self, img: np.ndarray) -> torch.Tensor:
        if img.ndim == 2:
            t = torch.from_numpy(img).unsqueeze(0).float()
        else:
            t = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        return t

    def extract_patches(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        c, h, w = tensor.shape
        ps, st = self.cfg.patch_size, self.cfg.patch_stride
        patches: List[torch.Tensor] = []
        for y in range(0, max(1, h - ps + 1), st):
            for x in range(0, max(1, w - ps + 1), st):
                patches.append(tensor[:, y : y + ps, x : x + ps])
        if not patches:
            patches.append(tensor)
        return patches


def random_flip_rotate(tensor: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() < 0.5:
        tensor = TF.hflip(tensor)
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        tensor = torch.rot90(tensor, k, dims=(-2, -1))
    return tensor


def build_train_transforms(augment: bool):
    def _fn(t: torch.Tensor) -> torch.Tensor:
        if augment:
            t = random_flip_rotate(t)
        return t

    return _fn
