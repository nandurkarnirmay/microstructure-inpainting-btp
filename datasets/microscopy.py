from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from preprocessing.pipeline import MicroscopyPreprocessor, PreprocessConfig


def collate_stack(batch):
    return torch.stack(batch, dim=0)


class MicroscopyFolderDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: Optional[str | Path] = None,
        preprocess: Optional[MicroscopyPreprocessor] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.pre = preprocess or MicroscopyPreprocessor(PreprocessConfig())
        self.transform = transform
        self.paths: List[Path] = []
        for ext in extensions:
            self.paths.extend(sorted(self.images_dir.glob(f"*{ext}")))
            self.paths.extend(sorted(self.images_dir.glob(f"*{ext.upper()}")))
        self.paths = sorted(set(self.paths))
        if not self.paths:
            raise FileNotFoundError(f"No images under {self.images_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def _load_mask(self, image_path: Path) -> Optional[torch.Tensor]:
        if self.masks_dir is None:
            return None
        cand = self.masks_dir / (image_path.stem + ".png")
        if not cand.is_file():
            cand = self.masks_dir / (image_path.stem + ".tif")
        if not cand.is_file():
            return None
        import numpy as np
        from PIL import Image

        m = np.array(Image.open(cand).convert("L")).astype(np.float32) / 255.0
        m = torch.from_numpy(m).unsqueeze(0).float()
        return m

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img_np = self.pre.load_image(str(p))
        img_np = self.pre.process_numpy(img_np)
        img = self.pre.to_tensor(img_np)
        mask = self._load_mask(p)
        if mask is None:
            mask = torch.zeros_like(img[:1])
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        return {"image": img, "mask": mask, "path": str(p)}


class SyntheticInpaintDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        hole_min: int = 32,
        hole_max: int = 96,
        image_size: int = 256,
        grayscale: bool = True,
        seed: int = 0,
    ):
        self.base = MicroscopyFolderDataset(
            images_dir,
            masks_dir=None,
            preprocess=MicroscopyPreprocessor(
                PreprocessConfig(
                    image_size=image_size,
                    grayscale=grayscale,
                    augment=False,
                )
            ),
            transform=None,
        )
        self.hole_min = hole_min
        self.hole_max = hole_max
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.base)

    def _random_mask(self, h: int, w: int, g: random.Random) -> torch.Tensor:
        hh = g.randint(self.hole_min, min(self.hole_max, h - 2))
        ww = g.randint(self.hole_min, min(self.hole_max, w - 2))
        y0 = g.randint(0, h - hh)
        x0 = g.randint(0, w - ww)
        m = torch.zeros(1, h, w)
        m[:, y0 : y0 + hh, x0 : x0 + ww] = 1.0
        return m

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        img = sample["image"]
        g = random.Random(self._rng.randint(0, 2**30))
        mask = self._random_mask(img.shape[-2], img.shape[-1], g)
        masked = img * (1.0 - mask)
        return {
            "image": img,
            "mask": mask,
            "masked": masked,
            "path": sample["path"],
        }
