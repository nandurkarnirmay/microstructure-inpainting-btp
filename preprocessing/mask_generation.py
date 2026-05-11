from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass
class MaskGenConfig:
    binarize_threshold: float = 0.5
    morph_kernel: int = 3
    min_component_area: int = 32


def logits_to_mask(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) > threshold).float()


def refine_mask_numpy(mask: np.ndarray, cfg: MaskGenConfig) -> np.ndarray:
    m = (mask > cfg.binarize_threshold).astype(np.uint8)
    k = max(1, int(cfg.morph_kernel))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= cfg.min_component_area:
            out[labels == i] = 1
    return out.astype(np.float32)
