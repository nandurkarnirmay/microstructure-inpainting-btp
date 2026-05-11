from __future__ import annotations

import numpy as np
import torch


def phase_fraction(img: np.ndarray, threshold: float = 0.5) -> float:
    if img.ndim == 3:
        img = img[..., 0]
    m = (img >= threshold).astype(np.float64)
    return float(m.mean())


def phase_fraction_error_percent(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    pf_p = phase_fraction(pred, threshold)
    pf_g = phase_fraction(gt, threshold)
    denom = max(abs(pf_g), 1e-6)
    return float(abs(pf_p - pf_g) / denom * 100.0)


def _two_point_correlation_2d(binary_hw: np.ndarray) -> np.ndarray:
    h, w = binary_hw.shape
    p = binary_hw.astype(np.float64)
    p = p - p.mean()
    f = np.fft.fft2(p)
    ac = np.real(np.fft.ifft2(f * np.conj(f))) / (h * w * np.var(p) + 1e-12)
    ac = np.fft.fftshift(ac)
    return ac


def radial_profile(ac: np.ndarray, max_r: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    h, w = ac.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    if max_r is None:
        max_r = min(cy, cx, max(h, w) // 2)
    mask = r <= float(max_r)
    r_int = r[mask].astype(np.int32).ravel()
    vals = ac[mask].ravel()
    sums = np.bincount(r_int, weights=vals, minlength=max_r + 1)
    cnt = np.bincount(r_int, minlength=max_r + 1)
    prof = sums / (cnt + 1e-12)
    rs = np.arange(max_r + 1, dtype=np.float64)
    return rs, prof


def s2_mse(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5, max_r: int = 64) -> float:
    if pred.ndim == 3:
        pred = pred[..., 0]
    if gt.ndim == 3:
        gt = gt[..., 0]
    bp = (pred >= threshold).astype(np.float64)
    bg = (gt >= threshold).astype(np.float64)
    acp = _two_point_correlation_2d(bp)
    acg = _two_point_correlation_2d(bg)
    _, pr = radial_profile(acp, max_r=max_r)
    _, gr = radial_profile(acg, max_r=max_r)
    m = min(len(pr), len(gr))
    return float(np.mean((pr[:m] - gr[:m]) ** 2))


def s2_mse_from_tensors(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5, max_r: int = 64) -> float:
    p = pred.detach().cpu().float().numpy()
    g = gt.detach().cpu().float().numpy()
    if p.ndim == 4:
        p = p[0, 0]
    elif p.ndim == 3:
        p = p[0]
    if g.ndim == 4:
        g = g[0, 0]
    elif g.ndim == 3:
        g = g[0]
    p = np.clip(p, 0.0, 1.0)
    g = np.clip(g, 0.0, 1.0)
    return s2_mse(p, g, threshold=threshold, max_r=max_r)
