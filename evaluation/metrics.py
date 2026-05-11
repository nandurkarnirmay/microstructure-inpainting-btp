from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .microstructure_stats import phase_fraction_error_percent, s2_mse


@dataclass
class MetricsReport:
    psnr: float
    ssim: float
    mse: float
    mae: float
    dice: float
    iou: float
    boundary_f1: Optional[float] = None
    s2_mse: Optional[float] = None
    phase_error_percent: Optional[float] = None
    fid: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "psnr": self.psnr,
            "ssim": self.ssim,
            "mse": self.mse,
            "mae": self.mae,
            "dice": self.dice,
            "iou": self.iou,
            "boundary_f1": self.boundary_f1,
        }
        if self.s2_mse is not None:
            d["s2_mse"] = self.s2_mse
        if self.phase_error_percent is not None:
            d["phase_error_percent"] = self.phase_error_percent
        if self.fid is not None:
            d["fid"] = self.fid
        return d


def _to_numpy01(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().float().numpy()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        x = np.transpose(x, (1, 2, 0))
    return np.clip(x, 0.0, 1.0)


def dice_iou(pred: torch.Tensor, gt: torch.Tensor, thr: float = 0.5) -> tuple[float, float]:
    p = pred.detach()
    g = gt.detach()
    if p.dim() == 4:
        p = p[0:1]
    if g.dim() == 4:
        g = g[0:1]
    p = (p > thr).float()
    g = (g > thr).float()
    inter = (p * g).sum()
    denom = p.sum() + g.sum() + 1e-8
    dice = (2.0 * inter + 1e-8) / denom
    union = p.sum() + g.sum() - inter + 1e-8
    iou = (inter + 1e-8) / union
    return float(dice.item()), float(iou.item())


def boundary_f1_score(pred: torch.Tensor, gt: torch.Tensor, thr: float = 0.5, width: int = 2) -> float:
    p = pred.detach()
    g = gt.detach()
    if p.dim() == 3:
        p = p.unsqueeze(0)
    if g.dim() == 3:
        g = g.unsqueeze(0)
    p = p[:, :1, :, :]
    g = g[:, :1, :, :]
    p = (p > thr).float()
    g = (g > thr).float()

    def edge(m: torch.Tensor) -> torch.Tensor:
        dil = m
        ero = m
        for _ in range(width):
            dil = F.max_pool2d(dil, 3, 1, 1)
        for _ in range(width):
            ero = -F.max_pool2d(-ero, 3, 1, 1)
        return (dil - ero).clamp(0, 1)

    ep, eg = edge(p), edge(g)
    inter = (ep * eg).sum()
    prec = inter / (ep.sum() + 1e-8)
    rec = inter / (eg.sum() + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return float(f1.item())


def compute_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    data_range: float = 1.0,
    *,
    include_microstructure: bool = True,
    s2_max_r: int = 64,
    phase_threshold: float = 0.5,
) -> MetricsReport:
    p = _to_numpy01(pred)
    g = _to_numpy01(gt)
    if p.ndim == 2:
        p = p[..., None]
        g = g[..., None]
    mse = float(np.mean((p - g) ** 2))
    mae = float(np.mean(np.abs(p - g)))
    psnr = float(peak_signal_noise_ratio(g, p, data_range=data_range))
    h, w = p.shape[0], p.shape[1]
    win = min(7, h, w)
    if win % 2 == 0:
        win = max(3, win - 1)
    if p.ndim == 3 and p.shape[2] == 1:
        ssim = float(structural_similarity(g[..., 0], p[..., 0], data_range=data_range, win_size=win))
    elif p.ndim == 2:
        ssim = float(structural_similarity(g, p, data_range=data_range, win_size=win))
    else:
        ssim = float(structural_similarity(g, p, data_range=data_range, channel_axis=-1, win_size=win))
    dice, iou = dice_iou(pred, gt)
    bf1 = boundary_f1_score(pred, gt)
    s2v: Optional[float] = None
    pe: Optional[float] = None
    if include_microstructure:
        s2v = float(s2_mse(p, g, threshold=phase_threshold, max_r=s2_max_r))
        pe = float(phase_fraction_error_percent(p, g, threshold=phase_threshold))
    return MetricsReport(
        psnr=psnr,
        ssim=ssim,
        mse=mse,
        mae=mae,
        dice=dice,
        iou=iou,
        boundary_f1=bf1,
        s2_mse=s2v,
        phase_error_percent=pe,
        fid=None,
    )


def compute_full_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    fid_real_dir: Optional[str] = None,
    fid_gen_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> MetricsReport:
    metric_keys = {"data_range", "include_microstructure", "s2_max_r", "phase_threshold"}
    rep = compute_metrics(pred, gt, **{k: v for k, v in kwargs.items() if k in metric_keys})
    fid_val: Optional[float] = None
    if fid_real_dir and fid_gen_dir and device is not None:
        try:
            from .fid import compute_fid_between_folders

            fid_val = compute_fid_between_folders(fid_real_dir, fid_gen_dir, device=device)
        except Exception:
            fid_val = None
    return MetricsReport(
        psnr=rep.psnr,
        ssim=rep.ssim,
        mse=rep.mse,
        mae=rep.mae,
        dice=rep.dice,
        iou=rep.iou,
        boundary_f1=rep.boundary_f1,
        s2_mse=rep.s2_mse,
        phase_error_percent=rep.phase_error_percent,
        fid=fid_val,
    )
