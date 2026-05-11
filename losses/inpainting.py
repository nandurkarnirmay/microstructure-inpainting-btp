from __future__ import annotations

import torch
import torch.nn.functional as F


def annulus_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    m = (mask > 0.5).float()
    dil, ero = m, m
    for _ in range(width):
        dil = F.max_pool2d(dil, kernel_size=3, stride=1, padding=1)
    for _ in range(width):
        ero = -F.max_pool2d(-ero, kernel_size=3, stride=1, padding=1)
    return (dil - ero).clamp(0, 1)


def annulus_mse(pred: torch.Tensor, target: torch.Tensor, ring: torch.Tensor) -> torch.Tensor:
    ring = ring.expand_as(pred)
    num = ring.sum() * pred.shape[1] + 1e-8
    diff = (pred - target) * ring
    return (diff.pow(2).sum()) / num


def erode_mask(mask: torch.Tensor, iters: int) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    m = (mask > 0.5).float()
    for _ in range(iters):
        m = -F.max_pool2d(-m, kernel_size=3, stride=1, padding=1)
    return m


def composite_keep_boundary(pred: torch.Tensor, image: torch.Tensor, hole: torch.Tensor, border: int) -> torch.Tensor:
    inner = erode_mask(hole, border)
    return pred * inner + image * (1.0 - inner)


def _gaussian_kernel2d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-0.5 * (ax / (sigma + 1e-8)) ** 2)
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)


def blur_channels(x: torch.Tensor, kernel_size: int = 9, sigma: float = 2.0) -> torch.Tensor:
    b, c, h, w = x.shape
    k = _gaussian_kernel2d(kernel_size, sigma, x.device, x.dtype)
    k = k.repeat(c, 1, 1, 1)
    pad = kernel_size // 2
    return F.conv2d(x, k, padding=pad, groups=c)


def composite_feathered(
    pred: torch.Tensor,
    image: torch.Tensor,
    hole: torch.Tensor,
    border: int,
    feather_kernel: int = 9,
    feather_sigma: float = 2.0,
) -> torch.Tensor:
    inner = erode_mask(hole, border)
    alpha = blur_channels(inner.float(), kernel_size=feather_kernel, sigma=feather_sigma).clamp(0.0, 1.0)
    if float(alpha.sum()) < 1e-4:
        return composite_keep_boundary(pred, image, hole, border)
    return pred * alpha + image * (1.0 - alpha)


def composite(pred: torch.Tensor, image: torch.Tensor, hole: torch.Tensor, border: int, mode: str = "feather", **kwargs) -> torch.Tensor:
    if mode == "hard":
        return composite_keep_boundary(pred, image, hole, border)
    fk = int(kwargs.get("feather_kernel", 9))
    fs = float(kwargs.get("feather_sigma", 2.0))
    return composite_feathered(pred, image, hole, border, feather_kernel=fk, feather_sigma=fs)


def kl_to_standard_normal(z: torch.Tensor, z_ref: torch.Tensor | None = None) -> torch.Tensor:
    if z_ref is None:
        z_ref = torch.randn_like(z)
    return F.mse_loss(z, z_ref)


def gaussian_anchor_loss(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * z.pow(2).mean()
