from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.inpainting import annulus_mask, annulus_mse, erode_mask, gaussian_anchor_loss, kl_to_standard_normal


def _freeze_generator(G: nn.Module) -> None:
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)


def _kl_diagonal(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (z.pow(2).mean())


def optimize_latent_seed(
    G: nn.Module,
    target: torch.Tensor,
    hole_mask: torch.Tensor,
    *,
    z_init: Optional[torch.Tensor] = None,
    annulus_w: int = 8,
    out_hw: Optional[Tuple[int, int]] = None,
    steps: int = 500,
    lr: float = 0.05,
    kl_weight: float = 0.01,
    use_kl: bool = True,
    kl_diag_weight: float = 0.5,
    interior_weight: float = 0.35,
    border: int = 8,
    grad_clip: float = 3.0,
    lr_min_ratio: float = 0.05,
    renormalize_every: int = 0,
    log_fn: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = target.device
    h, w = target.shape[-2], target.shape[-1]
    if out_hw is None:
        out_hw = (h, w)
    _freeze_generator(G)

    if z_init is None:
        z_init = torch.randn(1, 1, 8, 8, device=device)
    z = nn.Parameter(z_init.clone().to(device).detach())
    opt = torch.optim.Adam([z], lr=lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=lr * lr_min_ratio)

    ring = annulus_mask(hole_mask, annulus_w)
    core = erode_mask(hole_mask, border)

    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        out = G(z, out_hw=out_hw)
        l_ann = annulus_mse(out, target, ring)
        l_int = F.l1_loss(out * core, target * core)
        if use_kl:
            l_kl = kl_to_standard_normal(z)
            l_reg = kl_diag_weight * _kl_diagonal(z) + (1.0 - min(kl_diag_weight, 0.999)) * l_kl
        else:
            l_reg = gaussian_anchor_loss(z)
        loss = l_ann + interior_weight * l_int + kl_weight * l_reg
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([z], grad_clip)
        opt.step()
        sched.step()

        if renormalize_every > 0 and (t + 1) % renormalize_every == 0:
            with torch.no_grad():
                z.sub_(z.mean()).div_(z.std() + 1e-6)

        if log_fn is not None and (t % max(1, steps // 20) == 0 or t == steps - 1):
            log_fn(
                t,
                {
                    "annulus_mse": float(l_ann.detach().item()),
                    "interior_l1": float(l_int.detach().item()),
                    "reg": float(l_reg.detach().item()),
                },
            )

    with torch.no_grad():
        out_final = G(z, out_hw=out_hw)
    return z.detach(), out_final.detach()


def optimize_latent_from_cfg(
    G: nn.Module,
    target: torch.Tensor,
    hole_mask: torch.Tensor,
    z_init: Optional[torch.Tensor],
    zcfg: Dict,
    gopt_cfg: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w = target.shape[-2], target.shape[-1]
    return optimize_latent_seed(
        G,
        target,
        hole_mask,
        z_init=z_init,
        annulus_w=int(gopt_cfg.get("annulus_width", 8)),
        out_hw=(h, w),
        steps=int(zcfg.get("steps", 800)),
        lr=float(zcfg.get("lr", 0.02)),
        kl_weight=float(zcfg.get("kl_weight", 0.035)),
        use_kl=bool(zcfg.get("use_kl", True)),
        kl_diag_weight=float(zcfg.get("kl_diag_weight", 0.55)),
        interior_weight=float(zcfg.get("interior_weight", 0.45)),
        border=int(gopt_cfg.get("annulus_width", 8)),
        grad_clip=float(zcfg.get("grad_clip", 3.0)),
        lr_min_ratio=float(zcfg.get("lr_min_ratio", 0.05)),
        renormalize_every=int(zcfg.get("renormalize_every", 0)),
    )
