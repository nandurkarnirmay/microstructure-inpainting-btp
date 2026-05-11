from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from losses.adversarial import gradient_penalty_wgan, wasserstein_disc_loss, wasserstein_gen_loss
from losses.inpainting import annulus_mask, annulus_mse, erode_mask
from models.gan import spatial_seed_size_for_hole
from utils.io import ensure_dir, save_checkpoint


def make_center_hole_mask(b: int, c: int, h: int, w: int, hole: int, device: torch.device) -> torch.Tensor:
    m = torch.zeros(b, c, h, w, device=device)
    hh, ww = hole, hole
    y0 = (h - hh) // 2
    x0 = (w - ww) // 2
    m[:, :, y0 : y0 + hh, x0 : x0 + ww] = 1.0
    return m


def train_gopt_loop(
    G: nn.Module,
    D: nn.Module,
    train_loader: DataLoader,
    cfg: Dict,
    device: torch.device,
    ckpt_dir: Path,
    log_dir: Path,
) -> Tuple[torch.Tensor, dict]:
    ckpt_dir = ensure_dir(ckpt_dir)
    log_dir = ensure_dir(log_dir)
    writer = SummaryWriter(log_dir)

    hole = int(cfg["hole_size"])
    annulus_w = int(cfg["annulus_width"])
    content_w = float(cfg["content_weight"])
    every = int(cfg["content_every"])
    gp_w = float(cfg["gp_weight"])
    scale = int(cfg.get("scale_factor", 8))
    epochs = int(cfg["epochs"])

    opt_g = torch.optim.Adam(G.parameters(), lr=float(cfg["lr_g"]), betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=float(cfg["lr_d"]), betas=(0.5, 0.999))
    grad_clip = float(cfg.get("grad_clip", 1.0))

    first_batch = next(iter(train_loader))
    img0 = first_batch["image"][:1].to(device)
    _, _, H, W = img0.shape
    out_hw = (H, W)
    if hole > min(H, W) - 4:
        raise ValueError(f"gopt hole_size {hole} must be smaller than image min side {min(H, W)}")
    seed_side = spatial_seed_size_for_hole(hole, scale)
    z_fixed = torch.randn(1, 1, seed_side, seed_side, device=device)

    global_step = 0
    for epoch in range(epochs):
        G.train()
        D.train()
        pbar = tqdm(train_loader, desc=f"G-opt ep{epoch+1}/{epochs}")
        for i, batch in enumerate(pbar):
            real = batch["image"].to(device)
            b, _, h, w = real.shape
            hole_m_b = make_center_hole_mask(b, 1, h, w, hole, device)

            z_rand = torch.randn(b, 1, seed_side, seed_side, device=device)
            with torch.no_grad():
                fake_det = G(z_rand, out_hw=(h, w))
            d_real = D(real, hole_m_b)
            d_fake = D(fake_det, hole_m_b)
            loss_d = wasserstein_disc_loss(d_real, d_fake) + gradient_penalty_wgan(D, real, fake_det, hole_m_b, gp_w)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(D.parameters(), grad_clip)
            opt_d.step()

            z_rand = torch.randn(b, 1, seed_side, seed_side, device=device)
            fake = G(z_rand, out_hw=(h, w))
            loss_g = wasserstein_gen_loss(D(fake, hole_m_b))

            if i % every == 0:
                fixed = G(z_fixed, out_hw=(h, w))
                ring_b = annulus_mask(hole_m_b, annulus_w)
                l_cl = annulus_mse(fixed, real, ring_b)
                loss_g = loss_g + content_w * l_cl
                iw = float(cfg.get("interior_content_weight", 0.0))
                if iw > 0.0:
                    core = erode_mask(hole_m_b, annulus_w)
                    loss_g = loss_g + iw * F.l1_loss(fixed * core, real * core)

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(G.parameters(), grad_clip)
            opt_g.step()

            writer.add_scalars("gopt/loss", {"D": float(loss_d.detach().item()), "G": float(loss_g.detach().item())}, global_step)
            pbar.set_postfix(D=f"{loss_d.item():.3f}", G=f"{loss_g.item():.3f}")
            global_step += 1

        save_checkpoint(
            ckpt_dir / "gopt_last.pt",
            model=G,
            optimizer=opt_g,
            epoch=epoch,
            extra={
                "D": D.state_dict(),
                "opt_d": opt_d.state_dict(),
                "z_fixed": z_fixed.detach().cpu(),
                "hole": hole,
                "annulus_w": annulus_w,
                "seed_side": seed_side,
                "out_hw": out_hw,
                "scale_factor": scale,
            },
        )

    writer.close()
    meta = {
        "z_fixed": z_fixed.detach().cpu(),
        "hole": hole,
        "annulus_w": annulus_w,
        "seed_side": int(seed_side),
        "scale_factor": int(scale),
    }
    torch.save({**meta, "G": G.state_dict(), "D": D.state_dict()}, ckpt_dir / "gopt_bundle.pt")
    return z_fixed.detach(), meta
