from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from losses.adversarial import gradient_penalty_wgan, wasserstein_disc_loss, wasserstein_gen_loss
from utils.io import ensure_dir, save_checkpoint


def train_gan_loop(
    G: nn.Module,
    D: nn.Module,
    train_loader: DataLoader,
    cfg: Dict,
    device: torch.device,
    ckpt_dir: Path,
    log_dir: Path,
) -> None:
    ckpt_dir = ensure_dir(ckpt_dir)
    log_dir = ensure_dir(log_dir)
    writer = SummaryWriter(log_dir)
    opt_g = torch.optim.Adam(G.parameters(), lr=float(cfg["lr_g"]), betas=(float(cfg.get("beta1", 0.5)), 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=float(cfg["lr_d"]), betas=(float(cfg.get("beta1", 0.5)), 0.999))
    lambda_l1 = float(cfg.get("lambda_l1", 10.0))
    lambda_l1_bg = float(cfg.get("lambda_l1_bg", 0.08))
    gp_w = float(cfg.get("gp_weight", 10.0))
    n_critic = int(cfg.get("n_critic", 1))
    epochs = int(cfg["epochs"])
    grad_clip = float(cfg.get("grad_clip", 1.0))

    global_step = 0
    for epoch in range(epochs):
        G.train()
        D.train()
        pbar = tqdm(train_loader, desc=f"GAN ep{epoch+1}/{epochs}")
        for batch in pbar:
            real = batch["image"].to(device)
            hole = batch["mask"].to(device)
            masked = batch["masked"].to(device)

            for _ in range(n_critic):
                with torch.no_grad():
                    fake = G(masked, hole)
                d_real = D(real, hole)
                d_fake = D(fake.detach(), hole)
                loss_d = wasserstein_disc_loss(d_real, d_fake) + gradient_penalty_wgan(D, real, fake.detach(), hole, gp_w)
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(D.parameters(), grad_clip)
                opt_d.step()

            fake = G(masked, hole)
            d_fake_g = D(fake, hole)
            loss_g_adv = wasserstein_gen_loss(d_fake_g)
            loss_l1 = torch.mean(torch.abs((fake - real) * hole)) * lambda_l1
            loss_bg = torch.mean(torch.abs((fake - real) * (1.0 - hole))) * lambda_l1_bg
            loss_g = loss_g_adv + loss_l1 + loss_bg
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(G.parameters(), grad_clip)
            opt_g.step()

            writer.add_scalars("gan/loss", {"D": float(loss_d.item()), "G": float(loss_g.item())}, global_step)
            pbar.set_postfix(D=f"{loss_d.item():.3f}", G=f"{loss_g.item():.3f}")
            global_step += 1

        save_checkpoint(
            ckpt_dir / "gan_last.pt",
            model=G,
            optimizer=opt_g,
            epoch=epoch,
            extra={"D": D.state_dict(), "opt_d": opt_d.state_dict()},
        )
    torch.save({"G": G.state_dict(), "D": D.state_dict()}, ckpt_dir / "gan_weights.pt")
    writer.close()
