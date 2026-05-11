from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from losses.segmentation import bce_dice_loss
from utils.io import ensure_dir, save_checkpoint


def _dice_metric(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    t = targets.float()
    inter = (probs * t).sum()
    denom = probs.sum() + t.sum() + 1e-8
    return float((2 * inter / denom).item())


def train_unet_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: Dict,
    device: torch.device,
    ckpt_dir: Path,
    log_dir: Path,
) -> None:
    ckpt_dir = ensure_dir(ckpt_dir)
    log_dir = ensure_dir(log_dir)
    writer = SummaryWriter(log_dir)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    epochs = int(cfg["epochs"])
    best = -1.0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"UNet ep{epoch+1}/{epochs}")
        for batch in pbar:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(imgs)
            loss = bce_dice_loss(logits, masks)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n += 1
            pbar.set_postfix(loss=f"{running/max(n,1):.4f}")
        writer.add_scalar("unet/train_loss", running / max(n, 1), epoch)

        if val_loader is not None:
            model.eval()
            dices = []
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(device)
                    masks = batch["mask"].to(device)
                    logits = model(imgs)
                    dices.append(_dice_metric(logits, masks))
            val_dice = sum(dices) / max(len(dices), 1)
            writer.add_scalar("unet/val_dice", val_dice, epoch)
            if val_dice > best:
                best = val_dice
                save_checkpoint(ckpt_dir / "unet_best.pt", model=model, optimizer=opt, epoch=epoch)
        save_checkpoint(ckpt_dir / "unet_last.pt", model=model, optimizer=opt, epoch=epoch)
    writer.close()
