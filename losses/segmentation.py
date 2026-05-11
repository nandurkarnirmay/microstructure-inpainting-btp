import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    t = targets.float()
    inter = (probs * t).sum(dim=(1, 2, 3))
    denom = probs.pow(2).sum(dim=(1, 2, 3)) + t.pow(2).sum(dim=(1, 2, 3)) + eps
    return 1.0 - (2.0 * inter + eps) / denom


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets.float())
    dice = dice_loss(logits, targets).mean()
    return bce_weight * bce + (1.0 - bce_weight) * dice
