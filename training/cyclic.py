from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from losses.inpainting import composite


def cyclic_inpaint_round(
    image: torch.Tensor,
    mask: torch.Tensor,
    G_msk: nn.Module,
    G_spatial: nn.Module | None,
    unet: nn.Module,
    *,
    max_rounds: int = 2,
    defect_threshold: float = 0.15,
    border: int = 8,
    use_spatial: bool = False,
    z_bundle: Dict | None = None,
    blend_mode: str = "feather",
    feather_kernel: int = 9,
    feather_sigma: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    x = image.clone()
    m = mask.clone()
    if float(m.sum()) < 1e-6:
        return x, m, 0
    rounds = 0
    for r in range(max_rounds):
        rounds = r + 1
        if use_spatial and G_spatial is not None and z_bundle is not None:
            z = z_bundle.get("z")
            if z is None:
                raise ValueError("z_bundle must contain 'z' for spatial path")
            out = G_spatial(z.to(x.device), out_hw=x.shape[-2:])
            pred = composite(
                out,
                x,
                m,
                border,
                mode=blend_mode,
                feather_kernel=feather_kernel,
                feather_sigma=feather_sigma,
            )
        else:
            masked = x * (1.0 - m)
            pred = G_msk(masked, m)
            pred = composite(
                pred,
                x,
                m,
                border,
                mode=blend_mode,
                feather_kernel=feather_kernel,
                feather_sigma=feather_sigma,
            )

        with torch.no_grad():
            logits = unet(pred)
            probs = torch.sigmoid(logits)
            residual = (probs * (1.0 - m)).mean()
        if residual.item() < defect_threshold:
            return pred, m, rounds
        m = torch.clamp(m + (probs > 0.5).float() * (1.0 - m), 0.0, 1.0)
        x = pred
    return pred, m, rounds
