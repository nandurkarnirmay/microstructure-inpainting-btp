from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from losses.inpainting import composite
from models.gan import MaskGuidedGenerator, SpatialNoiseGenerator, spatial_seed_size_for_hole
from models.unet import UNet
from optimizers.z_opt import optimize_latent_from_cfg
from preprocessing.mask_generation import MaskGenConfig, logits_to_mask, refine_mask_numpy
from preprocessing.pipeline import MicroscopyPreprocessor, PreprocessConfig


class InpaintPipeline:
    def __init__(self, cfg: Dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        pre = cfg.get("preprocessing", {})
        self.pre = MicroscopyPreprocessor(
            PreprocessConfig(
                image_size=pre.get("image_size"),
                grayscale=bool(pre.get("grayscale", True)),
                normalize=pre.get("normalize", "minmax"),
                clahe_clip=float(pre.get("clahe_clip", 2.0)),
                clahe_grid=int(pre.get("clahe_grid", 8)),
                denoise_h=float(pre.get("denoise_h", 0.0)),
            )
        )
        self.mask_cfg = MaskGenConfig(**cfg.get("mask", {}))
        self.blending = cfg.get("blending", {"mode": "feather", "feather_kernel": 9, "feather_sigma": 2.0})

    def _blend(self, pred: torch.Tensor, image: torch.Tensor, hole: torch.Tensor, border: int) -> torch.Tensor:
        mode = str(self.blending.get("mode", "feather"))
        return composite(
            pred,
            image,
            hole,
            border,
            mode=mode,
            feather_kernel=int(self.blending.get("feather_kernel", 9)),
            feather_sigma=float(self.blending.get("feather_sigma", 2.0)),
        )

    def load_unet(self, path: str | Path, **kwargs) -> UNet:
        m = UNet(**kwargs).to(self.device)
        state = torch.load(path, map_location=self.device, weights_only=False)
        sd = state["model"] if "model" in state else state
        m.load_state_dict(sd, strict=True)
        m.eval()
        return m

    def load_gan(self, path: str | Path, **kwargs) -> MaskGuidedGenerator:
        g = MaskGuidedGenerator(**kwargs).to(self.device)
        obj = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(obj, dict) and "G" in obj:
            g.load_state_dict(obj["G"], strict=True)
        elif isinstance(obj, dict) and "model" in obj:
            g.load_state_dict(obj["model"], strict=True)
        else:
            g.load_state_dict(obj, strict=True)
        g.eval()
        return g

    def load_spatial_g(self, path: str | Path, **kwargs) -> SpatialNoiseGenerator:
        g = SpatialNoiseGenerator(**kwargs).to(self.device)
        obj = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(obj, dict) and "G" in obj:
            g.load_state_dict(obj["G"], strict=True)
        elif isinstance(obj, dict) and "model" in obj:
            g.load_state_dict(obj["model"], strict=True)
        else:
            g.load_state_dict(obj, strict=True)
        g.eval()
        return g

    def preprocess_numpy(self, img) -> torch.Tensor:
        x = self.pre.process_numpy(img)
        return self.pre.to_tensor(x).unsqueeze(0).to(self.device)

    def segment_defects(self, unet: UNet, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = unet(tensor)
        return logits_to_mask(logits, threshold=float(self.mask_cfg.binarize_threshold))

    def refine_mask(self, mask_t: torch.Tensor) -> torch.Tensor:
        m = mask_t.squeeze().cpu().numpy()
        m2 = refine_mask_numpy(m, self.mask_cfg)
        return torch.from_numpy(m2).float().unsqueeze(0).unsqueeze(0).to(self.device)

    def inpaint_masked_gan(self, G: MaskGuidedGenerator, image: torch.Tensor, hole: torch.Tensor) -> torch.Tensor:
        masked = image * (1.0 - hole)
        with torch.no_grad():
            out = G(masked, hole)
        grown = torch.nn.functional.max_pool2d(hole, kernel_size=5, stride=1, padding=2)
        return self._blend(out, image, grown, border=0)

    def inpaint_spatial_gopt(
        self,
        G: SpatialNoiseGenerator,
        image: torch.Tensor,
        hole: torch.Tensor,
        z_fixed: torch.Tensor,
        seed_side: int,
        scale: int = 8,
    ) -> torch.Tensor:
        h, w = image.shape[-2], image.shape[-1]
        with torch.no_grad():
            out = G(z_fixed.to(self.device), out_hw=(h, w))
        return self._blend(out, image, hole, border=int(self.cfg.get("gopt", {}).get("annulus_width", 8)))

    def z_opt_refinement(
        self,
        G: SpatialNoiseGenerator,
        image: torch.Tensor,
        hole: torch.Tensor,
        z_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zcfg = self.cfg.get("zopt", {})
        h, w = image.shape[-2], image.shape[-1]
        hole_sz = int(self.cfg.get("gopt", {}).get("hole_size", 64))
        side = spatial_seed_size_for_hole(hole_sz, int(self.cfg.get("gopt", {}).get("scale_factor", 8)))
        if z_init is None:
            z_init = torch.randn(1, 1, side, side, device=self.device)
        z, out = optimize_latent_from_cfg(G, image, hole, z_init, zcfg, self.cfg.get("gopt", {}))
        fused = self._blend(out, image, hole, border=int(self.cfg.get("gopt", {}).get("annulus_width", 8)))
        return z, fused
