#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
import torch

from inference.pipeline import InpaintPipeline
from preprocessing.mask_generation import MaskGenConfig, logits_to_mask, refine_mask_numpy
from utils.config import load_config


def build_demo(config_path: str):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = InpaintPipeline(cfg, device)
    pre = pipe.pre
    unet = None
    G = None

    def load_models(unet_path: str | None, gan_path: str | None):
        nonlocal unet, G
        unet_path = (unet_path or "").strip() or None
        gan_path = (gan_path or "").strip() or None
        if unet_path:
            unet = pipe.load_unet(
                unet_path,
                in_channels=int(cfg.get("unet", {}).get("in_channels", 1)),
                base_channels=int(cfg.get("unet", {}).get("base_channels", 32)),
                num_levels=int(cfg.get("unet", {}).get("num_levels", 4)),
            )
        if gan_path:
            G = pipe.load_gan(
                gan_path,
                in_channels=int(cfg.get("gan", {}).get("in_channels", 2)),
                out_channels=int(cfg.get("gan", {}).get("out_channels", 1)),
                base=int(cfg.get("gan", {}).get("base_channels", 48)),
            )

    def run(image, unet_path: str, gan_path: str):
        if image is None:
            return None, None, "no image"
        load_models(unet_path or None, gan_path or None)
        import numpy as np

        arr = np.array(image.convert("RGB"))
        if cfg.get("preprocessing", {}).get("grayscale", True):
            import cv2

            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)[..., None]
        arr = arr.astype("float32")
        if arr.max() > 1.5:
            arr = arr / 255.0
        x = pre.process_numpy(arr)
        t = pre.to_tensor(x).unsqueeze(0).to(device)
        if unet is None or G is None:
            return None, None, "Provide U-Net and GAN checkpoints"
        with torch.no_grad():
            logits = unet(t)
            m = logits_to_mask(logits, threshold=0.5)
        m_np = m.squeeze().cpu().numpy()
        m_ref = torch.from_numpy(refine_mask_numpy(m_np, MaskGenConfig())).float().unsqueeze(0).unsqueeze(0).to(device)
        out = pipe.inpaint_masked_gan(G, t, m_ref)
        out_np = (out.squeeze().cpu().clamp(0, 1).numpy() * 255).astype("uint8")
        mask_np = (m_ref.squeeze().cpu().numpy() * 255).astype("uint8")
        return out_np, mask_np, "ok"

    return run


def main():
    config_path = str(ROOT / "configs" / "default.yaml")
    runner = build_demo(config_path)
    with gr.Blocks() as demo:
        gr.Markdown("## Microscopy inpainting (U-Net mask + mask-guided GAN)")
        with gr.Row():
            img = gr.Image(type="pil", label="Micrograph")
            out = gr.Image(label="Restored")
            msk = gr.Image(label="Predicted mask")
        unet_path = gr.Text(label="U-Net checkpoint (.pt)")
        gan_path = gr.Text(label="GAN weights (.pt)")
        status = gr.Text(label="status")
        btn = gr.Button("Run")
        btn.click(runner, inputs=[img, unet_path, gan_path], outputs=[out, msk, status])
    demo.launch()


if __name__ == "__main__":
    main()
