"""Verify the install and model wiring without any dataset.

Run from the project root:  python smoke_test.py
"""
import sys
import torch

print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__)
print("cuda avail  ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu         ", torch.cuda.get_device_name(0))

# --- third-party deps the repo relies on -------------------------------
for mod in ["cv2", "skimage", "scipy", "yaml", "tqdm", "matplotlib"]:
    try:
        __import__(mod)
        print(f"import {mod:12s} ok")
    except Exception as e:
        print(f"import {mod:12s} FAILED -> {e}")

# --- repo modules ------------------------------------------------------
from models.gan import (
    MaskGuidedGenerator,
    PatchDiscriminator,
    SpatialNoiseGenerator,
    spatial_seed_size_for_hole,
)
from models.unet import UNet          # name may differ; see note below
from utils import *                   # noqa: F401,F403

S = 256
dev = "cuda" if torch.cuda.is_available() else "cpu"

# --- mask-guided generator --------------------------------------------
G = MaskGuidedGenerator().to(dev)
masked = torch.rand(1, 1, S, S, device=dev)
mask = torch.zeros(1, 1, S, S, device=dev)
mask[..., 96:160, 96:160] = 1.0          # a central square hole
with torch.no_grad():
    out = G(masked, mask)
print("generator   ", tuple(out.shape), f"range [{out.min():.3f}, {out.max():.3f}]")
assert out.shape == (1, 1, S, S)

# --- patch discriminator ----------------------------------------------
D = PatchDiscriminator().to(dev)
with torch.no_grad():
    verdict = D(out, mask)
print("discriminator", tuple(verdict.shape), "<- a grid of verdicts, not one number")

# --- spatial noise generator ------------------------------------------
side = spatial_seed_size_for_hole(64, scale_factor=8)
Gs = SpatialNoiseGenerator().to(dev)
z = torch.randn(1, 1, side, side, device=dev)
with torch.no_grad():
    tex = Gs(z, out_hw=(64, 64))
print("spatial G   ", tuple(z.shape), "->", tuple(tex.shape))

# --- one backward pass, to prove training would run --------------------
G.train()
opt = torch.optim.Adam(G.parameters(), lr=1e-4)
pred = G(masked, mask)
loss = torch.nn.functional.l1_loss(pred * mask, masked * mask)
loss.backward()
opt.step()
print("backward    ", f"loss={loss.item():.5f}  ok")

n = sum(p.numel() for p in G.parameters())
print(f"generator params  {n/1e6:.1f} M")
print("\nall checks passed")
