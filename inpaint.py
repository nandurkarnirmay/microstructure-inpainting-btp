"""Run the trained generator directly, with YOUR mask. Bypasses infer.py.

  python inpaint.py --ckpt checkpoints/gan/gan_weights.pt ^
                    --image data/test/x_damaged.png ^
                    --mask data/test/x_mask.png ^
                    --out outputs/run3/final.png
"""
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch

from models.gan import MaskGuidedGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True, help="damaged image")
    ap.add_argument("--mask", required=True, help="white = hole")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--feather", type=int, default=2)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- load weights -------------------------------------------------
    sd = torch.load(args.ckpt, map_location=dev)
    if isinstance(sd, dict) and "G" in sd:
        sd = sd["G"]
    G = MaskGuidedGenerator().to(dev)
    G.load_state_dict(sd)
    G.eval()
    print(f"loaded generator from {args.ckpt}")

    # ---- load image and mask ------------------------------------------
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        raise SystemExit("could not read image or mask")

    img = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)
    msk = cv2.resize(msk, (args.size, args.size), interpolation=cv2.INTER_NEAREST)

    x = img.astype(np.float32) / 255.0
    m = (msk > 127).astype(np.float32)
    print(f"hole covers {100*m.mean():.2f}% of the image")

    # masked input: hole zeroed out
    masked = x * (1.0 - m)

    t_masked = torch.from_numpy(masked)[None, None].to(dev)
    t_mask = torch.from_numpy(m)[None, None].to(dev)

    with torch.no_grad():
        pred = G(t_masked, t_mask)[0, 0].cpu().numpy()
    print(f"generator output range [{pred.min():.3f}, {pred.max():.3f}]")

    # ---- composite: real pixels outside, prediction inside --------------
    alpha = m.copy()
    if args.feather > 0:
        k = args.feather * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), args.feather / 2.0)
    final = x * (1 - alpha) + pred * alpha

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), (final * 255).clip(0, 255).astype(np.uint8))

    # side-by-side for inspection
    strip = np.hstack([x, masked, pred, final])
    cv2.imwrite(str(out.parent / "strip.png"), (strip * 255).clip(0, 255).astype(np.uint8))
    print(f"wrote {out}")
    print(f"wrote {out.parent/'strip.png'}  (damaged | masked input | raw G output | composited)")


if __name__ == "__main__":
    main()
