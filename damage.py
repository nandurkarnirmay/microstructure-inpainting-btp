"""Damage an image for an inpainting demo, keeping the clean original and the mask.

Default mode is a CENTRAL SQUARE, which matches the placeholder mask infer.py
uses when no --unet_ckpt is supplied.

  python damage.py --image clean.png --out_dir data/test --mode center --hole 64
"""
import argparse
from pathlib import Path

import numpy as np
import cv2


def central_square(shape, hole):
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    y0, x0 = (h - hole) // 2, (w - hole) // 2
    m[y0:y0 + hole, x0:x0 + hole] = 255
    return m


def scratches(shape, rng, n=6, thickness=4):
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    for _ in range(n):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(m, p1, p2, 255, int(rng.integers(2, thickness + 1)))
    return m


def blobs(shape, rng, n=5, rmax=22):
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    for _ in range(n):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.circle(m, c, int(rng.integers(6, rmax)), 255, -1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_dir", default="data/test")
    ap.add_argument("--mode", choices=["center", "scratch", "blob"], default="center")
    ap.add_argument("--hole", type=int, default=64, help="central square side")
    ap.add_argument("--size", type=int, default=256, help="resize to this; 0 = keep")
    ap.add_argument("--fill", type=int, default=255, help="0=black 255=white")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    if args.size:
        img = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)

    rng = np.random.default_rng(args.seed)
    if args.mode == "center":
        mask = central_square(img.shape, args.hole)
    elif args.mode == "scratch":
        mask = scratches(img.shape, rng)
    else:
        mask = blobs(img.shape, rng)

    damaged = img.copy()
    damaged[mask > 0] = args.fill

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    cv2.imwrite(str(out / f"{stem}_clean.png"), img)
    cv2.imwrite(str(out / f"{stem}_damaged.png"), damaged)
    cv2.imwrite(str(out / f"{stem}_mask.png"), mask)

    pct = 100.0 * (mask > 0).sum() / mask.size
    print(f"{stem}: {pct:.1f}% of pixels removed ({args.mode})")
    print(f"  clean   -> {out/f'{stem}_clean.png'}")
    print(f"  damaged -> {out/f'{stem}_damaged.png'}")
    print(f"  mask    -> {out/f'{stem}_mask.png'}")


if __name__ == "__main__":
    main()
