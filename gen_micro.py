"""Generate synthetic microstructure images for inpainting experiments.

Writes 256x256 grayscale PNGs: spinodal (bicontinuous) and Voronoi (grain) structures.
"""
import argparse
from pathlib import Path

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, distance_transform_edt


def spinodal(size, rng, corr_len=14.0, volume_fraction=0.5, smooth=1.0):
    """Gaussian random field thresholded at a chosen volume fraction."""
    field = rng.standard_normal((size, size))
    field = gaussian_filter(field, sigma=corr_len / 3.0, mode="wrap")
    thresh = np.quantile(field, 1.0 - volume_fraction)
    binary = (field > thresh).astype(np.float32)
    if smooth > 0:
        binary = gaussian_filter(binary, sigma=smooth, mode="wrap")
    return binary


def voronoi(size, rng, n_seeds=40, wall=1.6):
    """Grain structure: bright grains separated by dark boundaries."""
    seeds = rng.integers(0, size, size=(n_seeds, 2))
    pts = np.zeros((size, size), dtype=np.uint8)
    pts[seeds[:, 0], seeds[:, 1]] = 1
    # label each pixel by nearest seed
    _, labels = distance_transform_edt(1 - pts, return_indices=True)
    lab = labels[0] * size + labels[1]
    # boundaries = where the nearest-seed label changes
    gx = np.abs(np.diff(lab, axis=0, prepend=lab[:1]))
    gy = np.abs(np.diff(lab, axis=1, prepend=lab[:, :1]))
    edges = ((gx + gy) > 0).astype(np.float32)
    edges = gaussian_filter(edges, sigma=wall)
    img = 1.0 - np.clip(edges * 3.0, 0, 1)
    # per-grain intensity variation, like a real micrograph
    tint = rng.uniform(0.72, 1.0, size=n_seeds + 1)
    idx = np.searchsorted(np.unique(lab), lab)
    img = img * tint[idx % (n_seeds + 1)]
    return img.astype(np.float32)


def add_realism(img, rng, noise=0.035, shading=0.12):
    """Light sensor noise plus a smooth illumination gradient."""
    s = img.shape[0]
    yy, xx = np.mgrid[0:s, 0:s] / s
    a, b = rng.uniform(-1, 1, 2)
    img = img * (1.0 - shading * (a * xx + b * yy))
    img = img + rng.standard_normal(img.shape) * noise
    img = (img - img.min()) / (np.ptp(img) + 1e-8)
    return (img * 255).clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/train_images")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--kind", choices=["spinodal", "voronoi", "both"], default="both")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    kinds = ["spinodal", "voronoi"] if args.kind == "both" else [args.kind]
    per = args.n // len(kinds)

    written = 0
    for kind in kinds:
        for i in range(per):
            if kind == "spinodal":
                img = spinodal(
                    args.size, rng,
                    corr_len=rng.uniform(8, 22),
                    volume_fraction=rng.uniform(0.35, 0.65),
                )
            else:
                img = voronoi(
                    args.size, rng,
                    n_seeds=int(rng.integers(18, 70)),
                    wall=rng.uniform(1.0, 2.4),
                )
            img = add_realism(img, rng)
            cv2.imwrite(str(out / f"{kind}_{i:05d}.png"), img)
            written += 1
            if written % 250 == 0:
                print(f"  {written} written", flush=True)

    print(f"{written} images written to {out}")


if __name__ == "__main__":
    main()
