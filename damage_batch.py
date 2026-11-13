"""Build a U-Net training set: damaged images + their masks, with VARIED damage.

  python damage_batch.py --clean_dir data/train_images ^
                         --out_images data/unet_images ^
                         --out_masks  data/unet_masks ^
                         --n 1000
"""
import argparse
from pathlib import Path

import numpy as np
import cv2


def rect_mask(shape, rng, lo=32, hi=112):
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    for _ in range(int(rng.integers(1, 3))):          # 1-2 rectangles
        bw = int(rng.integers(lo, hi + 1))
        bh = int(rng.integers(lo, hi + 1))
        x0 = int(rng.integers(0, max(1, w - bw)))
        y0 = int(rng.integers(0, max(1, h - bh)))
        m[y0:y0 + bh, x0:x0 + bw] = 255
    return m


def blob_mask(shape, rng):
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    for _ in range(int(rng.integers(2, 6))):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.circle(m, c, int(rng.integers(12, 40)), 255, -1)
    return m


def scratch_mask(shape, rng):
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    for _ in range(int(rng.integers(2, 6))):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(m, p1, p2, 255, int(rng.integers(3, 9)))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--out_images", default="data/unet_images")
    ap.add_argument("--out_masks", default="data/unet_masks")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--fill", type=int, default=255)
    ap.add_argument("--p_rect", type=float, default=0.6,
                    help="share of rectangles; the GAN only trained on these")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fill_random", action="store_true",
                    help="draw a different grey level per image instead of using --fill. "
                         "Stops brightness being a reliable cue, forcing the U-Net to learn "
                         "that damage is FLAT while real structure carries speckle noise.")
    ap.add_argument("--fill_lo", type=int, default=0)
    ap.add_argument("--fill_hi", type=int, default=255)
    args = ap.parse_args()

    src = sorted(Path(args.clean_dir).glob("*.png"))
    if not src:
        raise SystemExit(f"no .png in {args.clean_dir}")

    di, dm = Path(args.out_images), Path(args.out_masks)
    di.mkdir(parents=True, exist_ok=True)
    dm.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    counts = {"rect": 0, "blob": 0, "scratch": 0}
    fills = []
    manifest = []

    for i in range(args.n):
        f = src[i % len(src)]
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)

        r = rng.random()
        if r < args.p_rect:
            kind, m = "rect", rect_mask(img.shape, rng)
        elif r < args.p_rect + (1 - args.p_rect) / 2:
            kind, m = "blob", blob_mask(img.shape, rng)
        else:
            kind, m = "scratch", scratch_mask(img.shape, rng)
        counts[kind] += 1

        if args.fill_random:
            fill = int(rng.integers(args.fill_lo, args.fill_hi + 1))
        else:
            fill = args.fill
        fills.append(fill)

        # what the damaged region covered up, BEFORE we paint over it --
        # this is what decides whether the damage is visible at all
        sel = m > 0
        bg_mean = float(img[sel].mean())
        bg_std = float(img[sel].std())
        contrast = abs(fill - bg_mean)

        dmg = img.copy()
        dmg[m > 0] = fill

        stem = f"{f.stem}_{i:05d}"
        cv2.imwrite(str(di / f"{stem}.png"), dmg)
        cv2.imwrite(str(dm / f"{stem}.png"), m)

        manifest.append({
            "stem": stem,
            "kind": kind,
            "fill": fill,
            "area_pct": round(100 * float(sel.mean()), 3),
            "bg_mean": round(bg_mean, 1),
            "bg_std": round(bg_std, 1),
            "contrast": round(contrast, 1),
        })

    import csv
    mpath = di.parent / f"{di.name}_manifest.csv"
    with open(mpath, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    frac = []
    for p in sorted(dm.glob("*.png"))[:200]:
        frac.append((cv2.imread(str(p), 0) > 127).mean())
    print(f"{args.n} pairs written")
    print(f"  manifest -> {mpath}")
    low = sum(1 for r in manifest if r["contrast"] < 15)
    print(f"  near-invisible (contrast < 15 grey levels): {low} of {len(manifest)}"
          f"  ({100*low/len(manifest):.0f}%)")
    print(f"  images -> {di}")
    print(f"  masks  -> {dm}")
    print(f"  mix: {counts}")
    if args.fill_random:
        print(f"  fill: RANDOM per image, range [{min(fills)}, {max(fills)}], mean {np.mean(fills):.0f}")
    else:
        print(f"  fill: fixed at {args.fill}")
    print(f"  mean damaged area: {100*np.mean(frac):.1f}%  (min {100*min(frac):.1f}%, max {100*max(frac):.1f}%)")


if __name__ == "__main__":
    main()
