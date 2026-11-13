"""Classical damage detector: find regions with no texture.

Painted damage is perfectly uniform; real microstructure always carries
speckle noise. So local standard deviation alone separates them -- no
training, no weights, no GPU.

Single image (writes a mask infer.py-style):
  python var_detect.py --image data/test/x_damaged.png --out_dir outputs/var/x

Whole folder, scored against ground truth:
  python var_detect.py --images data/unet_vtest_images ^
                       --true   data/unet_vtest_masks ^
                       --out_root outputs/var_vtest
"""
import argparse
from pathlib import Path

import numpy as np
import cv2


def detect(img, win=5, thr=3.0, min_area=40):
    """White where the image is locally flat, i.e. painted over."""
    im = img.astype(np.float32)
    mu = cv2.blur(im, (win, win))
    var = np.maximum(cv2.blur(im * im, (win, win)) - mu * mu, 0.0)
    m = (np.sqrt(var) < thr).astype(np.uint8)

    # close pinholes, then drop specks too small to be real damage
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 255
    return out


def dice(a, b):
    a, b = a > 127, b > 127
    return 2 * (a & b).sum() / (a.sum() + b.sum() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="single damaged image")
    ap.add_argument("--out_dir", help="where to write mask.png for --image")
    ap.add_argument("--images", help="folder of damaged images")
    ap.add_argument("--true", help="folder of ground-truth masks (optional)")
    ap.add_argument("--out_root", help="write one sub-folder per image, like infer.py")
    ap.add_argument("--win", type=int, default=5)
    ap.add_argument("--thr", type=float, default=3.0)
    ap.add_argument("--min_area", type=int, default=40)
    args = ap.parse_args()

    if args.image:
        img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"could not read {args.image}")
        m = detect(img, args.win, args.thr, args.min_area)
        out = Path(args.out_dir or ".")
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "mask.png"), m)
        print(f"wrote {out/'mask.png'}   marked {100*(m>127).mean():.2f}% of the image")
        return

    if not args.images:
        raise SystemExit("give either --image or --images")

    scores = []
    for p in sorted(Path(args.images).glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        m = detect(img, args.win, args.thr, args.min_area)
        if args.out_root:
            d = Path(args.out_root) / p.stem
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / "mask.png"), m)
        if args.true:
            gt = cv2.imread(str(Path(args.true) / p.name), cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                scores.append(dice(m, gt))

    print(f"{len(list(Path(args.images).glob('*.png')))} images processed")
    if scores:
        s = np.array(scores)
        print(f"mean Dice {s.mean():.4f}   median {np.median(s):.4f}   "
              f"min {s.min():.3f}   below 0.5: {(s < 0.5).sum()}/{len(s)}")


if __name__ == "__main__":
    main()
