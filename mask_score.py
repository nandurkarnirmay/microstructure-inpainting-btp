"""Compare a predicted mask against the true one.

  python mask_score.py --pred outputs/unet_demo/mask.png --true data/test_demo/x_mask.png
"""
import argparse
from pathlib import Path

import numpy as np
import cv2


def load(path, label):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{label} not found: {p.resolve()}")
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise SystemExit(f"{label} exists but could not be read as an image: {p}")
    print(f"{label:6s} {p}  shape {m.shape}")
    return m > 127


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="mask the U-Net predicted")
    ap.add_argument("--true", required=True, help="mask damage.py wrote")
    args = ap.parse_args()

    a = load(args.pred, "pred")
    b = load(args.true, "true")

    if a.shape != b.shape:
        print(f"resizing pred {a.shape} -> {b.shape}")
        a = cv2.resize(a.astype(np.uint8), (b.shape[1], b.shape[0]),
                       interpolation=cv2.INTER_NEAREST) > 0

    inter = (a & b).sum()
    union = (a | b).sum()
    missed = (b & ~a).sum()      # damage the U-Net failed to mark
    extra = (a & ~b).sum()       # healthy pixels it wrongly marked

    print()
    print(f"Dice            {2*inter/(a.sum()+b.sum()+1e-9):.4f}")
    print(f"IoU             {inter/(union+1e-9):.4f}")
    print(f"predicted area  {100*a.mean():.2f}%")
    print(f"true area       {100*b.mean():.2f}%")
    print()
    print(f"missed damage   {missed:6d} px  ({100*missed/(b.sum()+1e-9):.1f}% of the real hole)")
    print(f"over-marked     {extra:6d} px  ({100*extra/(b.sum()+1e-9):.1f}% of the real hole)")
    if missed > extra:
        print("\n-> under-segmenting: dilate the mask before inpainting")
    elif extra > missed:
        print("\n-> over-segmenting: it is replacing some healthy pixels")


if __name__ == "__main__":
    main()
