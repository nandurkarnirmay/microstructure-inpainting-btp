"""Keep the real pixels, take the network's output only inside the hole.

  python composite.py --pred outputs/demo/inpaint_gan.png ^
                      --damaged data/test/x_damaged.png ^
                      --mask data/test/x_mask.png ^
                      --out outputs/demo/final.png
"""
import argparse
from pathlib import Path

import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="network output")
    ap.add_argument("--damaged", required=True, help="the damaged input")
    ap.add_argument("--mask", required=True, help="white = hole")
    ap.add_argument("--out", required=True)
    ap.add_argument("--feather", type=int, default=3,
                    help="soften the join over this many pixels; 0 = hard edge")
    args = ap.parse_args()

    pred = cv2.imread(args.pred, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    dmg = cv2.imread(args.damaged, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    m = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

    if pred.shape != dmg.shape:
        pred = cv2.resize(pred, (dmg.shape[1], dmg.shape[0]))
    if m.shape != dmg.shape:
        m = cv2.resize(m, (dmg.shape[1], dmg.shape[0]), interpolation=cv2.INTER_NEAREST)

    alpha = (m > 127).astype(np.float32)
    if args.feather > 0:
        k = args.feather * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), args.feather / 2.0)

    out = dmg * (1 - alpha) + pred * alpha
    cv2.imwrite(args.out, out.clip(0, 255).astype(np.uint8))

    inside = alpha > 0.5
    print(f"composited -> {args.out}")
    print(f"  {inside.sum()} px taken from the network ({100*inside.mean():.1f}%)")
    print(f"  {(~inside).sum()} px kept from the original")


if __name__ == "__main__":
    main()
