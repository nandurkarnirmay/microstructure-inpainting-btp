"""Contact sheet: damaged input | true mask | predicted mask | overlay.

  python mask_sheet.py --images data/unet_vtest_images ^
                       --true   data/unet_vtest_masks ^
                       --pred   outputs/vtest_new ^
                       --out    outputs/sheet_new.png

Overlay colours:  GREEN = found,  RED = missed,  BLUE = wrongly marked.
So a good detector gives a green patch; a timid one gives red; a
hallucinating one gives blue somewhere unrelated.
"""
import argparse
from pathlib import Path

import numpy as np
import cv2


def find_pred(root: Path, stem: str):
    for cand in (root / stem / "mask.png", root / f"{stem}.png"):
        if cand.exists():
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--true", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8, help="how many rows")
    ap.add_argument("--tile", type=int, default=200)
    ap.add_argument("--worst", action="store_true",
                    help="show the worst-scoring images instead of the first n")
    args = ap.parse_args()

    img_dir, true_dir, pred_root = Path(args.images), Path(args.true), Path(args.pred)
    stems = [p.stem for p in sorted(true_dir.glob("*.png"))]
    if not stems:
        raise SystemExit(f"no masks in {true_dir.resolve()}")

    items = []
    for s in stems:
        ip = img_dir / f"{s}.png"
        tp = true_dir / f"{s}.png"
        pp = find_pred(pred_root, s)
        if not (ip.exists() and tp.exists() and pp):
            continue
        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(tp), cv2.IMREAD_GRAYSCALE) > 127
        pr = cv2.imread(str(pp), cv2.IMREAD_GRAYSCALE)
        if pr.shape != gt.shape:
            pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        pr = pr > 127
        dice = 2 * (pr & gt).sum() / (pr.sum() + gt.sum() + 1e-9)
        items.append((s, img, gt, pr, dice))

    if not items:
        raise SystemExit("nothing matched -- check --pred points at the folder of per-image sub-folders")

    if args.worst:
        items.sort(key=lambda t: t[4])
    items = items[:args.n]

    T = args.tile
    rows = []
    for s, img, gt, pr, dice in items:
        g = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        t = cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        p = cv2.cvtColor((pr * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        ov = cv2.cvtColor((img * 0.45).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        ov[gt & pr] = (0, 220, 0)        # found
        ov[gt & ~pr] = (0, 0, 230)       # missed
        ov[~gt & pr] = (230, 120, 0)     # false alarm

        tiles = [cv2.resize(x, (T, T), interpolation=cv2.INTER_NEAREST)
                 for x in (g, t, p, ov)]
        row = np.hstack(tiles)
        cv2.putText(row, f"{s}  dice {dice:.3f}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)

    sheet = np.vstack(rows)
    header = np.full((26, sheet.shape[1], 3), 24, np.uint8)
    for i, lab in enumerate(["damaged input", "true mask", "predicted", "overlay"]):
        cv2.putText(header, lab, (i * T + 6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    sheet = np.vstack([header, sheet])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out}  ({len(items)} rows)")
    print("green = found,  red = missed,  blue = wrongly marked")


if __name__ == "__main__":
    main()
