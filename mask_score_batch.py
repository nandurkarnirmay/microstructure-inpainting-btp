"""Score two detectors over a folder of test images, head to head.

  python mask_score_batch.py --a outputs/vtest_new --b outputs/vtest_old ^
                             --true data/unet_vtest_masks ^
                             --label_a random-fill --label_b fixed-fill

Each --a / --b root may hold either:
    <root>/<stem>/mask.png      (one sub-folder per image, as infer.py writes)
    <root>/<stem>.png           (flat)
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


def load_bin(path: Path, shape=None):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if shape is not None and m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def score(pred, true):
    inter = (pred & true).sum()
    dice = 2 * inter / (pred.sum() + true.sum() + 1e-9)
    iou = inter / ((pred | true).sum() + 1e-9)
    missed = (true & ~pred).sum()
    extra = (pred & ~true).sum()
    denom = true.sum() + 1e-9
    return dice, iou, 100 * missed / denom, 100 * extra / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="predictions from detector A")
    ap.add_argument("--b", help="predictions from detector B (optional)")
    ap.add_argument("--true", required=True, help="folder of ground-truth masks")
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--manifest", help="the *_manifest.csv damage_batch.py wrote; "
                                       "enables the breakdown by damage type and contrast")
    args = ap.parse_args()

    info = {}
    if args.manifest:
        import csv
        with open(args.manifest) as fh:
            for row in csv.DictReader(fh):
                info[row["stem"]] = row

    truth_dir = Path(args.true)
    root_a = Path(args.a)
    root_b = Path(args.b) if args.b else None

    truths = sorted(truth_dir.glob("*.png"))
    if not truths:
        raise SystemExit(f"no masks in {truth_dir.resolve()}")

    rows = []
    skipped = []
    for t in truths:
        gt = load_bin(t)
        if gt is None:
            skipped.append((t.stem, "unreadable truth"))
            continue

        pa = find_pred(root_a, t.stem)
        if pa is None:
            skipped.append((t.stem, f"no prediction under {root_a}"))
            continue
        ma = load_bin(pa, gt.shape)

        mb = None
        if root_b is not None:
            pb = find_pred(root_b, t.stem)
            if pb is None:
                skipped.append((t.stem, f"no prediction under {root_b}"))
                continue
            mb = load_bin(pb, gt.shape)

        rows.append((t.stem, score(ma, gt), score(mb, gt) if mb is not None else None))

    if not rows:
        raise SystemExit("nothing scored -- check the folder names and that infer.py wrote mask.png")

    two = rows[0][2] is not None
    print()
    if two:
        print(f"{'image':<28}{args.label_a:>14}{args.label_b:>14}")
        for stem, sa, sb in rows:
            print(f"{stem:<28}{sa[0]:>14.4f}{sb[0]:>14.4f}")
    else:
        print(f"{'image':<28}{'dice':>10}{'missed%':>10}{'extra%':>10}")
        for stem, sa, _ in rows:
            print(f"{stem:<28}{sa[0]:>10.4f}{sa[2]:>10.1f}{sa[3]:>10.1f}")

    da = np.array([r[1] for r in rows])
    print()
    print(f"n = {len(rows)} images")
    print(f"{args.label_a:<14} dice {da[:,0].mean():.4f}   iou {da[:,1].mean():.4f}   "
          f"missed {da[:,2].mean():.1f}%   over-marked {da[:,3].mean():.1f}%")

    if two:
        db = np.array([r[2] for r in rows])
        print(f"{args.label_b:<14} dice {db[:,0].mean():.4f}   iou {db[:,1].mean():.4f}   "
              f"missed {db[:,2].mean():.1f}%   over-marked {db[:,3].mean():.1f}%")
        diff = da[:, 0] - db[:, 0]
        wins = int((diff > 0).sum())
        print()
        print(f"dice difference ({args.label_a} - {args.label_b}): "
              f"mean {diff.mean():+.4f}, {args.label_a} wins {wins}/{len(diff)}")
        if len(diff) > 1 and diff.std(ddof=1) > 0:
            t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
            print(f"paired t = {t:+.2f}  (|t| > 2.1 is significant at n = 20)")

    if info:
        def group(key_fn, title):
            buckets = {}
            for i, (stem, sa, sb) in enumerate(rows):
                meta = info.get(stem)
                if meta is None:
                    continue
                k = key_fn(meta)
                buckets.setdefault(k, []).append((sa[0], sb[0] if sb else None, sa[2]))
            if not buckets:
                return
            print(f"\n{title}")
            head = f"  {'group':<22}{'n':>4}{args.label_a:>14}"
            if two:
                head += f"{args.label_b:>14}"
            head += f"{'missed%':>10}"
            print(head)
            for k in sorted(buckets):
                v = buckets[k]
                line = f"  {str(k):<22}{len(v):>4}{np.mean([x[0] for x in v]):>14.4f}"
                if two:
                    line += f"{np.mean([x[1] for x in v]):>14.4f}"
                line += f"{np.mean([x[2] for x in v]):>10.1f}"
                print(line)

        group(lambda m: m["kind"], "by damage type")

        def cbucket(m):
            c = float(m["contrast"])
            if c < 10:
                return "contrast <10 (invisible)"
            if c < 25:
                return "contrast 10-25 (faint)"
            if c < 60:
                return "contrast 25-60"
            return "contrast >60 (obvious)"

        group(cbucket, "by how visible the damage is")

    if skipped:
        print(f"\nskipped {len(skipped)}:")
        for stem, why in skipped[:10]:
            print(f"  {stem}: {why}")


if __name__ == "__main__":
    main()
