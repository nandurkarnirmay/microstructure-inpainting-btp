"""Paired comparison of two batch_eval runs on the same images.

  python paired_test.py --a outputs/voronoi_specialist/per_image.csv ^
                        --b outputs/voronoi_mixed/per_image.csv ^
                        --label_a specialist --label_b mixed
"""
import argparse, csv, math
from collections import defaultdict


def load(path):
    d = defaultdict(dict)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            d[int(r["hole"])][r["image"]] = {k: float(v) for k, v in r.items()
                                             if k not in ("hole", "image")}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--metrics", default="ssim,boundary_f1,s2_mse,phase_error_percent")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    mets = args.metrics.split(",")
    lower_is_better = {"s2_mse", "phase_error_percent", "mae"}

    for hole in sorted(set(A) & set(B)):
        imgs = sorted(set(A[hole]) & set(B[hole]))
        print(f"\nhole {hole}   ({len(imgs)} paired images)")
        for m in mets:
            diffs = [A[hole][i][m] - B[hole][i][m] for i in imgs]
            n = len(diffs)
            mean = sum(diffs) / n
            var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
            se = math.sqrt(var / n)
            t = mean / se if se else 0.0
            base = m[5:] if m.startswith("hole_") else m
            lo = base in lower_is_better
            wins = sum(1 for d in diffs if (d < 0 if lo else d > 0))
            better = args.label_a if (mean < 0 if lo else mean > 0) else args.label_b
            star = "*" if abs(t) > 2.093 else " "   # two-sided 0.05, df=19
            print(f"  {m:22s} mean diff {mean:+.5g}   t = {t:+6.2f} {star}  "
                  f"{args.label_a} wins {wins}/{n}   -> {better}")

    print("\n* marks |t| > 2.09, significant at p < 0.05 for 20 paired samples")


if __name__ == "__main__":
    main()
