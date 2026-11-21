r"""How large a hole can the GAN fill before the structure breaks down?

Cuts a centred square hole of each size into every clean test image, repairs
it with the true mask (so only the repair is tested), and scores the result
against the clean original. Same preprocessing and composite as the fixed
infer.py, via e2e_eval.py (keep both files in the project root).

  python hole_sweep.py --gan_ckpt checkpoints\gan_voronoi\gan_weights.pt ^
      --clean_dir data\test_voronoi --sizes 32 48 64 96 128 160 --n 20

Methods: GAN, OpenCV Navier-Stokes, OpenCV Telea (no training), and SHUFFLE:
the hole filled with real structure from elsewhere in the same clean image
(shifted by half the image). Shuffle has the right look and statistics but no
relation to what was in the hole, so its scores are the CHANCE LEVEL. A method
only reconstructs the hole if it beats shuffle; dense structures (many
interfaces) give a high chance level even for boundary F1.
Scores, all inside the hole: error (grey levels), boundary F1, F1 within 8 px
of the hole edge, F1 deeper than 8 px (walls carried THROUGH the hole).

Hole size is also given in CHORD LENGTHS: the mean width of the light phase
measured along rows and columns. It works for any two-phase structure (Voronoi
grains, spinodal domains, trabecular bone), so results from different structures
can be compared on one axis. For Voronoi, the grain diameter is printed too.

Writes outputs\hole_sweep\sweep.csv and, if matplotlib is installed, sweep.png.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from e2e_eval import band_f1, boundary_f1, load_generator, make_loader, repair  # noqa: E402


def grain_diameter(clean01):
    light = (clean01 > 0.5).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(light, 4)
    h, w = clean01.shape
    areas = []
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        if a >= 20 and x > 0 and y > 0 and x + bw < w and y + bh < h:  # skip grains cut by the frame
            areas.append(a)
    return float(2 * np.sqrt(np.median(areas) / np.pi)) if areas else float("nan")


def chord_length(clean01):
    """Mean length of light-phase runs along rows and columns (runs touching the frame excluded).
    Works for any two-phase structure: grains, spinodal domains, trabeculae."""
    light = clean01 > 0.5
    runs = []
    for arr in (light, light.T):
        for line in arr:
            d = np.diff(np.concatenate([[0], line.astype(np.int8), [0]]))
            starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
            for a0, b0 in zip(starts, ends):
                if a0 > 0 and b0 < len(line):
                    runs.append(b0 - a0)
    return float(np.mean(runs)) if runs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan_ckpt")
    ap.add_argument("--clean_dir", required=True, help="clean images not used in training")
    ap.add_argument("--sizes", type=int, nargs="+", default=[32, 48, 64, 96, 128, 160])
    ap.add_argument("--n", type=int, default=20, help="images per size")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--fill", type=int, default=128)
    ap.add_argument("--edge_px", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "hole_sweep"))
    ap.add_argument("--save_examples", action="store_true", help="save the first image's repairs per size")
    ap.add_argument("--dry_run", action="store_true", help="no generator (tests the plumbing)")
    a = ap.parse_args()

    clean_paths = sorted(Path(a.clean_dir).glob("*.png"))[: a.n]
    if not clean_paths:
        raise SystemExit(f"no .png in {a.clean_dir}")
    if a.dry_run:
        G = lambda masked, m: np.zeros_like(masked)
    else:
        if not a.gan_ckpt:
            raise SystemExit("--gan_ckpt is required (or --dry_run)")
        G = load_generator(a.gan_ckpt)
    load = make_loader(a.dry_run, a.size)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp_damaged.png"

    rows, diam, chord = [], [], []
    for k, cp in enumerate(clean_paths):
        clean8 = cv2.resize(cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE), (a.size, a.size),
                            interpolation=cv2.INTER_AREA)
        clean = load(cp)
        diam.append(grain_diameter(clean))
        chord.append(chord_length(clean))
        for s in a.sizes:
            if s >= a.size:
                continue
            m8 = np.zeros((a.size, a.size), np.uint8)
            o = (a.size - s) // 2
            m8[o:o + s, o:o + s] = 255
            true = m8 > 0
            d8 = clean8.copy()
            d8[true] = a.fill
            cv2.imwrite(str(tmp), d8)          # go through the same loader as real files
            dmg = load(tmp)
            m = true.astype(np.float32)
            dist = cv2.distanceTransform(m8 // 255, cv2.DIST_L2, 3)
            u8 = (np.clip(dmg, 0, 1) * 255).astype(np.uint8)
            shuf = np.roll(clean, (a.size // 2, a.size // 2), axis=(0, 1))
            results = {
                "gan": repair(G, dmg, m, 1),
                "shuffle": repair(lambda _x, _m: shuf, dmg, m, 1),
                "cv_ns": cv2.inpaint(u8, m8, 5, cv2.INPAINT_NS).astype(np.float32) / 255,
                "cv_telea": cv2.inpaint(u8, m8, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255,
            }
            for meth, rep in results.items():
                rows.append({
                    "image": cp.stem, "hole_px": s, "method": meth,
                    "mae": float(np.abs(rep - clean)[true].mean() * 255),
                    "bf1": float(boundary_f1(rep, clean, true)),
                    "bf1_edge": float(band_f1(rep, clean, true & (dist <= a.edge_px))),
                    "bf1_deep": float(band_f1(rep, clean, dist > a.edge_px)),
                    "grey_pct": float(((rep[true] > 0.3) & (rep[true] < 0.7)).mean() * 100),
                    "clean_grey_pct": float(((clean[true] > 0.3) & (clean[true] < 0.7)).mean() * 100),
                })
                if a.save_examples and k == 0:
                    cv2.imwrite(str(out / f"example_{s:03d}_{meth}.png"), (np.clip(rep, 0, 1) * 255).astype(np.uint8))
            if a.save_examples and k == 0:
                cv2.imwrite(str(out / f"example_{s:03d}_damaged.png"), (np.clip(dmg, 0, 1) * 255).astype(np.uint8))
        print(f"  {k + 1}/{len(clean_paths)} {cp.stem}")
    tmp.unlink(missing_ok=True)

    with open(out / "sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    gd = float(np.nanmedian(diam)) if np.isfinite(diam).any() else float("nan")
    cl = float(np.nanmean(chord))
    methods = ["gan", "cv_ns", "cv_telea", "shuffle"]
    sizes = [s for s in a.sizes if s < a.size]
    print(f"\nmean light-phase chord length: {cl:.1f} px   "
          f"median grain diameter: {gd:.1f} px (meaningful for Voronoi only)")
    print(f"\n{'hole px':>8}{'chords':>8}{'grains':>8} | " + " | ".join(f"{m + ' err/F1/deep':>22}" for m in methods))
    summary = {}
    for s in sizes:
        line = f"{s:>8}{s / cl:>8.1f}{s / gd:>8.1f} | "
        cells = []
        for meth in methods:
            sel = [r for r in rows if r["hole_px"] == s and r["method"] == meth]
            e = np.mean([r["mae"] for r in sel])
            f = np.nanmean([r["bf1"] for r in sel])
            dp = np.nanmean([r["bf1_deep"] for r in sel])
            summary[(s, meth)] = (e, f, dp)
            cells.append(f"{e:8.1f} {f:6.3f} {dp:6.3f}")
        print(line + " | ".join(f"{c:>22}" for c in cells))
    print("\nwashed-out check: % of hole pixels that are mid-grey (0.3-0.7). "
          "Real structure is mostly black or white; a blurred fill is mostly grey.")
    print(f"{'hole px':>8}{'clean':>8}" + "".join(f"{m:>10}" for m in methods))
    for s in sizes:
        sel = [r for r in rows if r["hole_px"] == s]
        line = f"{s:>8}{np.mean([r['clean_grey_pct'] for r in sel]):>8.1f}"
        for meth in methods:
            line += f"{np.mean([r['grey_pct'] for r in sel if r['method'] == meth]):>10.1f}"
        print(line)
    print(f"\nper image -> {out / 'sweep.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        x = [s / cl for s in sizes]
        for meth, st in zip(methods, ["-o", "--s", ":^", "-.x"]):
            ax[0].plot(x, [summary[(s, meth)][2] for s in sizes], st, label=meth)
            ax[1].plot(x, [summary[(s, meth)][0] for s in sizes], st, label=meth)
        ax[0].set_ylabel("boundary F1, deep inside hole")
        ax[1].set_ylabel("error in hole (grey levels)")
        for axi in ax:
            axi.set_xlabel(f"hole side in chord lengths (1 chord = {cl:.0f} px)")
            axi.grid(alpha=0.3)
            axi.legend()
        ax[0].set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(out / "sweep.png", dpi=150)
        print(f"plot      -> {out / 'sweep.png'}")
    except ImportError:
        print("(matplotlib not installed: no plot, the table and CSV have everything)")


if __name__ == "__main__":
    main()
