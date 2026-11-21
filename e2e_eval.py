r"""End-to-end test: do better damage masks give better final repairs?

For every test image, repair it with the SAME generator using each mask source
(the true mask, the fixed U-Net, the buggy U-Net, the texture rule ...) and
compare the result with the clean original.

  python e2e_eval.py --gan_ckpt checkpoints\gan_voronoi\gan_weights.pt ^
      --damaged data\unet_vtest_images --clean_dir data\test_voronoi_new ^
      --true_masks data\unet_vtest_masks ^
      --masks true=data\unet_vtest_masks unet_fix=outputs\vtest_fix ^
              unet_bug=outputs\vtest_bug texture=outputs\var_vtest ^
      --out outputs\e2e

Images go through the same preprocessing as training and infer.py (CLAHE,
normalise) unless --raw is given. Repair matches the fixed infer.py: hole
zeroed, G(masked, mask), hole grown by 2 px, feathered (9 px, sigma 2.2).
Older note -- the first version used inpaint.py's raw-image path:
composite of real pixels outside and the generator inside.

Scored INSIDE THE TRUE DAMAGE only, because outside it every method keeps the
same real pixels and a whole-image score would dilute the difference:
  mae_hole   mean |repaired - clean| over true damage pixels (0-255 scale)
  left_pct   % of true damage pixels still showing the paint (|repaired - damaged| < 2)
  phase_err  |light-phase fraction of repair - of clean| inside the hole, in % points.
             MAE rewards blur (a grey average scores well); phase_err does not.
  bound_f1   do the phase boundaries (grain walls) in the hole land where the clean
             image has them? F1 of boundary pixels, 2-px tolerance, 0-1, higher better.
             This is the structure test: blur and smeared walls score low.
  bf1_edge   boundary F1 within --edge_px of the hole edge (walls continued a little way)
  bf1_deep   boundary F1 deeper than --edge_px inside the hole (walls carried THROUGH it).
             Walls that stop at the hole edge give a decent bf1_edge but a low bf1_deep.
             Thin damage (scratches) has no deep region and is left out of bf1_deep.

--classical adds rows repaired WITHOUT the GAN (OpenCV Telea / Navier-Stokes
inpainting, using the true mask) -- the baseline the GAN has to beat.
Damage a mask misses is never repaired, so it stays painted and costs MAE.

Clean originals are matched by stem: voronoi_00007_00007 -> voronoi_00007.png
(damage_batch.py appends _<index> to the clean file's stem).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import cv2


def find_mask(root: Path, stem: str):
    for cand in (root / stem / "mask.png", root / f"{stem}.png", root / f"{stem}_mask.png"):
        if cand.is_file():
            return cand
    return None


def clean_for(clean_dir: Path, stem: str):
    base = stem.rsplit("_", 1)[0] if "_" in stem else stem
    for cand in (clean_dir / f"{base}.png", clean_dir / f"{stem}.png"):
        if cand.is_file():
            return cand
    return None


def read_gray(p: Path, size: int, interp) -> np.ndarray:
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise SystemExit(f"could not read {p}")
    if im.shape != (size, size):
        im = cv2.resize(im, (size, size), interpolation=interp)
    return im


def load_generator(ckpt: str):
    import torch
    from models.gan import MaskGuidedGenerator

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(ckpt, map_location=dev, weights_only=False)
    if isinstance(sd, dict) and "G" in sd:          # gan_weights.pt (end of training)
        sd = sd["G"]
    elif isinstance(sd, dict) and "model" in sd:    # gan_last.pt (saved every epoch)
        print(f"  {ckpt}: training checkpoint from epoch {sd.get('epoch', '?')}")
        sd = sd["model"]
    G = MaskGuidedGenerator().to(dev)
    G.load_state_dict(sd)
    G.eval()

    def run(masked: np.ndarray, m: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t1 = torch.from_numpy(masked)[None, None].to(dev)
            t2 = torch.from_numpy(m)[None, None].to(dev)
            return G(t1, t2)[0, 0].cpu().numpy()

    return run


def repair(G, damaged01: np.ndarray, m: np.ndarray, feather: int) -> np.ndarray:
    masked = damaged01 * (1.0 - m)
    pred = G(masked.astype(np.float32), m.astype(np.float32))
    grown = cv2.dilate(m.astype(np.float32), np.ones((5, 5), np.uint8))
    alpha = np.clip(cv2.GaussianBlur(grown, (9, 9), 2.2), 0, 1) if feather > 0 else grown
    return damaged01 * (1 - alpha) + pred * alpha


def make_loader(raw: bool, size: int):
    """Return f(path) -> float32 HxW in [0,1], preprocessed exactly like infer.py unless raw."""
    if raw:
        return lambda p: read_gray(Path(p), size, cv2.INTER_AREA).astype(np.float32) / 255.0
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from preprocessing.pipeline import MicroscopyPreprocessor, PreprocessConfig
    from utils.config import load_config

    cfg = load_config(str(Path(__file__).resolve().parent / "configs" / "default.yaml"))
    pc = cfg.get("preprocessing", {})
    pre = MicroscopyPreprocessor(PreprocessConfig(
        image_size=pc.get("image_size"), grayscale=bool(pc.get("grayscale", True)),
        normalize=pc.get("normalize", "minmax"), clahe_clip=float(pc.get("clahe_clip", 2.0)),
        clahe_grid=int(pc.get("clahe_grid", 8)), denoise_h=float(pc.get("denoise_h", 0.0))))

    def load(p):
        x = pre.process_numpy(pre.load_image(str(p)))
        x = x[..., 0] if x.ndim == 3 else x
        if x.shape != (size, size):
            x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
        return np.clip(x, 0, 1).astype(np.float32)

    return load


def boundaries(img01: np.ndarray) -> np.ndarray:
    b = (img01 > 0.5).astype(np.uint8)
    return cv2.morphologyEx(b, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0


def boundary_f1(fixed, clean, true, tol=2):
    region = cv2.dilate(true.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    bp, bg = boundaries(fixed) & region, boundaries(clean) & region
    if bg.sum() == 0 and bp.sum() == 0:
        return 1.0
    if bg.sum() == 0 or bp.sum() == 0:
        return 0.0
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    near_g = cv2.dilate(bg.astype(np.uint8), k) > 0
    near_p = cv2.dilate(bp.astype(np.uint8), k) > 0
    prec = (bp & near_g).sum() / bp.sum()
    rec = (bg & near_p).sum() / bg.sum()
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


EDGE_PX = 8


def band_f1(fixed, clean, band, tol=2):
    if band.sum() < 30:
        return float("nan")
    bp, bg = boundaries(fixed) & band, boundaries(clean) & band
    if bg.sum() == 0 and bp.sum() == 0:
        return float("nan")
    if bg.sum() == 0 or bp.sum() == 0:
        return 0.0
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    prec = (bp & (cv2.dilate(bg.astype(np.uint8), k) > 0)).sum() / bp.sum()
    rec = (bg & (cv2.dilate(bp.astype(np.uint8), k) > 0)).sum() / bg.sum()
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def score(row, label, fixed, clean, dmg, true, save_dir, stem):
    err = np.abs(fixed - clean)[true] * 255
    left = (np.abs(fixed - dmg)[true] * 255 < 2).mean() * 100
    phase = abs((fixed[true] > 0.5).mean() - (clean[true] > 0.5).mean()) * 100
    row[f"{label}_mae"] = round(float(err.mean()), 2)
    row[f"{label}_left"] = round(float(left), 1)
    row[f"{label}_phase"] = round(float(phase), 1)
    row[f"{label}_bf1"] = round(float(boundary_f1(fixed, clean, true)), 3)
    dist = cv2.distanceTransform(true.astype(np.uint8), cv2.DIST_L2, 3)
    row[f"{label}_bf1_edge"] = round(float(band_f1(fixed, clean, true & (dist <= EDGE_PX))), 3)
    row[f"{label}_bf1_deep"] = round(float(band_f1(fixed, clean, dist > EDGE_PX)), 3)
    if save_dir is not None:
        d = save_dir / label
        d.mkdir(exist_ok=True)
        cv2.imwrite(str(d / f"{stem}.png"), (np.clip(fixed, 0, 1) * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan_ckpt", help="generator weights (not needed with --dry_run)")
    ap.add_argument("--damaged", required=True, help="folder of damaged test images")
    ap.add_argument("--clean_dir", required=True, help="folder of the clean originals")
    ap.add_argument("--true_masks", required=True, help="folder of true damage masks")
    ap.add_argument("--masks", nargs="+", required=True, help="label=folder pairs, one per mask source")
    ap.add_argument("--out", default="outputs/e2e")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--feather", type=int, default=2)
    ap.add_argument("--save_images", action="store_true", help="write each repaired image")
    ap.add_argument("--classical", action="store_true",
                    help="also score OpenCV Telea and Navier-Stokes inpainting with the true mask")
    ap.add_argument("--edge_px", type=int, default=8, help="width of the hole-edge band for bf1_edge / bf1_deep")
    ap.add_argument("--raw", action="store_true",
                    help="skip preprocessing (the old behaviour; the GAN was trained on preprocessed images)")
    ap.add_argument("--dry_run", action="store_true", help="no generator: fill holes with black (tests the plumbing only)")
    a = ap.parse_args()

    global EDGE_PX
    EDGE_PX = a.edge_px
    sources = []
    for item in a.masks:
        if "=" not in item:
            raise SystemExit(f"--masks entries must be label=folder, got {item!r}")
        label, folder = item.split("=", 1)
        sources.append((label, Path(folder)))

    if a.dry_run:
        G = lambda masked, m: np.zeros_like(masked)
    else:
        if not a.gan_ckpt:
            raise SystemExit("--gan_ckpt is required (or use --dry_run)")
        G = load_generator(a.gan_ckpt)

    load = make_loader(a.raw or a.dry_run, a.size)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in Path(a.damaged).glob("*.png"))
    if not stems:
        raise SystemExit(f"no .png in {a.damaged}")

    rows, skipped = [], []
    for stem in stems:
        cpath = clean_for(Path(a.clean_dir), stem)
        tpath = find_mask(Path(a.true_masks), stem)
        if cpath is None or tpath is None:
            skipped.append((stem, "no clean original" if cpath is None else "no true mask"))
            continue
        dmg = load(Path(a.damaged) / f"{stem}.png")
        clean = load(cpath)
        true = read_gray(tpath, a.size, cv2.INTER_NEAREST) > 127
        if true.sum() == 0:
            skipped.append((stem, "empty true mask"))
            continue
        # sanity: outside the damage, damaged and clean should be the same picture
        outside_diff = float(np.abs(dmg - clean)[~true].mean() * 255)
        row = {"stem": stem, "outside_diff": round(outside_diff, 2)}
        for label, folder in sources:
            mp = find_mask(folder, stem)
            if mp is None:
                row[f"{label}_mae"] = row[f"{label}_left"] = row[f"{label}_phase"] = row[f"{label}_bf1"] = float("nan")
                row[f"{label}_bf1_edge"] = row[f"{label}_bf1_deep"] = float("nan")
                continue
            m = (read_gray(mp, a.size, cv2.INTER_NEAREST) > 127).astype(np.float32)
            fixed = repair(G, dmg, m, a.feather)
            score(row, label, fixed, clean, dmg, true, out if a.save_images else None, stem)
        if a.classical:
            d8 = (dmg * 255).astype(np.uint8)
            t8 = true.astype(np.uint8) * 255
            for label, flag in (("cv_telea", cv2.INPAINT_TELEA), ("cv_ns", cv2.INPAINT_NS)):
                fixed = cv2.inpaint(d8, t8, 5, flag).astype(np.float32) / 255.0
                score(row, label, fixed, clean, dmg, true, out if a.save_images else None, stem)
        rows.append(row)

    if not rows:
        raise SystemExit("nothing scored: " + "; ".join(f"{s}: {w}" for s, w in skipped[:5]))

    with open(out / "e2e_per_image.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    labels = [l for l, _ in sources] + (["cv_telea", "cv_ns"] if a.classical else [])
    print(f"\n{len(rows)} images scored, {len(skipped)} skipped")
    od = np.array([r["outside_diff"] for r in rows])
    print(f"sanity: mean |damaged - clean| outside the damage = {od.mean():.2f} grey levels "
          f"(raw: ~0. preprocessed: a few levels is normal, CLAHE reacts to the patch; above ~15 means the wrong clean originals)")
    print(f"\n{'source':<14}{'n':>4}{'MAE in hole':>14}{'unrepaired %':>15}{'phase err':>12}{'bound F1':>10}{'F1 edge':>9}{'F1 deep':>9}")
    for l in labels:
        v = np.array([r[f"{l}_mae"] for r in rows], float)
        u = np.array([r[f"{l}_left"] for r in rows], float)
        ph = np.array([r[f"{l}_phase"] for r in rows], float)
        bf = np.array([r[f"{l}_bf1"] for r in rows], float)
        be = np.array([r[f"{l}_bf1_edge"] for r in rows], float)
        bd = np.array([r[f"{l}_bf1_deep"] for r in rows], float)
        ok = ~np.isnan(v)
        print(f"{l:<14}{ok.sum():>4}{np.nanmean(v):>14.2f}{np.nanmean(u):>15.1f}{np.nanmean(ph):>12.1f}{np.nanmean(bf):>10.3f}{np.nanmean(be):>9.3f}{np.nanmean(bd):>9.3f}")

    ref = labels[0]
    print(f"\npaired vs '{ref}' (positive = worse than {ref}):")
    for l in labels[1:]:
        a_ = np.array([r[f"{l}_mae"] for r in rows], float)
        b_ = np.array([r[f"{ref}_mae"] for r in rows], float)
        ok = ~np.isnan(a_) & ~np.isnan(b_)
        d = a_[ok] - b_[ok]
        if len(d) > 1 and d.std(ddof=1) > 0:
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            print(f"  {l:<12} MAE diff {d.mean():+.2f}   worse on {(d > 0.5).sum()}/{len(d)}   paired t {t:+.2f}")
        else:
            print(f"  {l:<12} MAE diff {d.mean() if len(d) else float('nan'):+.2f}")
    print(f"\nboundary F1, paired vs '{ref}' (negative = worse structure than {ref}):")
    for l in labels[1:]:
        a_ = np.array([r[f"{l}_bf1"] for r in rows], float)
        b_ = np.array([r[f"{ref}_bf1"] for r in rows], float)
        ok = ~np.isnan(a_) & ~np.isnan(b_)
        d = a_[ok] - b_[ok]
        if len(d) > 1 and d.std(ddof=1) > 0:
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            print(f"  {l:<12} F1 diff {d.mean():+.3f}   better on {(d > 0.01).sum()}/{len(d)}   paired t {t:+.2f}")
    print(f"\nper-image table -> {out / 'e2e_per_image.csv'}")
    for s, why in skipped[:10]:
        print(f"  skipped {s}: {why}")


if __name__ == "__main__":
    main()
