"""Sweep hole sizes over many test images; report mean metrics per size.

  python batch_eval.py --ckpt checkpoints/gan/gan_weights.pt ^
                       --clean_dir data/test_clean ^
                       --holes 32,48,64,80,96,112 ^
                       --out outputs/sweep
"""
import argparse, csv, json
from pathlib import Path

import numpy as np
import cv2
import torch
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

from models.gan import MaskGuidedGenerator


def radial_s2(binary, max_r=40):
    max_r = max(4, min(max_r, min(binary.shape) // 3))
    """Radially averaged two-point correlation via the autocorrelation theorem."""
    f = np.fft.fft2(binary.astype(np.float32))
    acf = np.real(np.fft.ifft2(f * np.conj(f))) / binary.size
    acf = np.fft.fftshift(acf)
    c = np.array(acf.shape) // 2
    yy, xx = np.mgrid[0:acf.shape[0], 0:acf.shape[1]]
    r = np.sqrt((yy - c[0])**2 + (xx - c[1])**2).astype(int)
    out = np.zeros(max_r)
    for k in range(max_r):
        sel = r == k
        if sel.any():
            out[k] = acf[sel].mean()
    return out


def boundary_f1(pb, gb, tol=2):
    """F1 between edge maps, allowing a tolerance of `tol` pixels."""
    ep = cv2.Canny((pb * 255).astype(np.uint8), 50, 150) > 0
    eg = cv2.Canny((gb * 255).astype(np.uint8), 50, 150) > 0
    if ep.sum() == 0 and eg.sum() == 0:
        return 1.0
    if ep.sum() == 0 or eg.sum() == 0:
        return 0.0
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    ep_d = cv2.dilate(ep.astype(np.uint8), k) > 0
    eg_d = cv2.dilate(eg.astype(np.uint8), k) > 0
    prec = (ep & eg_d).sum() / ep.sum()
    rec = (eg & ep_d).sum() / eg.sum()
    return float(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))


def metrics(pred, gt, box=None):
    if box is not None:
        y0, x0, side = box
        pred = pred[y0:y0 + side, x0:x0 + side]
        gt = gt[y0:y0 + side, x0:x0 + side]
    p = pred.astype(np.float32) / 255.0
    g = gt.astype(np.float32) / 255.0
    pb = (p > 0.5).astype(np.float32)
    gb = (g > 0.5).astype(np.float32)
    inter = (pb * gb).sum()
    dice = 2 * inter / (pb.sum() + gb.sum() + 1e-8)
    s2p, s2g = radial_s2(pb), radial_s2(gb)
    return {
        "psnr": float(psnr_fn(g, p, data_range=1.0)),
        "ssim": float(ssim_fn(g, p, data_range=1.0,
                              win_size=min(7, (min(g.shape) // 2) * 2 - 1))),
        "mae": float(np.abs(p - g).mean()),
        "dice": float(dice),
        "s2_mse": float(((s2p - s2g) ** 2).mean()),
        "boundary_f1": boundary_f1(pb, gb),
        "phase_error_percent": float(100 * abs(pb.mean() - gb.mean())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clean_dir", required=True, help="folder of CLEAN test images")
    ap.add_argument("--holes", default="32,48,64,80,96,112")
    ap.add_argument("--out", default="outputs/sweep")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(args.ckpt, map_location=dev)
    if isinstance(sd, dict) and "G" in sd:
        sd = sd["G"]
    G = MaskGuidedGenerator().to(dev); G.load_state_dict(sd); G.eval()

    files = sorted(p for p in Path(args.clean_dir).glob("*.png"))[: args.limit]
    if not files:
        raise SystemExit(f"no .png files in {args.clean_dir}")
    holes = [int(h) for h in args.holes.split(",")]
    print(f"{len(files)} images x {len(holes)} hole sizes = {len(files)*len(holes)} runs\n")

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    per_image = []

    for hole in holes:
        per = []
        for f in files:
            gt = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (args.size, args.size), interpolation=cv2.INTER_AREA)

            m = np.zeros(gt.shape, np.float32)
            y0 = x0 = (args.size - hole) // 2
            m[y0:y0 + hole, x0:x0 + hole] = 1.0

            x = gt.astype(np.float32) / 255.0
            masked = x * (1 - m)
            with torch.no_grad():
                pr = G(torch.from_numpy(masked)[None, None].to(dev),
                       torch.from_numpy(m)[None, None].to(dev))[0, 0].cpu().numpy()
            final = x * (1 - m) + pr * m
            fin8 = (final * 255).clip(0, 255).astype(np.uint8)
            mm = metrics(fin8, gt)
            hh = metrics(fin8, gt, box=(y0, x0, hole))
            mm.update({'hole_' + k: v for k, v in hh.items()})
            per.append(mm)
            per_image.append({'hole': hole, 'image': f.name, **mm})

        agg = {"hole": hole, "frac_missing_pct": round(100 * hole * hole / args.size**2, 2),
               "n": len(per)}
        for k in per[0]:
            v = [d[k] for d in per]
            agg[f"{k}_mean"] = round(float(np.mean(v)), 6)
            agg[f"{k}_std"] = round(float(np.std(v)), 6)
        rows.append(agg)
        print(f"hole {hole:3d} ({agg['frac_missing_pct']:5.2f}% missing)  "
              f"SSIM {agg['ssim_mean']:.4f}+-{agg['ssim_std']:.4f}  "
              f"S2 {agg['s2_mse_mean']:.3e}  phase {agg['phase_error_percent_mean']:.2f}%")

    with open(outdir / "sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (outdir / "sweep.json").write_text(json.dumps(rows, indent=2))

    with open(outdir / "per_image.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_image[0].keys())); w.writeheader(); w.writerows(per_image)
    print(f"\nwrote {outdir/'sweep.csv'}, {outdir/'sweep.json'} and {outdir/'per_image.csv'}")


if __name__ == "__main__":
    main()
