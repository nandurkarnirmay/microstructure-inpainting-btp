r"""Damage one image, let the U-Net find the damage, let the GAN repair it, and
save everything as one labelled strip.

  python demo_pipeline.py --image data\test_voronoi\voronoi_00007.png --mode scratch ^
      --unet_ckpt checkpoints\unet\unet_best_FIX.pt ^
      --gan_ckpt checkpoints\gan_voronoi\gan_weights.pt --out_dir outputs\demo_pipeline

Strip, left to right:
  clean | damaged | true mask | U-Net mask | GAN raw output |
  repaired (as infer.py) | repaired (whole hole) | error of whole-hole repair

The models run exactly as in infer.py: same preprocessing (CLAHE, normalise),
same U-Net thresholding and clean-up, same GAN call. Two repairs are shown:
  - "as infer.py": infer.py's composite, which first ERODES the hole by
    gopt.annulus_width (8 px) and keeps the damaged pixels in that rim.
  - "whole hole": the GAN output used over the entire predicted hole,
    feathered at the edge (as inpaint.py does).
If the first one shows a grey rim, or leaves scratches untouched, that is the
erosion, not the GAN.

--mode   center | rect | blob | scratch   (damage shape, random unless center)
--mask   use your own mask instead (white = damage)
--fill   grey level painted into the damage (128 = what the U-Net trained on)
--dry_run  no PyTorch: texture rule instead of U-Net, OpenCV instead of GAN
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ------------------------------------------------------------------ damage
def make_mask(mode, size, hole, rng):
    m = np.zeros((size, size), np.uint8)
    if mode == "center":
        a = (size - hole) // 2
        m[a:a + hole, a:a + hole] = 255
        return m
    from damage_batch import blob_mask, rect_mask, scratch_mask
    return {"rect": rect_mask, "blob": blob_mask, "scratch": scratch_mask}[mode]((size, size), rng)


# ------------------------------------------------------------------ scoring
def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum() + 1e-9)


def walls(img01):
    b = (img01 > 0.5).astype(np.uint8)
    return cv2.morphologyEx(b, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0


def boundary_f1(pred01, clean01, true, tol=2):
    region = cv2.dilate(true.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    bp, bg = walls(pred01) & region, walls(clean01) & region
    if bp.sum() == 0 or bg.sum() == 0:
        return float(bp.sum() == bg.sum())
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    prec = (bp & (cv2.dilate(bg.astype(np.uint8), k) > 0)).sum() / bp.sum()
    rec = (bg & (cv2.dilate(bp.astype(np.uint8), k) > 0)).sum() / bg.sum()
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def report(name, rep01, clean01, dmg01, true):
    err = np.abs(rep01 - clean01)[true].mean() * 255
    left = (np.abs(rep01 - dmg01)[true] * 255 < 2).mean() * 100
    print(f"  {name:<24} error in hole {err:6.1f}   unrepaired {left:5.1f}%   "
          f"boundary F1 {boundary_f1(rep01, clean01, true):.3f}")


# ------------------------------------------------------------------ strip
def tile(img01, label, size, colour=None):
    if colour is None:
        t = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
        t = cv2.resize(t, (size, size), interpolation=cv2.INTER_NEAREST)
        t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
    else:
        t = colour
    band = np.full((28, size, 3), 255, np.uint8)
    cv2.putText(band, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([band, t])


def overlay(base01, pred, true, size):
    """grey image; green = found, red = missed, blue = false alarm"""
    g = (np.clip(base01, 0, 1) * 255).astype(np.uint8)
    o = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).astype(np.float32)
    for sel, col in ((pred & true, (0, 200, 0)), (true & ~pred, (0, 0, 230)), (pred & ~true, (230, 120, 0))):
        o[sel] = 0.35 * o[sel] + 0.65 * np.array(col, np.float32)
    return cv2.resize(o.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)


def heat(err01, size):
    e = (np.clip(err01 / 0.5, 0, 1) * 255).astype(np.uint8)  # 0.5 (=128 grey levels) -> full scale
    return cv2.resize(cv2.applyColorMap(e, cv2.COLORMAP_INFERNO), (size, size), interpolation=cv2.INTER_NEAREST)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="CLEAN image to damage")
    ap.add_argument("--mode", choices=["center", "rect", "blob", "scratch"], default="center")
    ap.add_argument("--mask", help="your own damage mask (white = damage); overrides --mode")
    ap.add_argument("--hole", type=int, default=80, help="side of the centre square")
    ap.add_argument("--fill", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unet_ckpt")
    ap.add_argument("--gan_ckpt")
    ap.add_argument("--out_dir", default="outputs/demo_pipeline")
    ap.add_argument("--tile", type=int, default=256, help="size of each panel in the strip")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # ---- 1. damage the clean image (at the model's working size)
    clean8 = cv2.imread(a.image, cv2.IMREAD_GRAYSCALE)
    if clean8 is None:
        raise SystemExit(f"could not read {a.image}")
    if a.dry_run:
        size = 256
    else:
        from utils.config import load_config
        cfg = load_config(str(ROOT / "configs" / "default.yaml"))
        size = int(cfg.get("preprocessing", {}).get("image_size") or 256)
    clean8 = cv2.resize(clean8, (size, size), interpolation=cv2.INTER_AREA)
    if a.mask:
        tm = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
        if tm is None:
            raise SystemExit(f"could not read {a.mask}")
        tm = cv2.resize(tm, (size, size), interpolation=cv2.INTER_NEAREST)
    else:
        tm = make_mask(a.mode, size, a.hole, rng)
    true = tm > 127
    dmg8 = clean8.copy()
    dmg8[true] = a.fill
    cv2.imwrite(str(out / "clean.png"), clean8)
    cv2.imwrite(str(out / "damaged.png"), dmg8)
    cv2.imwrite(str(out / "true_mask.png"), tm)

    # ---- 2+3. detect and repair
    if a.dry_run:
        from var_detect import detect
        x = dmg8.astype(np.float32) / 255
        c = clean8.astype(np.float32) / 255
        pm = detect(dmg8) > 127
        raw = cv2.inpaint(dmg8, pm.astype(np.uint8) * 255, 5, cv2.INPAINT_NS).astype(np.float32) / 255
        ero = cv2.erode(pm.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=8).astype(np.float32)
        alpha = np.clip(cv2.GaussianBlur(ero, (9, 9), 2.2), 0, 1)
        rep_infer = raw * alpha + x * (1 - alpha)
    else:
        if not (a.unet_ckpt and a.gan_ckpt):
            raise SystemExit("give --unet_ckpt and --gan_ckpt (or --dry_run)")
        import torch
        from inference.pipeline import InpaintPipeline

        dev = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        pipe = InpaintPipeline(cfg, dev)
        pre = pipe.pre

        def prep(path):
            return pre.to_tensor(pre.process_numpy(pre.load_image(str(path)))).unsqueeze(0).to(dev)

        t = prep(out / "damaged.png")
        tc = prep(out / "clean.png")
        u = cfg.get("unet", {})
        unet = pipe.load_unet(a.unet_ckpt, in_channels=int(u.get("in_channels", 1)),
                              base_channels=int(u.get("base_channels", 32)),
                              num_levels=int(u.get("num_levels", 4)))
        m = pipe.refine_mask(pipe.segment_defects(unet, t))
        g = cfg.get("gan", {})
        G = pipe.load_gan(a.gan_ckpt, in_channels=int(g.get("in_channels", 2)),
                          out_channels=int(g.get("out_channels", 1)), base=int(g.get("base_channels", 48)))
        with torch.no_grad():
            raw_t = G(t * (1.0 - m), m)
        rep_t = pipe.inpaint_masked_gan(G, t, m)          # exactly infer.py's output
        x = t[0, 0].cpu().numpy()
        c = tc[0, 0].cpu().numpy()
        pm = m[0, 0].cpu().numpy() > 0.5
        raw = raw_t[0, 0].cpu().numpy()
        rep_infer = rep_t[0, 0].cpu().numpy()

    alpha = np.clip(cv2.GaussianBlur(pm.astype(np.float32), (5, 5), 1.0), 0, 1)
    rep_whole = raw * alpha + x * (1 - alpha)

    for name, im in (("unet_mask", pm.astype(np.float32)), ("gan_raw", raw),
                     ("repaired_infer", rep_infer), ("repaired_whole", rep_whole)):
        cv2.imwrite(str(out / f"{name}.png"), (np.clip(im, 0, 1) * 255).astype(np.uint8))

    # ---- 4. strip
    T = a.tile
    panels = [
        tile(c, "clean", T),
        tile(x, "damaged", T),
        tile(true.astype(np.float32), "true mask", T),
        tile(None, "U-Net mask: grn ok, red miss, blu false", T, overlay(x, pm, true, T)),
        tile(raw, "GAN raw output", T),
        tile(rep_infer, "repaired, as infer.py", T),
        tile(rep_whole, "repaired, whole hole", T),
        tile(None, "error, whole hole", T, heat(np.abs(rep_whole - c) * true, T)),
    ]
    gap = np.full((T + 28, 6, 3), 255, np.uint8)
    strip = np.hstack([p for pair in zip(panels, [gap] * len(panels)) for p in pair][:-1])
    cv2.imwrite(str(out / "strip.png"), strip)

    # ---- 5. numbers
    print(f"damage: {a.mask or a.mode}, {100 * true.mean():.1f}% of the image, fill {a.fill}")
    print(f"U-Net mask: Dice {dice(pm, true):.3f}   found {100 * (pm & true).sum() / max(true.sum(), 1):.1f}% "
          f"of the damage   false alarm {100 * (pm & ~true).sum() / pm.size:.2f}% of the image")
    print("repair, scored inside the true damage (0-255 grey levels, preprocessed image):")
    report("as infer.py (eroded)", rep_infer, c, x, true)
    report("whole hole", rep_whole, c, x, true)
    print(f"\nstrip -> {out / 'strip.png'}   (single panels saved next to it)")


if __name__ == "__main__":
    main()
