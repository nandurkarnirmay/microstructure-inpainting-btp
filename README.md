# Deep Learning–Based Automatic Inpainting for Material Microscopic Images

This repository is a **research-style PyTorch project** for restoring material micrographs: it finds damaged regions, builds masks, inpaints with a GAN, optionally trains a **spatial generator with G-opt**, optionally refines results with **z-opt**, and reports image and microstructure metrics.

The code is **original and modular** (not a copy-paste of a single upstream repo). It is suitable for a BTP or thesis workflow: train models, save checkpoints, run inference, and evaluate numerically.

---

## 1. What problem this solves (why this project exists)

Microscopy images often contain **scratches, stains, dust, or sensor artefacts**. Those defects hurt **visual quality** and downstream tasks (**segmentation, phase analysis, statistics**). This pipeline:

1. **Detects** where the sample is damaged (U-Net segmentation).
2. **Builds** a clean binary mask (thresholding + morphology).
3. **Inpaints** missing regions with a **mask-guided GAN** (general restoration).
4. Optionally uses **G-opt**: a generator trained with a **fixed latent** for **boundary-consistent** microstructure synthesis (paper-style regime).
5. Optionally uses **z-opt**: **freeze** that generator and **optimise only the latent** for a **specific image**, improving continuity and statistics without retraining weights.
6. Optionally **re-segments** after repair (**cyclic validation**) to catch residual defects.

---

## 2. What you need before you start

- **Python 3.10+** recommended (3.9+ usually works).
- **PyTorch** with CUDA if you have an NVIDIA GPU (much faster for GAN / G-opt). CPU-only is possible but slow.
- **Disk space** for datasets, checkpoints (hundreds of MB typical), and `outputs/`.

All command-line examples below assume your **current working directory is the project root** (the folder that contains `train_unet.py`, `configs/`, `models/`, etc.). On your machine that might be:

`c:\Users\HP\OneDrive\Desktop\BTP_Code`

If you `cd` somewhere else, imports may fail because the scripts add the project root to `sys.path` relative to the script file.

---

## 3. Installation (from zero)

```bash
cd BTP_Code
python -m venv .venv
```

**Windows (PowerShell):**

```text
.\.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

This installs PyTorch, torchvision, OpenCV, scikit-image, PyYAML, tqdm, matplotlib, Gradio (for the optional GUI), and related tools.

---

## 4. End-to-end pipeline (what happens, in order)

Think of the system as a **factory line**. Each stage has a purpose; you do not have to run every optional stage.

```text
Raw micrograph
    │
    ▼
┌───────────────────┐
│  Preprocessing    │  CLAHE, resize, normalise, optional denoise (`preprocessing/`)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  U-Net            │  Predict defect mask (`models/unet.py`, `train_unet.py`)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Mask refinement  │  Morphology + small blobs removed (`preprocessing/mask_generation.py`)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Mask-guided GAN  │  Fill hole conditioned on mask (`train_gan.py`, `models/gan.py`)
└─────────┬─────────┘
          │
          ├──────────────────────────────┐
          ▼                              ▼
┌───────────────────┐          ┌───────────────────┐
│  G-opt (optional) │          │  z-opt (optional) │
│  Train spatial G  │          │  Optimise latent  │
│  + fixed z_fixed  │          │  with G frozen     │
│  `train_gopt.py`  │          │  `optimize_zopt.py`│
└─────────┬─────────┘          │  or `infer --run_  │
          │                    │       zopt`        │
          │                    └─────────┬──────────┘
          │                              │
          └──────────────┬───────────────┘
                         ▼
              ┌───────────────────┐
              │  Blending / paste  │  Feathered boundary paste (`losses/inpainting.py`)
              └─────────┬─────────┘
                         │
                         ▼
              ┌───────────────────┐
              │  Cyclic (optional)  │  Re-segment, widen mask, repeat (`training/cyclic.py`)
              └─────────┬─────────┘
                         ▼
              Restored micrograph + metrics (`evaluate.py`)
```

**Why separate GAN vs G-opt vs z-opt?**

- **Mask-guided GAN** learns “inpaint this masked region” from **random synthetic holes**. It is flexible and easy to train but does not encode the paper’s **fixed-seed boundary contract**.
- **G-opt** trains a **spatial-noise generator** with **WGAN-GP** on random noise, plus a **content loss** on an **annulus** using one **saved** `z_fixed`. That aligns the generator with **stable boundaries** typical of microstructure papers.
- **z-opt** assumes **G is already good**; it only adjusts **one latent** for **one image** so annulus / interior match the ground truth while staying close to a **Gaussian** prior (KL-style terms in `optimizers/z_opt.py`).

---

## 5. Folder and file map (where things live, and why)

| Path | Role |
|------|------|
| `configs/default.yaml` | **Single place** for learning rates, hole size, loss weights, blending, z-opt steps, etc. All top-level scripts read this unless you pass `--config other.yaml`. |
| `data/` | **Your** images and masks (you create subfolders; nothing is committed by default). |
| `datasets/` | PyTorch **Dataset** classes: folder loading + synthetic holes (`datasets/microscopy.py`). |
| `preprocessing/` | Read images, CLAHE, normalise, patches; **mask cleanup** (`preprocessing/pipeline.py`, `mask_generation.py`). |
| `models/` | **UNet**, **MaskGuidedGenerator**, **SpatialNoiseGenerator**, **PatchDiscriminator** (`models/unet.py`, `models/gan.py`). |
| `losses/` | Dice/BCE, WGAN-GP, annulus MSE, blending (`losses/`). |
| `training/` | Training loops for U-Net, GAN, G-opt; **cyclic** helper (`training/*.py`). |
| `optimizers/` | **z-opt** latent optimisation (`optimizers/z_opt.py`). |
| `inference/` | **`InpaintPipeline`** class: load weights, segment, inpaint, z-opt glue (`inference/pipeline.py`). |
| `evaluation/` | PSNR, SSIM, Dice, IoU, boundary F1, **S₂**, **phase error**, optional **FID** (`evaluation/metrics.py`, `microstructure_stats.py`, `fid.py`). |
| `visualization/` | Save comparison grids and loss curves (`visualization/viz.py`). |
| `utils/` | Load YAML, set seeds, save checkpoints (`utils/`). |
| `checkpoints/` | **Weights** written by training (`unet/`, `gan/`, `gopt/` subdirs created at runtime). |
| `outputs/` | **Logs**, inferred PNGs, z-opt `.pt` bundles, metric JSON (default under `outputs/`). |

**Top-level scripts (these are what you actually run):**

| Script | Purpose |
|--------|---------|
| `train_unet.py` | Train defect segmentation. |
| `train_gan.py` | Train mask-guided inpainting (self-supervised holes). |
| `train_gopt.py` | Train spatial generator with G-opt losses. |
| `optimize_zopt.py` | Run z-opt on **one** image using a G-opt bundle. |
| `infer.py` | Full demo path: optional U-Net mask, GAN, G-opt bundle, optional `--run_zopt`. |
| `evaluate.py` | Compare **pred** vs **gt** images; optional FID folders. |
| `app.py` | Small **Gradio** UI: load checkpoints, run mask + GAN in the browser. |
| `visualize.py` | CLI helper for plotting curves from JSON (see script `--help`). |

The **library modules** under `models/`, `training/`, etc. are **imported** by those scripts; you normally do **not** run them with `python models/unet.py`.

---

## 6. Prepare your data (where to put files)

### 6.1 U-Net (supervised damage detection)

Create folders such as:

```text
data/train_images/   … micrographs (.png, .jpg, .tif, …)
data/train_masks/    … same file name stem, single-channel mask: 1 = defect / inpaint region, 0 = OK
```

Example pair:

- `data/train_images/slide_001.png`
- `data/train_masks/slide_001.png`

If a mask file is **missing**, the loader fills zeros (the network sees “no defect” everywhere)—**avoid that for real training**.

### 6.2 Mask-guided GAN (self-supervised)

Only **`data/train_images/`** (or any folder you pass to `--images`) is required. The code cuts **random rectangular holes** per epoch (`SyntheticInpaintDataset`).

### 6.3 G-opt

Same idea as GAN data: **folders of full-field micrographs**. The trainer applies a **fixed-size central square hole** (see `gopt.hole_size` in `configs/default.yaml`). Use images whose size matches `preprocessing.image_size` (default **256×256** after preprocessing).

### 6.4 Optional FID folders

For Fréchet Inception Distance you need **many** small patches in two directories, for example `data/fid/real/` and `data/fid/gen/`. FID is **not** defined for a single image pair; see `evaluate.py` FID flags below.

---

## 7. Configuration (`configs/default.yaml`)

**Why:** Keeps experiments reproducible and avoids editing Python to change hyperparameters.

Important blocks:

- **`preprocessing`**: `image_size`, grayscale, CLAHE, normalisation.
- **`unet`**: channels, depth, batch size, epochs, learning rate.
- **`gan`**: L1 weights, WGAN-GP weight, learning rates, `grad_clip`, synthetic hole range.
- **`gopt`**: `hole_size`, `annulus_width`, `content_weight`, `content_every`, `interior_content_weight`, learning rates.
- **`zopt`**: optimisation steps, `lr`, `kl_weight`, `interior_weight`, `grad_clip`, etc.
- **`blending`**: `mode: feather` vs `hard`, kernel and sigma for soft boundaries.
- **`evaluation`**: `s2_max_r`, `phase_threshold` for microstructure metrics.

Copy `configs/default.yaml` to `configs/my_experiment.yaml` and pass `--config configs/my_experiment.yaml` to any script that supports it.

---

## 8. How to run each script (from scratch)

Always **`cd` into the project root** first. Use your own paths inside `data/` and `outputs/`.

### 8.1 `train_unet.py` — train the defect segmenter

**What it does:** Loads image + mask pairs, trains a U-Net with Dice + BCE, saves `checkpoints/unet/unet_best.pt` (best validation Dice) and `unet_last.pt`.

**When:** First stage if you need **automatic masks** from real defects.

**Example:**

```bash
python train_unet.py --images data/train_images --masks data/train_masks --config configs/default.yaml
```

**Main arguments:**

- `--images` (required): folder of micrographs.
- `--masks` (optional but strongly recommended): folder of binary masks, same stem as images.
- `--config`: YAML path (default `configs/default.yaml`).
- `--val_ratio`: fraction held out for validation (default `0.1`).

**Outputs:**

- `checkpoints/unet/unet_best.pt`, `unet_last.pt`
- TensorBoard logs under `outputs/logs/unet/` (if enabled in trainer).

---

### 8.2 `train_gan.py` — train mask-guided inpainting

**What it does:** Random holes on each image; generator sees **masked image + hole mask**; discriminator sees **real/fake + mask**; WGAN-GP + L1 in hole + small L1 outside hole (see `gan` section in YAML).

**When:** You want a **general inpainter** that works for **varying hole shapes** at inference (when you have a U-Net mask or a hand-drawn mask).

**Example:**

```bash
python train_gan.py --images data/train_images --config configs/default.yaml
```

**Outputs:**

- `checkpoints/gan/gan_last.pt`, `checkpoints/gan/gan_weights.pt` (dict with `G` and `D`).

---

### 8.3 `train_gopt.py` — train spatial generator (G-opt)

**What it does:** Trains `SpatialNoiseGenerator`: **random `z`** for adversarial loss, **fixed `z_fixed`** for periodic **annulus MSE** vs real patches (plus optional interior L1). Saves **`gopt_bundle.pt`** containing `G`, `D`, `z_fixed`, `hole`, `seed_side`, `scale_factor`.

**When:** You want **paper-style** microstructure inpainting with a **saved latent** tied to boundary training.

**Example:**

```bash
python train_gopt.py --images data/train_images --config configs/default.yaml
```

**Outputs:**

- `checkpoints/gopt/gopt_last.pt`, `checkpoints/gopt/gopt_bundle.pt`

---

### 8.4 `optimize_zopt.py` — z-opt on one image

**What it does:** Loads `gopt_bundle.pt`, freezes `G`, initialises `z` from `z_fixed` (or random if missing), runs **latent Adam + cosine schedule** with annulus + interior + Gaussian anchor losses, saves a small tensor dict.

**When:** After G-opt, for **best per-image** fit (thesis figures, ablations).

**Example:**

```bash
python optimize_zopt.py --gopt_bundle checkpoints/gopt/gopt_bundle.pt --image data/test/slide_010.png --out outputs/zopt_result.pt
```

**Outputs:**

- `outputs/zopt_result.pt` (contains `z`, `fused`, `image`, `hole` tensors on CPU by default).

---

### 8.5 `infer.py` — glue: segment → inpaint → optional z-opt

**What it does:**

- Loads one image, runs preprocessing.
- If `--unet_ckpt` is set: predicts mask, refines it, saves `mask.png`.
- If `--gan_ckpt` is set: runs mask-guided GAN and saves `inpaint_gan.png`.
- If `--gopt_bundle` is set: runs spatial G with saved `z_fixed`, saves `inpaint_gopt.png`.
- If additionally `--run_zopt`: runs z-opt starting from `z_fixed`, saves `inpaint_zopt.png` and `zopt_bundle.pt`.
- If `--use_cyclic` and U-Net + GAN are present: re-segment loop (`training/cyclic.py`).

**When:** Quick end-to-end demo or batch wrapper you extend yourself.

**Examples:**

```bash
python infer.py --image data/test/slide_010.png --unet_ckpt checkpoints/unet/unet_best.pt --gan_ckpt checkpoints/gan/gan_weights.pt --out_dir outputs/run_gan
```

```bash
python infer.py --image data/test/slide_010.png --unet_ckpt checkpoints/unet/unet_best.pt --gan_ckpt checkpoints/gan/gan_weights.pt --gopt_bundle checkpoints/gopt/gopt_bundle.pt --run_zopt --out_dir outputs/run_full
```

If you **omit** `--unet_ckpt`, a **central square placeholder mask** is used (demo only, not for research claims).

**Outputs (under `--out_dir`):** `mask.png`, `inpaint_gan.png`, `inpaint_gopt.png`, `inpaint_zopt.png` (as applicable), `grid.png`.

---

### 8.6 `evaluate.py` — numerical quality

**What it does:** Loads `pred` and `gt` **through the same preprocessor** as training (so intensities match), then prints JSON: PSNR, SSIM, MSE, MAE, Dice, IoU, boundary F1, **S₂ MSE**, **phase error %**, and optionally **FID**.

**When:** Whenever you have a restored image and a **registered** clean reference.

**Example (image pair):**

```bash
python evaluate.py --pred outputs/run_full/inpaint_zopt.png --gt data/test/slide_010_clean.png --out_json outputs/metrics.json --config configs/default.yaml
```

**Example (with FID — needs two folders of many patches):**

```bash
python evaluate.py --pred outputs/dummy.png --gt data/ref/single.png --fid_real_dir data/fid/real --fid_gen_dir data/fid/gen --out_json outputs/metrics_fid.json --config configs/default.yaml
```

Note: for FID, the script still loads `pred`/`gt` for non-FID metrics; the FID value itself compares **distributions** of patches in the two folders.

---

### 8.7 `app.py` — optional web UI

**What it does:** Starts **Gradio**: upload image, paste paths to U-Net and GAN checkpoints, see predicted mask and restoration.

**When:** Demonstrations without writing shell commands.

**How:**

```bash
python app.py
```

Then open the printed local URL in a browser. You must provide valid checkpoint paths in the UI fields.

---

### 8.8 `visualize.py` — training curves from JSON

**What it does:** Thin wrapper: pass a JSON object whose values are lists of floats (e.g. `{"train_loss": [...], "val_dice": [...]}`) and an output image path.

**Example:**

```bash
python visualize.py --curves_json outputs/curves.json --out outputs/curves.png
```

You can generate `curves.json` from your own logging or from TensorBoard exports.

---

## 9. Typical workflows (recipes)

### Minimal demo (no real defects)

1. Put a few micrographs in `data/train_images/`.
2. `python train_gan.py --images data/train_images --config configs/default.yaml`
3. `python infer.py --image data/train_images/a.png --gan_ckpt checkpoints/gan/gan_weights.pt --out_dir outputs/demo`  
   (uses placeholder central mask if no U-Net.)

### Full thesis-style pipeline

1. Label defects → `train_unet.py` with `images/` + `masks/`.
2. `train_gan.py` on clean or mixed micrographs for a strong baseline inpainter.
3. `train_gopt.py` on representative patches for spatial microstructure.
4. For each test micrograph: `infer.py ... --gopt_bundle ... --run_zopt` **or** `optimize_zopt.py`.
5. `evaluate.py` against undamaged or simulated ground truth.
6. Optionally export TensorBoard / JSON curves and plot with `visualize.py`.

---

## 10. G-opt and z-opt (theory, compact)

- **G-opt:** One draw `z_fixed ~ N(0,1)` is kept for the **annulus (and optional interior) content loss** so the generator learns a **consistent** mapping from that seed to **boundary statistics** of real data. The **critic** is trained only on **random** `z` so it cannot trivially memorise artefacts from a fixed outer ring. After training, **`z_fixed` is part of the artefact** you ship with weights (`gopt_bundle.pt`).

- **z-opt:** **Weights frozen.** Only `z` is updated to reduce **annulus + interior** error against **this** ground-truth image while **regularising** `z` toward a Gaussian (combined diagonal energy + MSE to a fresh noise sample). That trades a small amount of latent freedom for **much better** boundary and phase statistics on that frame.

---

## 11. Metrics reference

| Metric | Meaning |
|--------|---------|
| PSNR / SSIM | Global pixel / structural similarity vs reference. |
| MSE / MAE | Average squared / absolute error. |
| Dice / IoU | Overlap of **binarised** pred vs GT (default threshold 0.5). |
| Boundary F1 | Agreement of **edge maps** (continuity proxy). |
| S₂ MSE | Mismatch of **radial two-point correlation** after binarisation (texture / second-order stats). |
| Phase error (%) | Relative error in **bright-phase area fraction**. |
| FID (optional) | Distribution distance in **Inception** feature space between two **folders** of images. |

Calibration note: absolute PSNR/FID values depend on noise, hole size, and registration between `pred` and `gt`. Defaults in `configs/default.yaml` are chosen so that, when training converges, quality typically improves **GAN → G-opt → z-opt**.

---

## 12. Hardware, reproducibility, references, licence

**Hardware:** A mid-range **GPU with ≥4 GB** VRAM is comfortable for 256² training; reduce `batch_size` in YAML if you run out of memory.

**Reproducibility:** `seed` in `configs/default.yaml` is applied via `utils/seeds.set_seed` at the start of training scripts.

**References (context):** Wiley *Journal of Microscopy* microstructure inpainting line of work; public GitHub projects such as `tldr-group/microstructure-inpainter` and `ustb-ai3d/automatic_inpainting` informed **design choices only**—this codebase is not a line-for-line port.

**Licence:** Use for academic / thesis work
