# Microstructure Inpainting — B.Tech Project

Detecting and repairing damaged regions in microstructure images, as the first stage of the B.Tech project
*Prediction of Missing Bone Structure* (IIT Bhubaneswar, supervisor Dr. Kodanda Mangipudi).

> **Based on** [hellodeepuu/Microstructure_inpaintaing](https://github.com/hellodeepuu/Microstructure_inpaintaing)
> (licence: *use for academic / thesis work*). The original documentation is kept unchanged in
> [`docs/ORIGINAL_README.md`](docs/ORIGINAL_README.md). This repository adds a loader bug fix, a damage-generation
> and scoring toolkit, a training-free baseline detector, and the experiments listed below.
> Maintained by Nirmay Nandurkar.

---

## What it does

Two pipelines repair a damaged image with the same trained generator. They differ only in where the damage mask comes from.

| | Pipeline 1 — mask supplied | Pipeline 2 — mask detected |
|---|---|---|
| Input | damaged image **+** mask of the damage | damaged image only |
| Mask from | known (we made the damage) | U-Net segmentation |
| Repair | mask-guided GAN | same mask-guided GAN |
| Use | measuring the generator alone | realistic use, damage location unknown |

Test structures are synthetic **spinodal** (bicontinuous) and **Voronoi** (grain) images from `gen_micro.py`.

---

## Changes in this repository

### 1. U-Net loader fix (`datasets/microscopy.py`)

With `augment: true` in `configs/default.yaml`, the original loader called the random flip/rotate transform **twice**,
once on the image and once on the mask, so each got an independent draw. The mask then usually described the wrong
pixels. Damage that is symmetric under flips and 90° rotations (a centred square) survived; anything else did not.

The fix concatenates image and mask into one tensor and transforms them together.

| Check (spinodal training set `data/unet_images`, 200 samples) | Original loader | Fixed loader |
|---|---|---|
| Image–mask pairs aligned (`check_align.py`) | 12 % | 100 % |

12 % ≈ 1 in 8, as expected from 8 equally likely flip × rotation combinations drawn independently.

### 2. Results — damage detection on the Voronoi test set

20 held-out Voronoi images (`data/unet_vtest_images`), mixed rectangles, blobs and scratches. Both U-Nets use the
same data, config and 60 epochs; only the loader differs.

| Detector | Dice | Missed damage | Over-marked |
|---|---|---|---|
| U-Net, original loader | 0.343 | 73.8 % | 2.3 % |
| U-Net, fixed loader | **0.999** | 0.1 % | 0.1 % |
| Texture rule, no training (`var_detect.py`) | 0.862 | 21.0 % | 0.0 % |

- Fixed vs original U-Net: fixed wins 20/20, paired t = 9.28.
- Fixed U-Net vs texture rule: U-Net wins 20/20, paired t = 3.43. The texture rule fails only on the 3 scratch
  images (Dice ≈ 0.45; scratches are narrower than its window). Elsewhere it scores ≈ 0.93 and loses mainly a rim
  at the damage boundary.

**Limits of this result.**
- Damage is painted with a **flat** grey fill, so it can be found from lack of texture alone. That is why a
  training-free rule reaches 0.86. Real damage is not flat; detection of realistic damage is **untested**.
- The U-Net's advantage over the texture rule rests on 3 scratch images. A larger, scratch-heavy test set is needed.
- U-Net results produced before the loader fix (including the earlier fixed-fill vs random-fill comparison) were
  made with misaligned training data and should be re-run before being relied on.

### 3. Added scripts

| Script | Purpose |
|---|---|
| `gen_micro.py` | Generate synthetic spinodal / Voronoi images |
| `damage.py` | Damage one image (centre square, scratch or blob); saves damaged image + mask |
| `damage_batch.py` | Build a U-Net training/test set with varied damage; writes a manifest CSV |
| `inpaint.py` | Run the generator with a user-supplied mask (Pipeline 1) |
| `composite.py` | Keep real pixels outside the hole, network output inside |
| `var_detect.py` | Training-free detector: flags regions with no texture |
| `check_align.py` | Verify that training images and masks line up after augmentation |
| `mask_score.py` | Score one predicted mask against the truth |
| `mask_score_batch.py` | Score two detectors over a folder, paired t-test, breakdown by damage type and contrast |
| `mask_sheet.py` | Contact sheet: input, true mask, predicted mask, overlay |
| `batch_eval.py` | Sweep hole sizes over many images, mean metrics per size |
| `paired_test.py` | Paired comparison of two `batch_eval.py` runs |
| `smoke_test.py` | Check install and model wiring without data |

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python smoke_test.py
```

A CUDA GPU is strongly recommended. All commands below are Windows `cmd`, run from the project root.

---

## Data layout

`data/`, `checkpoints/` and `outputs/` are git-ignored; regenerate them with the commands below.

| Folder | Contents |
|---|---|
| `data/voronoi_train_images`, `data/spinodal_train_images` | clean synthetic images |
| `data/unet_voronoi_images`, `data/unet_voronoi_masks` | 3000 damaged Voronoi images + masks (U-Net training) |
| `data/unet_images`, `data/unet_masks` | 2000 damaged spinodal images + masks |
| `data/unet_vtest_images`, `data/unet_vtest_masks` | 20 damaged Voronoi test images + masks + manifest |

---

## Usage

**Generate clean images**
```bat
python gen_micro.py --out data\voronoi_train_images --n 3000 --kind voronoi --seed 21
```

**Build a damaged training set for the U-Net**
```bat
python damage_batch.py --clean_dir data\voronoi_train_images --out_images data\unet_voronoi_images --out_masks data\unet_voronoi_masks --n 3000 --fill 128 --seed 7
```

**Check alignment, then train the U-Net** (about 65 min for 3000 images, 60 epochs)
```bat
python check_align.py --images data\unet_voronoi_images --masks data\unet_voronoi_masks
python train_unet.py --images data\unet_voronoi_images --masks data\unet_voronoi_masks --config configs\default.yaml
```
The fixed loader should report about 100 % aligned. Anything well below that means images and masks are not lining up.

**Pipeline 1 — repair with a known mask**
```bat
python damage.py --image data\test_demo\spinodal_00009.png --out_dir data\test_demo --mode center --hole 80 --fill 128
python inpaint.py --ckpt checkpoints\gan\gan_weights.pt --image data\test_demo\spinodal_00009_damaged.png --mask data\test_demo\spinodal_00009_mask.png --out outputs\demo\final.png
python evaluate.py --pred outputs\demo\final.png --gt data\test_demo\spinodal_00009.png --out_json outputs\demo\metrics.json
```

**Pipeline 2 — detect, then repair**
```bat
python infer.py --image data\test_demo\spinodal_00009_damaged.png --unet_ckpt checkpoints\unet\unet_best.pt --gan_ckpt checkpoints\gan\gan_weights.pt --out_dir outputs\unet_demo
```

**Score detectors over the test set**
```bat
for %f in (data\unet_vtest_images\*.png) do python infer.py --image "%f" --unet_ckpt checkpoints\unet\unet_best.pt --out_dir outputs\vtest_unet\%~nf
python var_detect.py --images data\unet_vtest_images --true data\unet_vtest_masks --out_root outputs\var_vtest
python mask_score_batch.py --a outputs\vtest_unet --b outputs\var_vtest --true data\unet_vtest_masks --label_a unet --label_b texture --manifest data\unet_vtest_images_manifest.csv
```
In a `.bat` file, write `%%f` and `%%~nf` instead of `%f` and `%~nf`.

---

## Status and next steps

- **Done:** both pipelines run end to end; loader fixed; detection evaluated on Voronoi test damage.
- **Known limitation:** on Voronoi structures the generator reproduces grain texture but grain boundaries stop at the
  edge of the hole. There is no mechanism yet to enforce structural continuity across the gap.
- **Next:**
  1. Scratch-heavy and non-flat (textured) damage test sets.
  2. Measure whether better masks give better final repairs (Pipeline 2 end to end vs clean originals).
  3. Edge-guided reconstruction for boundary continuity; evaluate the G-opt / z-opt routes.
  4. Phase-field smoothing, then application to bone structure.

---

## Licence

The original code is provided by its author for **academic / thesis use**. No broader licence is granted, so this
repository is not offered for redistribution. Ask the original author before reusing it outside academic work.
