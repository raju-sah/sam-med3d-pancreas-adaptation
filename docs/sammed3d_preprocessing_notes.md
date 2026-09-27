# SAM-Med3D upstream preprocessing notes (review only — no integration)

- Upstream: https://github.com/uni-medical/SAM-Med3D
- Inspected commit (main): `f3de1fa10da98e46f49f176773d2b1e306ba131f`
  (dated 2025-09-21, "Update readme.md"; verified via GitHub API).
- Paper: arXiv:2310.15161 (Wang et al., SAM-Med3D). License: Apache-2.0.
- Files read at that commit: `train.py`, `utils/data_loader.py`,
  `utils/prepare_data_from_nnUNet.py`, `utils/infer_utils.py`,
  `medim_val_single.py`. Nothing vendored; this file is notes only.

## Offline preparation (`utils/prepare_data_from_nnUNet.py`)

- Images resampled to **1.5 mm isotropic** (`target_spacing=(1.5,1.5,1.5)`,
  torchio Resample, linear interpolation for the image).
- Masks converted to **per-class BINARY** (`label == class_idx` → 1 else 0),
  resampled with **nearest** interpolation onto the resampled image grid,
  then CropOrPad to the reference image size.
- Classes with physical volume < 10 mm³ are skipped.
- Output layout: one binary task dir per class
  (`data/.../<class>/<dataset>/{imagesTr,labelsTr}`).

## Online training transform (`train.py`, default `img_size=128`)

- `tio.Compose([ToCanonical(), CropOrPad(mask_name='label', (128,128,128)),
  RandomFlip(axes 0,1,2)])` — i.e. canonical orientation, then a
  **label-centered** 128³ crop/pad (label-aware; uses GT at train time).
- CT clamp `tio.Clamp(-1000, 1000)` applied in the data loader, but ONLY when
  the image path contains `"/ct_"` (upstream path convention).
- Normalization per step: `tio.ZNormalization(masking_method=lambda x: x > 0)`
  — foreground-masked z-score computed **per volume on the fly** from the
  image alone (no global statistics, no labels needed).
- Cases whose cropped label sum ≤ `threshold` (1000 in train.py; class
  default 500) are rejected and replaced by a random re-draw.
- Loss `DiceCELoss(sigmoid=True)` and Dice scored at 0.5 on `(gt > 0)`:
  training is **binary per class**. Prompts are sampled from GT, so GT is
  required at train time. Seeds fixed to 2023 in `main()`.

## Inference (`utils/infer_utils.py`, `medim_val_single.py`)

- `data_preprocess`: binarize one category → `Resample(1.5,1.5,1.5)` →
  `ToCanonical()` → `CropOrPad(mask_name='label', (128,128,128))` →
  foreground-masked `ZNormalization`. Same defaults: `crop_size=128`,
  `target_spacing=(1.5,1.5,1.5)`.
- ROI crop at inference ALSO centers on the (binarized) label, and prompt
  points are sampled from GT (`random_sample_next_click`); the readme states
  GT is required for prompts (else a fake GT / central click fallback).
- `data_postprocess`: prediction resampled back to the original grid with
  **nearest** interpolation; original spacing/origin/direction restored via
  SimpleITK metadata. Per-category binary predictions merged by writing the
  category index where each ROI mask is 1.

## Takeaways for our Phase 2 design

1. Canonical spatial target: 1.5 mm iso + RAS canonical + 128³ ROI.
   Our Task07 data is all RAS already (audit); canonicalize anyway.
2. Label handling is binary-per-class with nearest-only resampling — our
   multiclass {0,1,2} GT must be binarized in-memory per target, never
   overwritten on disk.
3. Label-aware 128³ cropping is TRAIN/ROI-scoped upstream too (it needs a
   mask). Our inference-safe deterministic pipeline must NOT depend on GT;
   label-aware sampling stays training-only (same separation as upstream).
4. Intensity: CT clamp [-1000,1000] + per-volume foreground-masked z-score
   needs no global stats and no labels — directly reusable as our common
   deterministic intensity step; train percentiles will justify the clamp.
5. Upstream `ToCanonical` uses torchio (RAS+ / LPS? — torchio canonical is
   RAS); our MONAI `Orientationd(axcodes='RAS')` is the equivalent.
