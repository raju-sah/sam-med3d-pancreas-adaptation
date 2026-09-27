# Data (gitignored)

This directory holds datasets. Nothing under `data/` is committed to Git
(see `.gitignore`: `data/`, `*.nii`, `*.nii.gz`, ...).

Planned source (Phase 1): public pancreas CT collection
(e.g. MSD Task 7 Pancreas or NIH Pancreas-CT — final choice documented in Phase 1).
Download happens on Kaggle or via scripted fetch, never hand-copied blobs.

Expected layout (Phase 1+):

```text
data/
  raw/        # untouched downloads, checksums recorded
  processed/  # resampled / cropped / normalized volumes
  splits/     # train/val/test JSONs with fixed seed; test IDs frozen
```
