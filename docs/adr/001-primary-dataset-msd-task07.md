# ADR 001: Primary dataset — MSD Task07_Pancreas

- Date: 2026-09-27
- Status: accepted
- Phase: 1

## Context

Need one primary 3D dataset with pancreas AND tumor labels for comparing
SAM-Med3D adaptations vs a U-Net baseline across data fractions.

## Alternatives considered

1. **MSD Task07_Pancreas** (chosen): portal-venous CT, labeled pancreas +
   tumor, 281 labeled training cases + 139 test images (labels withheld),
   CC-BY-SA 4.0, MSKCC source, multiple Kaggle mirrors.
2. **NIH Pancreas-CT**: healthy-pancreas only, no tumor labels. Rejected as
   primary — cannot support the tumor-segmentation objective. Reserved as a
   possible later external-validation set (healthy-only distribution shift).

## Decision

Lock MSD Task07_Pancreas as the primary dataset. All experimental
train/val/test splits derive from the 281 labeled training cases
(70/15/15, seed 42 → 197/42/42). Official 139 challenge-test images are
NOT our test set (labels unavailable); used at most for inference sanity.

## Verified facts (not from memory)

- Authoritative (http://medicaldecathlon.com): pancreas-tumor task, portal
  venous phase CT, MSKCC source, CC-BY-SA 4.0, cite arXiv:1902.09063;
  challenge paper https://doi.org/10.1038/s41467-022-30695-9.
  Note: site text says "282 Training" but 282+139=421 contradicts its own
  "420 volumes" total; the downloadable release contains 281 labeled
  training cases (281+139=420). We use the verified 281.
- Remote Kaggle audit (kernel `rajucode/task07-pancreas-phase-1-dataset-audit-cpu`
  v4, CPU): 281 imagesTr + 281 labelsTr, exact 1:1 match, 0 duplicates,
  0 missing, 0 unreadable; all masks contain labels {0,1,2} only; every case
  has both pancreas and tumor voxels (0 empty-tumor cases); all image/mask
  shapes match; all orientations RAS; 139/139 imagesTs headers readable.
- Mirror `dataset.json` (verbatim fields): name "Pancreas", modality CT,
  labels {"0": "background", "1": "pancreas", "2": "cancer"},
  licence "CC-BY-SA 4.0", reference "Memorial Sloan Kettering Cancer Center",
  release "1.0 04/05/2018", numTraining 281, numTest 139.
- Kaggle mirror chosen: `hansenc/pancreas-task07` — genuine Task07 structure
  (imagesTr/labelsTr/imagesTs + dataset.json), byte-identical file sizes to
  two sibling mirrors, cleanest copy (no macOS `._` junk files), most
  recently updated (2026-09-01). Mirror declares no license itself
  ("unknown"); license taken from the authoritative MSD release above.
  Cosmetic mirror difference: volumes stored uncompressed `.nii` while the
  original `dataset.json` references `.nii.gz`; content verified readable.

## Task labels

0 = background, 1 = pancreas, 2 = pancreatic tumor/cancer.

## Limitations

- Single-center (MSKCC), portal-venous CT only — generalization claims bounded.
- 281 cases is small; 10% fraction ≈ 19 training cases → high variance
  (multi-seed repeats budgeted in Phase 6).
- All labeled cases contain tumor; no healthy-only controls in-distribution.
- Anisotropic spacing (~0.7–0.9 mm in-plane, 2.5 mm axial — verify per-case
  in `results/dataset/task07_inventory.csv`).
- Challenge-test labels withheld → no official leaderboard comparison.

## Implications

- Frozen internal test (42 cases) is the ONLY ground-truth test set.
- Tumor-present-in-every-case simplifies stratification but limits
  healthy-control analysis (NIH set may fill this gap later).
- Splits, inventory, and manifest are tracked; volumes never enter Git.
