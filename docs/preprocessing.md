# Preprocessing (Phase 2)

Frozen policy: `configs/preprocess_phase2.yaml`. Rationale: ADR 002.
Upstream review: `docs/sammed3d_preprocessing_notes.md`.
Implementation: `src/data/preprocess.py`. Evidence: TRAIN-only profile
`results/dataset/phase2_train_profile.{json,csv}` (n=197, seed 42).

## Raw data representation

MSD Task07_Pancreas CT volumes, NIfTI, all RAS orientation (audit-verified).
Native spacing heterogeneous: in-plane 0.61–0.98 mm, axial 0.7–7.5 mm.
Labels discrete {0 background, 1 pancreas, 2 mass/tumor}; every labeled case
contains both foreground classes. Intensities: air/out-of-field ≤ −1000,
soft tissue < 1000 (per-case p99.5 max 700.4), bone/contrast/metal up to 4009.

## Common preprocessing (Layer A, model-independent)

1. Load NIfTI pair; verify 3D + image/label shape match; validate labels.
2. Channel-first `(1,D,H,W)`; preserve affine/zooms in metadata.
3. Canonical orientation RAS (`Orientationd`).
4. Resample to **1.5 mm isotropic**: image bilinear, label nearest, then
   re-assert {0,1,2}.
5. Clamp CT to **[-1000, 1000]** (soft tissue fully preserved; air/bone/metal
   handled per evidence in ADR 002). Implemented as MONAI
   `ScaleIntensityRangeD(..., clip=True)` with equal in/out bounds, which is
   exactly a clamp — used because the remotely installed MONAI 1.6.0 has no
   `ClipIntensityD/d` dict alias (verified at smoke-test time).
6. Per-volume foreground-masked z-score (`nonzero` ⇒ image > 0 after clamp):
   image-only, no labels, no global statistics → inference-safe.
7. Deterministic: no randomness in this path (seed recorded anyway).

## U-Net future expectations (Layer B)

Standard MONAI training pipeline in Phase 3 will reuse Layer A, then add
training-only label-aware patch sampling (foreground-biased, e.g. to counter
the 0.2% median foreground fraction) and augmentation. Inference path stays
exactly Layer A (+ fixed-size tiling decided in Phase 3).

## SAM-Med3D compatibility (Layer B)

Upstream canonical input: 1.5 mm iso + RAS + per-volume fg-masked z-score +
CT clamp (when path matches CT convention — ours applies it explicitly) +
per-class binary masks + label-centered 128³ ROI for training/prompted
inference. Our Layer A matches upstream through normalization; per-class
binarization happens in-memory via `derive_masks`; label-dependent 128³
cropping stays training/inference-ROI-scoped, never in the deterministic
preprocessing path (same separation as upstream).

## Orientation handling

`Orientationd(axcodes='RAS')` always applied (no-op on current data, guards
future/external data). Affines preserved alongside arrays for post-hoc
resampling back to native grids.

## Spacing / intensity decisions

See ADR 002. In short: 1.5_iso (min tumor keeps 195 voxels ≥ 50-voxel rule);
clamp [-1000,1000] (soft-tissue preservation rule); fg-masked z-score.

## Interpolation rules

Images: bilinear (trilinear in 3D). Labels: nearest ONLY. Assert {0,1,2}
after every label resample — unit-tested locally (`test_nearest_resample…`)
and smoke-tested remotely on real validation cases.

## Label representations

- `multiclass`: original {0,1,2}, never overwritten on disk.
- `whole`: (label==1)|(label==2) — pancreas region incl. mass.
- `tumor`: (label==2) — mass/tumor binary target.
- SAM-Med3D phases derive per-class binary masks from these in memory.

## Train-only statistics rule

All preprocessing parameters derive from the 197 train IDs only.
Val (42) used post-freeze for transform-correctness smoke test (3 cases).
Test (42) volumes never opened in Phase 2; test.json read only for
exclusion assertions. Guard `assert_not_test` + split-separation tests.

## Validation usage rule

Post-freeze val smoke test may verify execution correctness only
(geometry, labels, finite values, sane dims). A software bug found there
may be fixed; parameters must NOT be re-tuned on val appearance/performance.

## Test isolation

See AGENTS.md §4 and dataset card leakage rules. Phase 2 compliance:
kernel asserts 197/42/42 disjoint splits; profiling/figures train-only;
no test volume opened (verified: kernel code paths reference TEST_IDS
only in the disjointness assert).

## Visualization methodology

Representative train cases chosen deterministically from the train profile
(min/median/max tumor volume, min pancreas, fixed-seed random; deduped):
pancreas_398, pancreas_227, pancreas_409, pancreas_377, pancreas_364.
Per case: 3 views (sagittal/coronal/axial at foreground centroid) ×
[raw CT (display window [-150,250] — display only), GT overlay, pancreas,
mass/tumor]; plus raw-vs-preprocessed middle-slice comparison.
`results/figures/phase2/*.png`. Display windowing never touches data.

## Limitations

- 1.5 mm resampling upsamples thick-slice (up to 7.5 mm axial) cases ×5 in
  z — interpolated detail, documented, not new information.
- fg-masked z-score normalizes by non-air voxels; extreme metal artifacts
  remain clamped but present — noted for Phase 3 augmentation review.
- Percentiles estimated on strided samples (documented in profile JSON);
  min/max/mean/std exact.
