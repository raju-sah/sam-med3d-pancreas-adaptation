# Dataset card — MSD Task07_Pancreas (primary)

## Identity

- Dataset: Medical Segmentation Decathlon, Task 7 — Pancreas (Task07_Pancreas)
- Modality: CT, portal venous phase
- Targets: pancreas + pancreatic tumor/cancer (multi-class 3D segmentation)
- Label mapping (from mirror `dataset.json`, confirmed by voxel audit):
  `0 = background`, `1 = pancreas`, `2 = cancer`

## Sources

- Original: Memorial Sloan Kettering Cancer Center, via the MSD challenge
  (http://medicaldecathlon.com). Release 1.0, 04/05/2018.
- License (original release): CC-BY-SA 4.0. Cite arXiv:1902.09063.
- Kaggle mirror used for compute: `hansenc/pancreas-task07`
  (https://www.kaggle.com/datasets/hansenc/pancreas-task07).
  Chosen for genuine Task07 structure, byte-identical sizes to sibling
  mirrors, no junk files, most recent update. Mirror itself declares no
  license; terms inherited from the original release above.

## Verified counts (remote CPU audit, kernel v4)

- Labeled training cases: **281** (281 imagesTr + 281 labelsTr, exact match)
- Challenge-test images: **139** (headers readable; labels withheld)
- Duplicates / missing / unreadable / shape mismatches: **0**
- Label values observed: **{0, 1, 2} only**
- Cases with pancreas voxels: 281/281. Cases with tumor voxels: 281/281
  (no empty-tumor cases). Orientation: all RAS. All volumes 3D.
- Artifacts: `results/dataset/task07_inventory.csv` (per-case),
  `results/dataset/task07_summary.json` (aggregate + dataset.json copy).

## Official challenge-test handling

The 139 imagesTs have no public labels and are NOT our test set.
They may be used later for inference-only sanity checks, never for scoring.

## Our internal splits (FROZEN)

- Source: the 281 labeled cases only. Method: sort IDs, shuffle with
  `random.Random(42)`, 70/15/15 → **train 197 / val 42 / test 42**.
- Files: `data/splits/{train,val,test}.json`, `data/splits/split_manifest.json`
  (canonical SHA-256 `a46ad401…8e4a1c3a72`, identical remote and local).
- Generator: `scripts/make_splits.py` (`configs/split_phase1.yaml`).
- Regression test `tests/test_split.py::FrozenSplitRegression` fails on any
  silent alteration of the frozen split.

## Known limitations

Single-center portal-venous CT; 281 cases (small — expect variance at the
10% fraction); tumor present in every labeled case; anisotropic voxels;
no healthy in-distribution controls; challenge labels withheld.

## Leakage rules

Test IDs frozen: never train, tune, early-stop, or model-select on test.
Future 10/25/50/100% subsets sample ONLY from the 197 train IDs.
Val/test identical across all models. Post-freeze test predictions allowed
only for final scoring + qualitative error figures — never for re-tuning
(see AGENTS.md §4).

## Directory structure (mirror root `Dataset007_Pancreas/`)

```text
Dataset007_Pancreas/
  dataset.json
  imagesTr/pancreas_*.nii   # 281
  labelsTr/pancreas_*.nii   # 281
  imagesTs/pancreas_*.nii   # 139, no labels
```

Note: mirror stores uncompressed `.nii`; original names reference `.nii.gz`.

## Reproducibility

Audit code: `kaggle/run-001-audit/audit_task07.py` (kernel v4, CPU-only,
`enable_gpu=false`). Split code + config + manifest above. Re-running
`scripts/make_splits.py` against the same inventory reproduces splits exactly.
