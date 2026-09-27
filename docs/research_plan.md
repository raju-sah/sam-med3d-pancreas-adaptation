# Research plan — SAM-Med3D parameter-efficient adaptation for 3D pancreas segmentation

> All phases except Phase 0 are PLANNED. No results exist. Validation criteria
> are gates: a phase is done only when its criteria hold.

## Phase 0: project / environment setup — DONE

- Objective: reproducible scaffold + validated local/Kaggle workflow, zero training.
- Tasks: repo structure, `.gitignore`, `pyproject.toml`, configs, `src/utils`,
  sanity tests, `kaggle/` scaffold, README/AGENTS.md, this plan.
- Artifacts: this repo at v0.1.0 scaffold; passing `python3 -m unittest discover`.
- Validation: imports work, config loads, Kaggle CLI auth verified without
  exposing secrets, no training code, no data downloaded.
- Risks: local disk 92% full (18 GB free) — monitor; no local GPU (by design).

## Phase 1: dataset acquisition and validation

- Objective: choose and freeze the dataset + splits.
- Tasks: pick MSD Task 7 Pancreas vs NIH Pancreas-CT (license, Kaggle
  availability, tumor labels); scripted download (Kaggle side); checksum +
  volume/label inventory; fixed train/val/test split with frozen test IDs.
- Artifacts: `data/README.md` update, `data/splits/*.json`, dataset card in `docs/`.
- Validation: N volumes counted, shapes/spacings logged, splits reproducible
  from seed, test IDs untouched by any training code.
- Risks: license restrictions; large download size vs disk; label inconsistencies.

## Phase 2: preprocessing and visualization

- Objective: reproducible preprocessing + sanity visuals.
- Tasks: resampling, intensity normalization, cropping/patching pipeline
  (MONAI transforms, config-driven); slice/montage + 3D stat plots to
  `results/figures/`; preprocessing config frozen.
- Artifacts: `src/data/` pipeline, `configs/preprocess.yaml`, figures.
- Validation: pipeline deterministic given seed; visual check of N random
  cases incl. edge cases (small pancreas, tumor).
- Risks: spacing heterogeneity; foreground-background imbalance in patches.

## Phase 3: 3D U-Net baseline

- Objective: reference performance + training harness all later runs reuse.
- Tasks: MONAI 3D U-Net, Dice+CE loss, standard augmentation; train on Kaggle
  GPU at 100% data; log time/memory/params.
- Artifacts: `src/models/unet.py`, `src/training/`, `configs/unet_*.yaml`,
  `results/metrics/unet_*.csv`, checkpoint (gitignored).
- Validation: converges without NaN; overfits a single batch (capacity check);
  test evaluated exactly once per final model.
- Risks: under-tuned baseline flatters later comparisons — budget fair tuning.

## Phase 4: pretrained SAM-Med3D inference

- Objective: zero-shot reference point.
- Tasks: integrate official SAM-Med3D weights/code; prompt-based inference
  protocol (document prompts); run on test via Kaggle; log inference time/params.
- Artifacts: `src/models/sammed3d.py`, `configs/sammed3d_infer.yaml`, metrics CSV.
- Validation: weight checksums recorded; inference reproducible; prompt
  protocol documented so numbers are interpretable.
- Risks: weight license/availability; prompt sensitivity; GPU memory on large volumes.

## Phase 5: SAM-Med3D full fine-tuning

- Objective: upper-bound adaptation performance + cost.
- Tasks: full-weight fine-tune on 100% data (Kaggle); LR/schedule search
  (small, documented); log train time, memory, trainable params.
- Artifacts: `configs/sammed3d_full_*.yaml`, metrics CSV, checkpoints.
- Validation: beats inference baseline; no test leakage; run reproducible
  from config + seed.
- Risks: catastrophic forgetting / instability; high GPU-hour cost.

## Phase 6: limited-data experiments

- Objective: data-efficiency comparison across all models so far.
- Tasks: fixed-seed subsamples at ~10/25/50/100%; train U-Net + full FT
  (+ PEFT variants once Phase 7 defines them) at each fraction; same protocol.
- Artifacts: `data/splits/fraction_*.json`, per-fraction configs + CSVs,
  data-efficiency curves in `results/figures/`.
- Validation: subsamples nested (10% ⊂ 25% ⊂ 50% ⊂ 100%) from one seed;
  identical eval protocol per fraction.
- Risks: high variance at 10% — repeat with ≥2 seeds if budget allows.

## Phase 7: parameter-efficient adaptation

- Objective: the core contribution — PEFT/frozen-component variants.
- Tasks: literature + SAM-Med3D code review; implement variants (e.g. frozen
  encoder + trained decoder, lightweight adapters); compare vs full FT and
  U-Net at each data fraction; report params trained vs total.
- Artifacts: `src/models/adapters.py` (or equivalent), configs, CSVs, ADR
  justifying variant choices.
- Validation: trainable-param counts asserted in code/tests; each variant
  reproducible; cost metrics logged alongside accuracy.
- Risks: variant space is large — cap at 2–3 principled choices; document
  rejected options in one line each.

## Phase 8: ablations / error analysis

- Objective: understand *where* methods win/lose.
- Tasks: ablate prompts, patch size, augmentation, frozen depth; stratify
  errors (small pancreas, tumor vs non-tumor, boundary HD95 cases); qualitative
  figure panel of best/worst cases.
- Artifacts: ablation CSVs, error-analysis figures, notes in `docs/`.
- Validation: each ablation changes exactly one factor; claims backed by CSVs.
- Risks: over-claiming from small test set — report CIs / per-case spreads.

## Phase 9: final reproducibility pass

- Objective: anyone can rerun the headline numbers.
- Tasks: pin dependency versions, re-run smoke tests, verify every CSV
  regenerates or traces to a logged run; clean dead code/configs.
- Artifacts: locked requirements, green test suite, run index in `results/`.
- Validation: fresh-clone + Kaggle re-run reproduces headline table within
  documented tolerance.
- Risks: bit-exactness across GPUs — document tolerance, don't chase it.

## Phase 10: paper / README / results presentation

- Objective: portfolio-grade presentation for professors.
- Tasks: results tables + figures, methods write-up in `paper/`, README
  status update with real numbers (linked to CSVs), docs index.
- Artifacts: paper draft, final README, `results/` index.
- Validation: every number in text links to a CSV; no PLANNED items cited
  as done; citations real and read.
- Risks: scope creep — freeze methods at Phase 9; writing changes nothing technical.
