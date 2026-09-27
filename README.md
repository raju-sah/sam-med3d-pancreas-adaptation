# Parameter-Efficient Adaptation of SAM-Med3D for 3D Pancreatic / Pancreatic Tumor Segmentation

> Status (Phase 0): project scaffold + environment validation only.
> All experiments below are **PLANNED**. No metrics exist yet. No results are reported.

## Research problem

3D pancreatic (and pancreatic tumor) segmentation from CT is clinically valuable
and technically hard: small organ, low boundary contrast, high anatomical variance,
class imbalance. SAM-Med3D is a 3D foundation model for volumetric medical
segmentation, but full fine-tuning is expensive and data-hungry. This project asks:
how far do **parameter-efficient / frozen-component adaptations** of SAM-Med3D go
compared to full fine-tuning and a standard 3D U-Net baseline — especially
under **limited training data**?

## Motivation

- Pancreas segmentation is a canonical hard 3D task; strong results transfer credibility.
- Foundation-model adaptation (adapters, prompt tuning, partial freezing) is the
  practically relevant regime for labs without massive compute/data.
- A clean, reproducible comparison (frozen vs full vs PEFT vs U-Net, across data
  fractions) is portfolio-grade evidence of research engineering ability.

## Planned dataset / task

- Task: 3D semantic segmentation of pancreas (and tumor where labels exist).
- Candidate sources (final choice in Phase 1): **MSD Task 7 Pancreas** or
  **NIH Pancreas-CT**. Decision criteria: license permitting research use,
  availability on Kaggle, label quality for pancreas (+ tumor).
- Fixed train/val/test splits with frozen test IDs; test set never touched
  during training or model selection.

## Planned baselines and experiments

1. **Pretrained SAM-Med3D** — zero-shot / prompt-based inference, no training.
2. **Fully fine-tuned SAM-Med3D** — all weights updated.
3. **Parameter-efficient / frozen-component SAM-Med3D variants** — e.g. frozen
   encoder + trained decoder/head, lightweight adapters; exact variants fixed
   in Phase 7 after literature + codebase review.
4. **Standard 3D U-Net baseline** — MONAI-based, trained from scratch.

Limited-data axis (Phase 6): ~10% / 25% / 50% / 100% of training data,
same fixed seed subsampling, all models compared at each fraction.

## Planned evaluation metrics

- Segmentation: **Dice, IoU, HD95, Precision, Recall**
- Cost: **inference time, training time, GPU memory (if practical),
  total parameters, trainable parameters**
- Every reported number traces to a logged run: config + checkpoint +
  `results/metrics/*.csv`. No log, no claim.

## Reproducibility principles

- Config-driven experiments (`configs/`), fixed random seeds (`src/utils/seed.py`).
- Official SAM-Med3D code/docs + PyTorch/MONAI; no reinvented loaders.
- Test-set isolation enforced (see `AGENTS.md`).
- Major design decisions recorded in `docs/` (ADRs).

## Compute workflow: local vs Kaggle

- **Local (this machine):** code, configs, docs, lightweight tests, result analysis.
  No 3D training locally — machine has no NVIDIA GPU.
- **Kaggle GPU:** all heavy training via Kaggle CLI kernels.
  Workflow documented in `kaggle/README.md`.
- Git tracks code/configs/docs/metric CSVs/small figures only.
  Datasets, checkpoints, credentials, `.env`, large artifacts are gitignored.

## Repo layout

```text
├── README.md            # this file
├── AGENTS.md            # permanent agent instructions
├── pyproject.toml       # packaging; heavy deps are `kaggle` extras
├── configs/             # experiment configs (base.yaml = Phase 0 sanity)
├── src/                 # data | models | training | evaluation | utils
├── scripts/             # check_env.py + future helpers
├── tests/               # Phase 0 sanity tests (stdlib unittest)
├── notebooks/           # exploratory analysis (tracked sources only)
├── kaggle/              # kernel scaffold + push workflow docs
├── results/metrics|figures/  # tracked CSVs + small figures
├── data/ checkpoints/   # gitignored (READMEs explain layout)
├── docs/research_plan.md
└── paper/
```

## Current status

- [x] Phase 0: scaffold, env validation, Kaggle CLI/auth check, sanity tests
- [ ] Phase 1: dataset acquisition and validation — NOT STARTED
- [ ] Phases 2–10: see `docs/research_plan.md` — NOT STARTED
