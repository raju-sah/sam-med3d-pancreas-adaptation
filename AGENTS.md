# AGENTS.md — permanent instructions for coding agents on this repo

## 1. Research objectives (do not redesign without user approval)

Compare on 3D pancreatic / pancreatic tumor segmentation:
1. Pretrained SAM-Med3D (inference only)
2. Fully fine-tuned SAM-Med3D
3. Parameter-efficient / frozen-component SAM-Med3D variants
4. Standard 3D U-Net baseline (MONAI)

Plus a limited-data axis (~10/25/50/100% of training data).
Metrics: Dice, IoU, HD95, precision, recall, inference time, training time,
GPU memory if practical, total + trainable parameters.

## 2. Repository conventions

- Modular code: `src/data|models|training|evaluation|utils`. No monolith scripts.
- Config-driven experiments: every run has a YAML in `configs/`; no hardcoded
  hyperparameters in training code. Fixed seeds via `src/utils/seed.py`.
- Heavy 3D deps (torch, MONAI) live in `pyproject.toml` `[project.optional-dependencies]`
  under `kaggle`; keep local imports light so tests run without GPU/torch.
- Small diffs, fewest files. Reuse existing helpers before writing new ones.
- Mark deliberate shortcuts with a `ponytail:` comment naming the ceiling.

## 3. Reproducibility requirements

- Fixed seed everywhere; record seed in config and logs.
- Each experiment run logs: config copy, git SHA, data split IDs, metric CSV
  under `results/metrics/`. No log = did not happen.
- Prefer official SAM-Med3D code/docs and established tools (PyTorch/MONAI).

## 4. Test-set isolation (hard rule)

- Test IDs are frozen once in Phase 1. Never train on, tune on, or early-stop on test.
- Never print or commit test labels/predictions beyond aggregate metrics.
- Any split change requires user approval + versioned split file.

## 5. Experiment logging requirements

- Metric CSVs in `results/metrics/` and small figures in `results/figures/` are
  tracked in Git. Checkpoints, volumes, large logs are NOT (see `.gitignore`).
- Each CSV header must include enough context (run id, config ref, split, date)
  or sit beside a run README with that info.

## 6. Kaggle compute rules

- No 3D training on the local machine (no GPU). Heavy runs go to Kaggle GPU
  via CLI kernels; workflow in `kaggle/README.md`.
- Do NOT `kaggle kernels push` without explicit user approval.
- One push dir per run (`kaggle/run-00N/`) with its config copy.
- Never commit `kernel-metadata.json` containing private slugs without approval;
  the `.example.json` template is the tracked reference.

## 7. Secret handling

- Never print, log, or commit: `~/.kaggle/*`, API tokens, `.env`, keys.
- Verify auth safely: `kaggle kernels list --mine --page-size 1`
  (proves auth without exposing anything). Never `cat` credential files.
- Never ask the user to paste keys into source files.

## 8. No fabrication (hard rule)

- Never report Dice/IoU/HD95 or any metric unless produced by code in this repo
  and backed by a logged artifact. Mark everything else PLANNED.
- Never fabricate citations: only cite papers/docs actually read; record URLs.
- Never invent dataset stats, checkpoint sizes, or run times.

## 9. Design decisions

- Document major decisions (dataset choice, split strategy, PEFT variants,
  metric definitions) in `docs/` as short ADRs: context → decision → consequences.
- Communication: terse, technical, no fluff. Code and commit messages in normal style.
