# Kaggle compute workflow (Phase 0 scaffold — no runs submitted yet)

Local machine = code + config + analysis. Kaggle GPU = all heavy training.
Never train 3D models locally. Never commit secrets, data, or checkpoints.

## Auth (already verified Phase 0, details never printed)

- Kaggle CLI 2.2.4 installed at `~/.local/bin/kaggle`.
- Auth uses `~/.kaggle/access_token` (OAuth, CLI 2.x style). No `kaggle.json` present.
- Verified safe: `kaggle kernels list --mine --page-size 1` succeeds.
- Never `cat` / `print` / commit anything under `~/.kaggle/`.
- Never paste API keys into source files. No `KAGGLE_*` env vars currently set.

## Directory contract

- `kaggle/` holds everything needed for `kaggle kernels push`.
- A push directory must contain the entry script (e.g. `train.py`) plus
  `kernel-metadata.json` (real file, created from the `.example.json` template).
- The real `kernel-metadata.json` is NOT committed yet (created at push time).
- Copy template: `cp kaggle/kernel-metadata.example.json kaggle/<run-dir>/kernel-metadata.json`
  then set `id`, `title`, `code_file`, `enable_gpu=true`, dataset sources.

## Eventual commands (PLANNED, do not run in Phase 0)

```bash
# 1. Prepare push dir (Phase 3+, when train.py exists)
cp kaggle/kernel-metadata.example.json kaggle/run-001/kernel-metadata.json
# edit id/title/dataset_sources in kaggle/run-001/kernel-metadata.json

# 2. Push + run on Kaggle GPU
kaggle kernels push -p kaggle/run-001

# 3. Monitor
kaggle kernels status "USERNAME/SLUG"
kaggle kernels logs "USERNAME/SLUG"

# 4. Download outputs
kaggle kernels output "USERNAME/SLUG" -p results/kaggle-run-001/

# 5. Iterate: edit locally, push again (push = update for existing slug)
kaggle kernels push -p kaggle/run-001
```

## Rules

- One push dir per experiment run (`kaggle/run-00N/`), keeps history auditable.
- Config file used for the run is copied into the push dir and logged with results.
- Dataset attaches via `dataset_sources` in metadata, never via Git.
- Checkpoints download to `checkpoints/` or `results/` (both gitignored as binaries).
- Metric CSVs copied to `results/metrics/` (tracked) for the paper trail.
