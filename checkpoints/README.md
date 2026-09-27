# Checkpoints (gitignored)

Model weights live here or under `results/<run>/`. Never committed:
`.gitignore` excludes `checkpoints/` plus `*.pt *.pth *.ckpt *.safetensors`.

Download Kaggle outputs here:

```bash
kaggle kernels output "USERNAME/SLUG" -p checkpoints/<run-name>/
```
