"""Run A bootstrap (GPU smoke): clone PINNED repo commit, install deps, audit+smoke.

Robust bundling: the kernel carries only this file; all training source and
configs come from the pinned GitHub commit, recorded into every output.
"""

import os
import subprocess
import sys

COMMIT = "fa4dee6b221a803b4c1ca65c9e7790981d37c868"
REPO = "https://github.com/raju-sah/sam-med3d-pancreas-adaptation"
WORK = "/kaggle/working"

subprocess.check_call(["git", "clone", "--quiet", REPO, f"{WORK}/repo"])
subprocess.check_call(["git", "-C", f"{WORK}/repo", "checkout", "--quiet", COMMIT])
print("checked out:", subprocess.run(
    ["git", "-C", f"{WORK}/repo", "rev-parse", "HEAD"],
    capture_output=True, text=True).stdout.strip(), flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "monai", "nibabel"])
os.chdir(f"{WORK}/repo")
run = f"{WORK}/run"
subprocess.check_call([sys.executable, "scripts/train_unet.py", "--mode", "audit",
                       "--output-dir", run])
subprocess.check_call([sys.executable, "scripts/train_unet.py", "--mode", "smoke",
                       "--output-dir", run])
print("RUN A DONE", flush=True)
