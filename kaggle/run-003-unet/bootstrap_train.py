"""Run B bootstrap (GPU full baseline): clone PINNED repo commit, train.

Same robust-bundling approach as Run A. Runs scripts/train_unet.py --mode train.
"""

import os
import subprocess
import sys

COMMIT = "3f0c36d46467e4185927af00e4e55ae552f1d43f"
REPO = "https://github.com/raju-sah/sam-med3d-pancreas-adaptation"
WORK = "/kaggle/working"

subprocess.check_call(["git", "clone", "--quiet", REPO, f"{WORK}/repo"])
subprocess.check_call(["git", "-C", f"{WORK}/repo", "checkout", "--quiet", COMMIT])
print("checked out:", subprocess.run(
    ["git", "-C", f"{WORK}/repo", "rev-parse", "HEAD"],
    capture_output=True, text=True).stdout.strip(), flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "monai", "nibabel"])
os.chdir(f"{WORK}/repo")
subprocess.check_call([sys.executable, "scripts/train_unet.py", "--mode", "train",
                       "--output-dir", f"{WORK}/run"])
print("RUN B DONE", flush=True)
