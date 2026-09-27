"""Phase 0 environment reporter. Prints facts, never secrets."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return (out.stdout or out.stderr or "").strip().splitlines()[0:1] and "\n".join(
            (out.stdout or out.stderr).strip().splitlines()[:5]
        )
    except Exception as e:
        return f"unavailable ({e})"


def main() -> None:
    print(f"os: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"python: {sys.version.split()[0]} @ {sys.executable}")
    print(f"pip: {'found' if shutil.which('pip') else 'missing'}")
    print(f"uv: {_run(['uv', '--version'])}")
    print(f"git: {_run(['git', '--version'])}")
    print(f"kaggle: {_run(['kaggle', '--version'])}")
    kaggle_dir = os.path.expanduser("~/.kaggle")
    has_token = os.path.isfile(os.path.join(kaggle_dir, "access_token"))
    has_json = os.path.isfile(os.path.join(kaggle_dir, "kaggle.json"))
    print(f"kaggle_auth_files: access_token={has_token} kaggle.json={has_json} (contents never shown)")
    try:
        import torch  # noqa: F401

        print("torch: installed")
    except ImportError:
        print("torch: not installed (expected locally; installed on Kaggle GPU)")
    print(f"repo: {os.getcwd()}")


if __name__ == "__main__":
    main()
