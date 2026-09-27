"""Generate frozen Phase 1 splits from the verified audit inventory.

Reads case IDs from the Kaggle audit CSV (verified remotely), splits
deterministically, writes data/splits/{train,val,test}.json + manifest.
Only IDs/metadata are written — never image data.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.split import canonical_hash, canonical_order, make_split  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import repo_root, resolve  # noqa: E402


def main(cfg_path: str = "configs/split_phase1.yaml") -> None:
    cfg = load_config(resolve(cfg_path))
    inv_path = resolve(cfg["inventory_csv"])
    out_dir = resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ids: list[str] = []
    with open(inv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                raise ValueError(f"inventory has error case: {row['case_id']}")
            if row.get("has_image") != "True" or row.get("has_label") != "True":
                raise ValueError(f"inventory case missing data: {row['case_id']}")
            ids.append(row["case_id"])

    canonical = canonical_order(ids)
    split = make_split(canonical, seed=cfg["seed"], train_frac=cfg["train_frac"], val_frac=cfg["val_frac"])
    manifest = {
        "dataset": cfg["dataset"],
        "dataset_source": cfg["dataset_source"],
        "audit_kernel": cfg["audit_kernel"],
        "audit_kernel_version": cfg["audit_kernel_version"],
        "seed": cfg["seed"],
        "algorithm": "sorted IDs, random.Random(seed).shuffle, "
        f"round(n*{cfg['train_frac']}) train / round(n*{cfg['val_frac']}) val / remainder test",
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "n_labeled_cases": len(canonical),
        "n_train": len(split["train"]),
        "n_val": len(split["val"]),
        "n_test": len(split["test"]),
        "canonical_case_ids_sha256": canonical_hash(canonical),
        "canonical_case_ids": canonical,
        "rules": [
            "split by whole 3D case, never by slice",
            "internal test IDs FROZEN — never train/tune/early-stop on test",
            "future 10/25/50/100% subsets sample ONLY from train",
            "val and internal test identical across all future models",
        ],
    }
    for k in ("train", "val", "test"):
        (out_dir / f"{k}.json").write_text(json.dumps(split[k], indent=1) + "\n", encoding="utf-8")
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"n={len(canonical)} train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")
    print(f"sha256={manifest['canonical_case_ids_sha256']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/split_phase1.yaml")
