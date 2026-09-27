"""Phase 1 remote audit of the MSD Task07_Pancreas Kaggle mirror.

Runs on Kaggle CPU. Reads dataset structure + every labeled case header and
voxels, writes SMALL machine-readable outputs only (CSV + JSON).
No preprocessing, no resampling, no training.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes

INPUT_ROOT = Path("/kaggle/input")
OUT_DIR = Path("/kaggle/working")


def find_task_root() -> Path:
    """Locate the attached dataset dir containing dataset.json (any nesting)."""
    print(f"input listing: {[p.name for p in INPUT_ROOT.iterdir()]}", flush=True)
    for dirpath, dirnames, filenames in os.walk(INPUT_ROOT):
        print(f"walk: {dirpath} dirs={dirnames[:10]} files={filenames[:10]}", flush=True)
        if "dataset.json" in filenames:
            return Path(dirpath)
        if len(Path(dirpath).relative_to(INPUT_ROOT).parts) > 5:
            dirnames[:] = []
            continue
    raise FileNotFoundError(f"no dataset.json found under {INPUT_ROOT}")


def stem(p: Path) -> str:
    name = p.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return p.stem


def main() -> int:
    task_root = find_task_root()
    print(f"task_root: {task_root}", flush=True)
    with open(task_root / "dataset.json", encoding="utf-8") as f:
        dataset_json = json.load(f)

    images_tr = sorted((task_root / "imagesTr").glob("pancreas_*"))
    labels_tr = sorted((task_root / "labelsTr").glob("pancreas_*"))
    images_ts = sorted((task_root / "imagesTs").glob("pancreas_*")) if (task_root / "imagesTs").is_dir() else []

    image_ids = [stem(p) for p in images_tr]
    label_ids = [stem(p) for p in labels_tr]
    ts_ids = [stem(p) for p in images_ts]

    rows: list[dict] = []
    seen_label_values: set[int] = set()
    orientations: set[str] = set()
    mismatches: list[str] = []
    unreadable: list[str] = []

    for cid in image_ids:
        img_path = task_root / "imagesTr" / f"{cid}.nii"
        if not img_path.is_file():
            img_path = task_root / "imagesTr" / f"{cid}.nii.gz"
        lbl_path = task_root / "labelsTr" / f"{cid}.nii"
        if not lbl_path.is_file():
            lbl_path = task_root / "labelsTr" / f"{cid}.nii.gz"
        row: dict = {"case_id": cid, "has_image": img_path.is_file(), "has_label": lbl_path.is_file()}
        if not (row["has_image"] and row["has_label"]):
            mismatches.append(cid)
            rows.append(row)
            continue
        try:
            img = nib.load(str(img_path))
            lbl = nib.load(str(lbl_path))
            ishape, lshape = tuple(img.shape), tuple(lbl.shape)
            row["image_shape"] = "x".join(map(str, ishape))
            row["mask_shape"] = "x".join(map(str, lshape))
            row["ndim_image"] = len(ishape)
            row["ndim_mask"] = len(lshape)
            row["shapes_match"] = ishape == lshape
            if ishape != lshape:
                mismatches.append(cid)
            zooms = tuple(round(float(z), 4) for z in img.header.get_zooms()[:3])
            row["spacing"] = "x".join(map(str, zooms))
            codes = "".join(aff2axcodes(img.affine))
            row["orientation"] = codes
            orientations.add(codes)
            ldata = np.asarray(lbl.dataobj)
            uniq, counts = np.unique(ldata, return_counts=True)
            vals = [int(v) for v in uniq]
            seen_label_values.update(vals)
            row["label_values"] = ",".join(map(str, vals))
            cmap = dict(zip(vals, [int(c) for c in counts]))
            row["n_pancreas_voxels"] = cmap.get(1, 0)
            row["n_tumor_voxels"] = cmap.get(2, 0)
            row["empty_tumor"] = cmap.get(2, 0) == 0
            row["n_voxels_total"] = int(ldata.size)
            idata = np.asarray(img.dataobj, dtype=np.float64)
            row["img_min"] = float(idata.min())
            row["img_max"] = float(idata.max())
            row["img_mean"] = round(float(idata.mean()), 4)
            row["img_std"] = round(float(idata.std()), 4)
        except Exception as e:  # noqa: BLE001 — audit must not die on one bad case
            row["error"] = f"{type(e).__name__}: {e}"
            unreadable.append(cid)
        rows.append(row)
        if len(rows) % 50 == 0:
            print(f"audited {len(rows)}/{len(image_ids)}", flush=True)

    # Challenge-test images: header/shape readability only (labels withheld by design).
    ts_ok, ts_bad = 0, []
    for p in images_ts:
        try:
            h = nib.load(str(p))
            if len(h.shape) != 3:
                ts_bad.append(p.name)
            else:
                ts_ok += 1
        except Exception:  # noqa: BLE001
            ts_bad.append(p.name)

    canonical = sorted(set(image_ids) & set(label_ids))
    id_hash = hashlib.sha256("\n".join(canonical).encode()).hexdigest()

    summary = {
        "task_root_name": task_root.name,
        "n_images_tr": len(images_tr),
        "n_labels_tr": len(labels_tr),
        "n_images_ts": len(images_ts),
        "n_ts_readable": ts_ok,
        "ts_unreadable": ts_bad,
        "duplicate_image_ids": len(image_ids) - len(set(image_ids)),
        "duplicate_label_ids": len(label_ids) - len(set(label_ids)),
        "images_without_label": sorted(set(image_ids) - set(label_ids)),
        "labels_without_image": sorted(set(label_ids) - set(image_ids)),
        "n_canonical_labeled_cases": len(canonical),
        "canonical_id_list_sha256": id_hash,
        "distinct_label_values_seen": sorted(seen_label_values),
        "distinct_orientations_seen": sorted(orientations),
        "shape_mismatches": mismatches,
        "unreadable_cases": unreadable,
        "empty_tumor_cases": sorted(r["case_id"] for r in rows if r.get("empty_tumor") is True),
        "dataset_json": dataset_json,
    }

    with open(OUT_DIR / "task07_inventory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["case_id"])
        w.writeheader()
        w.writerows(rows)
    with open(OUT_DIR / "task07_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "dataset_json"}, indent=2)[:2000], flush=True)
    print(f"canonical_cases={len(canonical)} sha256={id_hash}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
