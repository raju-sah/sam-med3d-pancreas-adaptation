"""Phase 3 datasets: frozen deterministic prefix + training-only randomness.

- Deterministic prefix reuses src.data.preprocess (frozen Phase 2 policy).
- Prefix cached with PersistentDataset (disk; RAM-safe for 197 volumes).
- Random stage (label-aware crop + aug) applied fresh per iteration via
  RandomizedView wrapper — caching never freezes randomness.
- Split IDs from data/splits/*.json; test IDs read ONLY for exclusion
  asserts. Test volumes are never listed, loaded, or touched.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def find_task_root(hint: str | None = None) -> Path:
    """Locate Dataset007_Pancreas/ (Kaggle mount or local checkout)."""
    candidates = []
    if hint:
        candidates.append(Path(hint))
    candidates += [
        Path("/kaggle/input/datasets/hansenc/pancreas-task07/Dataset007_Pancreas"),
        Path("/kaggle/input"),
        Path("data"),
        Path("."),
    ]
    for base in candidates:
        if not base.exists():
            continue
        if (base / "dataset.json").is_file():
            return base
        for dirpath, dirnames, filenames in os.walk(base):
            if len(Path(dirpath).relative_to(base).parts) > 4:
                dirnames[:] = []
                continue
            if "dataset.json" in filenames:
                return Path(dirpath)
    raise FileNotFoundError("Task07 root (dataset.json) not found")


def case_paths(task_root: Path, case_id: str) -> dict[str, str]:
    for ext in (".nii", ".nii.gz"):
        ip, lp = task_root / "imagesTr" / f"{case_id}{ext}", task_root / "labelsTr" / f"{case_id}{ext}"
        if ip.is_file() and lp.is_file():
            return {"image": str(ip), "label": str(lp)}
    raise FileNotFoundError(f"pair missing for {case_id}")


def load_split_items(split: str, task_root: Path, splits_dir: str | Path | None = None) -> list[dict]:
    """Load file dicts for train/val. Refuses test; asserts test exclusion."""
    from src.data.preprocess import repo_split_ids

    if split == "test":
        raise ValueError("internal test is SEALED in Phase 3 (exclusion checks only)")
    ids = repo_split_ids(split, splits_dir)
    test_ids = set(repo_split_ids("test", splits_dir))
    leaked = test_ids & set(ids)
    if leaked:
        raise ValueError(f"test IDs leaked into {split}: {sorted(leaked)}")
    other = "val" if split == "train" else "train"
    if set(ids) & set(repo_split_ids(other, splits_dir)):
        raise ValueError(f"train/val overlap detected")
    return [case_paths(task_root, cid) for cid in ids]


def deterministic_prefix(cfg: dict[str, Any]):
    """Frozen Phase 2 deterministic transforms + tensor conversion for caching."""
    from src.data import preprocess as P

    import monai.transforms as T

    ToTensor = P._T(T, "ToTensord", "ToTensorD")
    base = P.build_monai_inference_transforms(cfg)
    return T.Compose([base, ToTensor(keys=["image", "label"])])


def training_random_stage(cfg: dict[str, Any]):
    """Label-aware crop + conventional augmentation (TRAIN ONLY)."""
    import monai.transforms as T

    Crop = P_RandCrop(cfg)
    return T.Compose([
        Crop,
        T.RandFlipd(keys=["image", "label"], prob=cfg["aug_flip_prob"], spatial_axis=(0, 1, 2)),
        T.RandRotate90d(keys=["image", "label"], prob=cfg["aug_rotate90_prob"], max_k=3,
                        spatial_axes=(0, 1)),
        T.RandScaleIntensityd(keys=["image"], factors=tuple(cfg["aug_scale_factors"]),
                              prob=cfg["aug_scale_prob"]),
        T.RandShiftIntensityd(keys=["image"], offsets=tuple(cfg["aug_shift_offsets"]),
                              prob=cfg["aug_shift_prob"]),
    ])


def P_RandCrop(cfg):
    import monai.transforms as T

    mod = T
    name = "RandCropByLabelClassesd" if hasattr(mod, "RandCropByLabelClassesd") else "RandCropByLabelClassesD"
    return getattr(mod, name)(
        keys=["image", "label"],
        label_key="label",
        spatial_size=tuple(cfg["patch_size"]),
        ratios=list(cfg["sampler_ratios"]),
        num_classes=cfg["sampler_num_classes"],
        num_samples=cfg["sampler_num_samples"],
    )


class RandomizedView:
    """Wraps a cached deterministic dataset; applies random stage per call.

    RandCropByLabelClassesd returns a list (num_samples); num_samples=1
    is configured, so element [0] is returned as the training item.
    """

    def __init__(self, cached, rand_transforms):
        self.cached = cached
        self.rand = rand_transforms

    def __len__(self):
        return len(self.cached)

    def __getitem__(self, idx):
        out = self.rand(dict(self.cached[idx]))
        return out[0] if isinstance(out, list) else out


def make_train_loader(cfg: dict[str, Any], task_root: Path, cache_dir: Path):
    from monai.data import DataLoader, PersistentDataset

    det = PersistentDataset(
        data=load_split_items("train", task_root),
        transform=deterministic_prefix(cfg),
        cache_dir=str(cache_dir),
    )
    train_ds = RandomizedView(det, training_random_stage(cfg))
    return DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    ), train_ds


def make_val_items(cfg: dict[str, Any], task_root: Path) -> list[dict]:
    return load_split_items("val", task_root)
