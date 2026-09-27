"""Phase 2 preprocessing: common (Layer A) + model-specific (Layer B) helpers.

Layer A (model-independent): NIfTI loading, channel-first convention, 3D and
alignment checks, canonical orientation metadata, discrete-label validation.
Layer B (model-specific, e.g. MONAI U-Net pipeline vs SAM-Med3D 1.5mm/128^3
ROI workflow) is configured via configs/preprocess_phase2.yaml; the MONAI
transform builder lazy-imports monai so this module imports without it.

Rules enforced here:
- labels are always nearest-neighbor resampled; {0,1,2} asserted afterwards
- derived binary masks are in-memory only; original GT never overwritten
- no test IDs may enter profiling (see assert_not_test / split helpers)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

VALID_LABELS = (0, 1, 2)
LABEL_NAMES = {0: "background", 1: "pancreas", 2: "mass_tumor"}


def validate_labels(mask: np.ndarray, valid: tuple[int, ...] = VALID_LABELS) -> set[int]:
    """Return present labels; raise ValueError if any label is invalid."""
    present = set(int(v) for v in np.unique(mask))
    bad = present - set(valid)
    if bad:
        raise ValueError(f"invalid label values {sorted(bad)}; expected subset of {list(valid)}")
    return present


def derive_masks(mask: np.ndarray) -> dict[str, np.ndarray]:
    """In-memory derived targets. Never writes to disk.

    whole: pancreas region (label 1 OR 2); tumor: label 2 only.
    """
    validate_labels(mask)
    m = np.asarray(mask)
    return {
        "multiclass": m.astype(np.uint8),
        "whole": ((m == 1) | (m == 2)).astype(np.uint8),
        "tumor": (m == 2).astype(np.uint8),
    }


def check_alignment(image: np.ndarray, label: np.ndarray) -> tuple[int, ...]:
    """Assert image/label share spatial shape; return it."""
    if image.shape[-3:] != label.shape[-3:]:
        raise ValueError(f"image/label shape mismatch: {image.shape} vs {label.shape}")
    if len(image.shape) != 4 or image.shape[0] < 1:
        raise ValueError(f"image must be channel-first 4D, got {image.shape}")
    if len(label.shape) != 3:
        raise ValueError(f"label must be 3D, got {label.shape}")
    return tuple(label.shape)


def ensure_channel_first(volume: np.ndarray) -> np.ndarray:
    """(D,H,W) -> (1,D,H,W); pass through already channel-first arrays."""
    v = np.asarray(volume)
    if v.ndim == 3:
        return v[np.newaxis]
    if v.ndim == 4:
        return v
    raise ValueError(f"expected 3D or channel-first 4D volume, got shape {v.shape}")


def resample_label_nearest(label: np.ndarray, zoom_factors: tuple[float, ...]) -> np.ndarray:
    """Nearest-neighbor label resampling (scipy); asserts discrete labels kept.

    Production path uses MONAI Spacingd(mode='nearest'); this helper documents
    and tests the invariant on synthetic data without requiring MONAI.
    """
    from scipy.ndimage import zoom

    out = zoom(np.asarray(label), zoom_factors, order=0)
    validate_labels(out)
    return out.astype(np.uint8)


def load_nifti_image_label(image_path: str | Path, label_path: str | Path) -> dict[str, Any]:
    """Load an image/label pair (nibabel). Returns arrays + spatial metadata."""
    import nibabel as nib

    img = nib.load(str(image_path))
    lbl = nib.load(str(label_path))
    image = ensure_channel_first(np.asarray(img.dataobj))
    label = np.asarray(lbl.dataobj)
    check_alignment(image, label)
    validate_labels(label)
    return {
        "image": image,
        "label": label.astype(np.uint8),
        "image_affine": np.asarray(img.affine),
        "label_affine": np.asarray(lbl.affine),
        "image_zooms": tuple(float(z) for z in img.header.get_zooms()[:3]),
    }


def repo_split_ids(split: str, splits_dir: str | Path | None = None) -> list[str]:
    """Load tracked split IDs (train/val/test) — IDs only, never volumes."""
    if splits_dir is None:
        from src.utils.paths import repo_root

        splits_dir = repo_root() / "data" / "splits"
    with open(Path(splits_dir) / f"{split}.json", encoding="utf-8") as f:
        return json.load(f)


def assert_not_test(case_ids: list[str], splits_dir: str | Path | None = None) -> None:
    """Guard: profiling/visualization selections must exclude frozen test IDs."""
    test_ids = set(repo_split_ids("test", splits_dir))
    leaked = test_ids & set(case_ids)
    if leaked:
        raise ValueError(f"frozen test IDs must not enter Phase 2 profiling: {sorted(leaked)}")


def _T(mod, *names):
    """First existing transform name (MONAI renamed *d aliases to *D)."""
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    raise AttributeError(f"none of {names} in monai.transforms")


def build_monai_inference_transforms(cfg: dict[str, Any]):
    """Build the deterministic MONAI Compose from the frozen config.

    Lazy-imports monai (available on Kaggle, absent locally). Keys used:
    orientation, target_spacing (or null for native), clamp_min/max,
    norm ('fg_zscore'), image_mode, label_mode.
    """
    import monai.transforms as T

    LoadImaged = _T(T, "LoadImaged", "LoadImageD")
    ChannelFirstd = _T(T, "EnsureChannelFirstd", "EnsureChannelFirstD")
    Orientationd = _T(T, "Orientationd", "OrientationD")
    Spacingd = _T(T, "Spacingd", "SpacingD")
    ScaleRanged = _T(T, "ScaleIntensityRangeD", "ScaleIntensityRanged")
    Normd = _T(T, "NormalizeIntensityd", "NormalizeIntensityD")

    pre: list = [
        LoadImaged(keys=["image", "label"]),
        ChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes=cfg["orientation"]),
    ]
    if cfg.get("target_spacing"):
        pre.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(cfg["target_spacing"]),
                mode=(cfg.get("image_mode", "bilinear"), cfg.get("label_mode", "nearest")),
            )
        )
    pre += [
        # ScaleIntensityRange with clip=True and equal in/out bounds == clamp.
        ScaleRanged(
            keys=["image"],
            a_min=cfg["clamp_min"],
            a_max=cfg["clamp_max"],
            b_min=cfg["clamp_min"],
            b_max=cfg["clamp_max"],
            clip=True,
        ),
        Normd(keys=["image"], nonzero=True, channel_wise=True),
    ]
    return T.Compose(pre)
