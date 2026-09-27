"""Segmentation metrics (Phase 3). Pure numpy/torch, no dataset needed.

Conventions (frozen):
- multiclass softmax prediction -> argmax over channel 0..2
- class 1 = pancreas/parenchyma, class 2 = mass/tumor; background excluded
  from all headline metrics
- macro foreground Dice = (dice_c1 + dice_c2) / 2  (checkpoint selector)
- whole region: (pred in {1,2}) vs (gt in {1,2})
- empty GT + empty pred -> Dice 1.0; empty GT + nonempty pred -> 0.0
  (documented client choice; all val cases have tumor GT per Phase 1,
  but code stays robust)
- HD95 via MONAI HausdorffDistanceMetric(percentile=95), lazy import;
  undefined (empty pred or empty GT) -> NaN + counted
"""

from __future__ import annotations

import numpy as np


def softmax_argmax(logits) -> np.ndarray:
    """(C,D,H,W) logits -> (D,H,W) uint8 class map."""
    import torch

    t = logits.detach() if torch.is_tensor(logits) else torch.as_tensor(logits)
    return torch.argmax(torch.softmax(t.float(), dim=0), dim=0).to(torch.uint8).cpu().numpy()


def binary_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    inter = float((p & g).sum())
    denom = float(p.sum() + g.sum())
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    inter = float((p & g).sum())
    union = float((p | g).sum())
    if union == 0:
        return 1.0
    return inter / union


def multiclass_report(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Dice/IoU per class + macro fg + whole-region. pred/gt: (D,H,W) uint8."""
    from src.data.preprocess import validate_labels

    validate_labels(pred)
    validate_labels(gt)
    d1 = binary_dice(pred == 1, gt == 1)
    d2 = binary_dice(pred == 2, gt == 2)
    whole_pred = (pred == 1) | (pred == 2)
    whole_gt = (gt == 1) | (gt == 2)
    dw = binary_dice(whole_pred, whole_gt)
    return {
        "dice_class1": d1,
        "dice_tumor": d2,
        "dice_macro_foreground": (d1 + d2) / 2.0,
        "dice_whole": dw,
        "iou_class1": binary_iou(pred == 1, gt == 1),
        "iou_tumor": binary_iou(pred == 2, gt == 2),
        "iou_whole": binary_iou(whole_pred, whole_gt),
    }


def hd95_per_target(pred: np.ndarray, gt: np.ndarray, voxel_size_mm: float = 1.5) -> dict[str, float]:
    """HD95 in mm for whole + tumor, evaluated on the 1.5mm-iso resampled grid.

    MONAI 1.6 HausdorffDistanceMetric has no spacing argument, so voxel
    distances are scaled by the isotropic voxel size (frozen 1.5 mm).
    Returns NaN for empty pred/GT with 'n_undefined' count. Lazy MONAI import.
    """
    import torch
    from monai.metrics import HausdorffDistanceMetric

    out: dict[str, float] = {}
    undefined = 0
    for name, p, g in (
        ("hd95_whole", (pred == 1) | (pred == 2), (gt == 1) | (gt == 2)),
        ("hd95_tumor", pred == 2, gt == 2),
    ):
        if not p.any() or not g.any():
            out[name] = float("nan")
            undefined += 1
            continue
        m = HausdorffDistanceMetric(include_background=True, percentile=95)
        with torch.no_grad():
            v = m(torch.as_tensor(p[None, None].astype(np.float32)),
                  torch.as_tensor(g[None, None].astype(np.float32)))
        out[name] = float(v.item()) * voxel_size_mm
    out["hd95_n_undefined"] = float(undefined)
    return out


def is_improvement(new: float, best: float, min_delta: float = 0.001) -> bool:
    """Checkpoint-selection rule: strictly better by more than min_delta."""
    return bool(new > best + min_delta)
