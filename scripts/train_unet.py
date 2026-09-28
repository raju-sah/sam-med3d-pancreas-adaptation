"""Phase 3 U-Net entry point. Modes: audit | smoke | train.

- audit: TRAIN-ONLY sampler audit (patch class distribution). Needs data.
- smoke: Run A gates (env, load, fwd/bwd, AMP, ckpt, sliding window, overfit).
- train: Run B full baseline (fit, best reload, final metrics, figures).
Test IDs are never loaded in any mode (enforced in src.training.data).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import data as D  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.training.trainer import (Trainer, config_hash, env_report,  # noqa: E402
                                  seed_everything, validate_volume)
from src.utils.config import load_config  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=15).stdout.strip()
    except Exception:
        return "unknown"


def mode_audit(cfg, task_root: Path, out: Path) -> dict:
    import numpy as np
    import torch

    loader, _ = D.make_train_loader(cfg, task_root, out / "cache")
    n, n_pan, n_tum, tum_fracs = 0, 0, 0, []
    it = iter(loader)
    target = 200
    while n < target:
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        lbl = b["label"]
        for i in range(lbl.shape[0]):
            l = np.asarray(lbl[i, 0].cpu())
            n += 1
            has_pan = bool(((l == 1) | (l == 2)).any())
            has_tum = bool((l == 2).any())
            n_pan += has_pan
            n_tum += has_tum
            tum_fracs.append(float((l == 2).mean()))
            if n >= target:
                break
    rep = {
        "patches_checked": n,
        "pct_with_pancreas": round(100 * n_pan / n, 2),
        "pct_with_tumor": round(100 * n_tum / n, 2),
        "pct_background_only": round(100 * (n - n_pan) / n, 2),
        "mean_tumor_voxel_frac": round(float(np.mean(tum_fracs)), 6),
    }
    print(json.dumps(rep, indent=1))
    with open(out / "sampler_audit.json", "w") as f:
        json.dump(rep, f, indent=1)
    if rep["pct_with_tumor"] < 5:
        raise RuntimeError("sampler audit FAILED: tumor patches effectively zero")
    return rep


def mode_smoke(cfg, task_root: Path, out: Path) -> dict:
    import numpy as np
    import torch

    from src.models.unet import build_unet, count_parameters

    gates = {}
    gates["cuda"] = torch.cuda.is_available()
    from src.training.trainer import env_report

    gates["env"] = env_report()
    import monai

    gates["monai"] = monai.__version__
    loader, _ = D.make_train_loader(cfg, task_root, out / "cache")
    batch = next(iter(loader))  # real train case, preprocessing+sampler OK
    gates["batch_shapes"] = [list(batch["image"].shape), list(batch["label"].shape)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(cfg).to(device)
    gates["params"] = count_parameters(model)
    from src.training.trainer import build_loss, build_optimizer

    loss_fn = build_loss(cfg)
    opt = build_optimizer(cfg, model)
    img = batch["image"].to(device)
    lbl = batch["label"].to(device)
    use_amp = bool(cfg["amp"]) and torch.cuda.is_available()
    from torch.amp import GradScaler, autocast

    scaler = GradScaler("cuda", enabled=use_amp)
    opt.zero_grad(set_to_none=True)
    with autocast("cuda", enabled=use_amp):
        loss = loss_fn(model(img), lbl)
    assert torch.isfinite(loss), "non-finite smoke loss"
    gates["loss0"] = float(loss.item())
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    gates["backward_amp_ok"] = True
    # checkpoint write/read
    tr = Trainer(cfg, model, loader, [], out)
    tr.save("smoke.pt", 0)
    tr.load(out / "smoke.pt")
    gates["ckpt_roundtrip_ok"] = True
    # sliding-window on one val case
    val_items = D.make_val_items(cfg, task_root)
    assert len(val_items) == 42, f"expected 42 val cases, got {len(val_items)}"
    from src.data import preprocess as P

    import monai.transforms as T

    det = T.Compose([P.build_monai_inference_transforms(cfg),
                     P._T(T, "ToTensord", "ToTensorD")(keys=["image", "label"])])
    d = det({"image": val_items[0]["image"], "label": val_items[0]["label"]})
    rep = validate_volume(model, d["image"], d["label"], cfg, device)
    gates["sliding_window_macro"] = round(rep["dice_macro_foreground"], 4)
    # single-batch overfit: fixed batch, bounded iters. Gate requires
    # substantial movement (loss < 0.7x start OR macro gain > 0.2) — the point
    # is detecting broken labels/loss/outputs, not full memorization.
    fixed_img, fixed_lbl = img.detach(), lbl.detach()
    l0 = gates["loss0"]
    for _ in range(120):
        opt.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=use_amp):
            l = loss_fn(model(fixed_img), fixed_lbl)
        scaler.scale(l).backward()
        scaler.step(opt)
        scaler.update()
    with torch.no_grad(), autocast("cuda", enabled=use_amp):
        l1 = float(loss_fn(model(fixed_img), fixed_lbl).item())
        pred = M.softmax_argmax(model(fixed_img)[0].cpu())
    d0 = M.multiclass_report(
        np.zeros_like(np.asarray(fixed_lbl[0, 0].cpu())), np.asarray(fixed_lbl[0, 0].cpu()))
    d1 = M.multiclass_report(pred, np.asarray(fixed_lbl[0, 0].cpu()))
    gates["overfit"] = {"iters": 120, "loss_start": round(l0, 4), "loss_end": round(l1, 4),
                        "macro_start": round(d0["dice_macro_foreground"], 4),
                        "macro_end": round(d1["dice_macro_foreground"], 4)}
    ok = (l1 < 0.7 * l0) or (d1["dice_macro_foreground"] > d0["dice_macro_foreground"] + 0.2)
    gates["overfit_pass"] = bool(ok)
    print(json.dumps(gates, indent=1, default=str))
    with open(out / "smoke_gates.json", "w") as f:
        json.dump(gates, f, indent=1, default=str)
    if not ok:
        raise RuntimeError("overfit gate FAILED: model cannot memorize a fixed batch")
    return gates


def mode_train(cfg, task_root: Path, out: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from monai.inferers import sliding_window_inference

    from src.models.unet import build_unet, count_parameters

    t_all = time.time()
    cache_dir = Path(cfg["cache_dir"]) if cfg.get("cache_dir") else out / "cache"
    loader, _ = D.make_train_loader(cfg, task_root, cache_dir)
    val_items = D.make_val_items(cfg, task_root)
    assert len(val_items) == 42
    model = build_unet(cfg)
    params = count_parameters(model)
    tr = Trainer(cfg, model, loader, val_items, out)
    fit = tr.fit()
    # reload best, final metrics incl. HD95 (single inference per case)
    tr.load(out / cfg["best_filename"])
    device = tr.device
    from src.data import preprocess as P

    import monai.transforms as T

    det = T.Compose([P.build_monai_inference_transforms(cfg),
                     P._T(T, "ToTensord", "ToTensorD")(keys=["image", "label"])])
    model.eval()
    per_case = []
    with torch.no_grad():
        for item in val_items:
            d = det({"image": item["image"], "label": item["label"]})
            logits = sliding_window_inference(
                d["image"].unsqueeze(0).to(device), tuple(cfg["sw_roi_size"]),
                cfg["sw_batch_size"], model, cfg["sw_overlap"], mode=cfg["sw_mode"])[0].cpu()
            pred = M.softmax_argmax(logits)
            gt = np.asarray(d["label"][0].cpu())
            rep = M.multiclass_report(pred, gt)
            rep.update(M.hd95_per_target(pred, gt,
                                         voxel_size_mm=float(cfg["target_spacing"][0])))
            rep["case"] = Path(item["image"]).stem
            rep["pred"] = pred  # kept in memory only for figures below
            per_case.append(rep)
    keys = ["dice_class1", "dice_tumor", "dice_macro_foreground", "dice_whole",
            "iou_class1", "iou_tumor", "iou_whole", "hd95_whole", "hd95_tumor"]
    final = {k: float(np.nanmean([r[k] for r in per_case])) for k in keys}
    summary = {
        "experiment_id": cfg["experiment_id"],
        "git_commit": git_commit(),
        "config_hash": config_hash(cfg),
        "split_manifest": cfg["split_manifest"],
        "train_cases": 197,
        "val_cases": 42,
        "test_cases_sealed": 42,
        "best_epoch": fit["best_epoch"],
        "best_val_macro_foreground_dice": round(fit["best_macro"], 6),
        "final_validation": {k: round(v, 6) for k, v in final.items()},
        "params": params,
        "total_training_seconds": round(time.time() - t_all, 1),
        "peak_gpu_memory_mb": max([h.get("peak_gpu_memory_mb", 0) for h in tr.history] + [0]),
        "env": env_report(),
        "config": cfg,
    }
    with open(out / "unet_baseline_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tr.history[0].keys()))
        w.writeheader()
        w.writerows(tr.history)
    with open(out / "unet_baseline_summary.json", "w") as f:
        json.dump({k: v for k, v in summary.items()}, f, indent=1)
    with open(out / "unet_baseline_percase.json", "w") as f:
        json.dump([{k: (v if not isinstance(v, np.ndarray) else None) for k, v in r.items()
                    if k != "pred"} for r in per_case], f, indent=1)
    # qualitative figures: worst/median/best tumor-Dice VALIDATION cases (post-freeze)
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    order = sorted(per_case, key=lambda r: r["dice_tumor"])
    picks = [("worst", order[0]), ("median", order[len(order) // 2]), ("best", order[-1])]
    for tag, r in picks:
        it = next(i for i in val_items if Path(i["image"]).stem == r["case"])
        d = det({"image": it["image"], "label": it["label"]})
        img = np.asarray(d["image"][0].cpu())
        gt = np.asarray(d["label"][0].cpu())
        pred = r["pred"]
        z = int(np.argwhere(gt == 2).mean(axis=0)[2]) if (gt == 2).any() else img.shape[2] // 2
        disp = np.clip(img[:, :, z], -150, 250)
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(disp, cmap="gray")
        ax[0].set_title("CT (VALIDATION)")
        ax[1].imshow(gt[:, :, z], cmap="jet", vmin=0, vmax=2)
        ax[1].set_title("GT")
        ax[2].imshow(pred[:, :, z], cmap="jet", vmin=0, vmax=2)
        ax[2].set_title("pred")
        err = np.zeros_like(gt[:, :, z])
        err[(pred[:, :, z] > 0) != (gt[:, :, z] > 0)] = 1
        ax[3].imshow(disp, cmap="gray")
        ax[3].imshow(np.ma.masked_where(err == 0, err), alpha=0.6, cmap="Reds")
        ax[3].set_title(f"error ({tag} tumor dice={r['dice_tumor']:.3f})")
        for a in ax:
            a.axis("off")
        fig.suptitle(f"{r['case']} VALIDATION")
        fig.tight_layout()
        fig.savefig(figdir / f"unet_val_{tag}_{r['case']}.png", dpi=100)
        plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=1))
    print("LEDGER_ROW:" + json.dumps({
        "experiment_id": cfg["experiment_id"], "model": "monai_unet",
        "dataset": cfg["dataset"], "train_fraction": cfg["train_fraction"],
        "seed": cfg["seed"], "git_commit": summary["git_commit"],
        "config_hash": summary["config_hash"], "accelerator": summary["env"].get("gpu_name"),
        "best_epoch": fit["best_epoch"],
        "val_macro_dice": summary["best_val_macro_foreground_dice"],
        "val_tumor_dice": final["dice_tumor"], "val_whole_pancreas_dice": final["dice_whole"],
        "training_time_s": summary["total_training_seconds"], "status": "complete",
        "notes": "test sealed; val-only selection"}))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/unet_baseline.yaml")
    ap.add_argument("--mode", choices=["audit", "smoke", "train"], required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    task_root = D.find_task_root(args.data_root)
    print(f"task_root: {task_root}")
    out = Path(args.output_dir or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "audit":
        mode_audit(cfg, task_root, out)
    elif args.mode == "smoke":
        mode_smoke(cfg, task_root, out)
    else:
        mode_train(cfg, task_root, out)


if __name__ == "__main__":
    main()
