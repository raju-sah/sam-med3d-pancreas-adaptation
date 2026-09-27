"""Phase 3 training harness (MONAI 3D U-Net). Config-driven, resumable.

- Checkpoint selection: macro foreground val Dice ONLY (frozen rule).
- Early stopping: patience counted in validation EVENTS, min_delta gate.
- Validation: full volumes, sliding window, deterministic, no GT cropping.
- Checkpoints carry epoch/model/optimizer/scheduler/scaler/best/config/rng.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def snapshot_rng() -> dict:
    """RNG states normalized to CPU for storage (map_location-proof)."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().cpu(),
        "cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else None,
    }


def restore_rng(r: dict) -> bool:
    """Best-effort RNG restore. Returns False (warns, never crashes) on mismatch,
    e.g. torch-version RNG layout differences across machines."""
    try:
        if "python" not in r:
            return True
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        st = r["torch"]
        if torch.is_tensor(st):
            st = st.cpu().to(torch.uint8).contiguous()
            if st.numel() == torch.get_rng_state().numel():
                torch.set_rng_state(st)
            else:
                print(f"WARNING: CPU RNG size mismatch "
                      f"({st.numel()} vs {torch.get_rng_state().numel()}); skipped")
                return False
        if r.get("cuda") is not None and torch.cuda.is_available():
            states = r["cuda"]
            if torch.is_tensor(states):
                states = [states]
            dev = torch.cuda.current_device()
            fixed = [s.to(device=f"cuda:{dev}", dtype=torch.uint8).contiguous()
                     for s in states if torch.is_tensor(s)]
            if fixed:
                torch.cuda.set_rng_state_all(fixed)
        return True
    except Exception as e:  # noqa: BLE001 — RNG restore must never kill training
        print(f"WARNING: RNG restore skipped ({type(e).__name__}: {e})")
        return False


def seed_everything(seed: int) -> None:
    import monai.utils as U

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    U.set_determinism(seed=seed)


def build_loss(cfg: dict[str, Any]):
    from monai.losses import DiceCELoss

    return DiceCELoss(
        include_background=cfg["loss_include_background"],
        to_onehot_y=cfg["loss_to_onehot_y"],
        softmax=cfg["loss_softmax"],
    )


def build_optimizer(cfg: dict[str, Any], model: nn.Module):
    return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def build_scheduler(cfg: dict[str, Any], optimizer) -> Any:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"])


@torch.no_grad()
def validate_volume(model, image: torch.Tensor, cfg: dict, device) -> dict[str, float]:
    """Sliding-window inference on one full volume -> metric report."""
    from monai.inferers import sliding_window_inference

    from src.training.metrics import multiclass_report, softmax_argmax

    model.eval()
    logits = sliding_window_inference(
        image.unsqueeze(0).to(device),
        roi_size=tuple(cfg["sw_roi_size"]),
        sw_batch_size=cfg["sw_batch_size"],
        predictor=model,
        overlap=cfg["sw_overlap"],
        mode=cfg["sw_mode"],
    )
    return multiclass_report(softmax_argmax(logits[0].cpu()))


def env_report() -> dict[str, Any]:
    import monai

    rep = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "monai": monai.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        rep.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
            "vram_mb": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
        })
    return rep


class Trainer:
    def __init__(self, cfg, model, train_loader, val_items, out_dir: Path):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_items = val_items
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "cache").mkdir(exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_fn = build_loss(cfg)
        self.opt = build_optimizer(cfg, model)
        self.sched = build_scheduler(cfg, self.opt)
        self.use_amp = bool(cfg["amp"]) and torch.cuda.is_available()
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.start_epoch = 0
        self.best_macro = float("-inf")
        self.best_epoch = -1
        self.bad_events = 0
        self.history: list[dict] = []
        self.t0 = time.time()

    # ----- checkpointing -----
    def _state(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_macro": self.best_macro,
            "best_epoch": self.best_epoch,
            "bad_events": self.bad_events,
            "config": self.cfg,
            "config_hash": config_hash(self.cfg),
            "seed": self.cfg["seed"],
            "rng": snapshot_rng(),
        }

    def save(self, name: str, epoch: int) -> Path:
        p = self.out / name
        torch.save(self._state(epoch), p)
        return p

    def load(self, path: str | Path) -> int:
        s = torch.load(str(path), map_location=self.device, weights_only=False)
        assert s["config_hash"] == config_hash(self.cfg), "config changed since checkpoint"
        self.model.load_state_dict(s["model"])
        self.opt.load_state_dict(s["optimizer"])
        self.sched.load_state_dict(s["scheduler"])
        self.scaler.load_state_dict(s["scaler"])
        self.best_macro, self.best_epoch, self.bad_events = (
            s["best_macro"], s["best_epoch"], s["bad_events"])
        r = s.get("rng") or {}
        restore_rng(r)
        self.start_epoch = s["epoch"] + 1
        return self.start_epoch

    # ----- training -----
    def train_one_epoch(self, epoch: int) -> float:
        from src.data.preprocess import validate_labels

        self.model.train()
        total, n = 0.0, 0
        for batch in self.train_loader:
            img = batch["image"].to(self.device)
            lbl = batch["label"].to(self.device)
            validate_labels(np.asarray(lbl.cpu()))
            self.opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=self.use_amp):
                loss = self.loss_fn(self.model(img), lbl)
            assert torch.isfinite(loss), "non-finite training loss"
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            total += float(loss.item())
            n += 1
        return total / max(n, 1)

    def run_validation(self) -> dict[str, float]:
        from src.data import preprocess as P

        import monai.transforms as T

        to_t = P._T(T, "ToTensord", "ToTensorD")
        det = T.Compose([P.build_monai_inference_transforms(self.cfg),
                         to_t(keys=["image", "label"])])
        agg: dict[str, list[float]] = {}
        for item in self.val_items:
            d = det({"image": item["image"], "label": item["label"]})
            rep = validate_volume(self.model, d["image"], self.cfg, self.device)
            for k, v in rep.items():
                agg.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in agg.items()}

    def fit(self, log=print) -> dict[str, Any]:
        from src.training.metrics import is_improvement

        cfg = self.cfg
        for epoch in range(self.start_epoch, cfg["max_epochs"]):
            t_ep = time.time()
            train_loss = self.train_one_epoch(epoch)
            self.sched.step()
            row: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "learning_rate": round(self.opt.param_groups[0]["lr"], 8),
                "epoch_time_seconds": round(time.time() - t_ep, 1),
                "peak_gpu_memory_mb": (
                    torch.cuda.max_memory_allocated() // (1024 * 1024)
                    if torch.cuda.is_available() else 0),
            }
            if (epoch + 1) % cfg["val_interval_epochs"] == 0 or epoch == cfg["max_epochs"] - 1:
                val = self.run_validation()
                row.update({f"val_{k}": round(v, 6) for k, v in val.items()})
                macro = val["dice_macro_foreground"]
                if is_improvement(macro, self.best_macro, cfg["early_stopping_min_delta"]):
                    self.best_macro, self.best_epoch, self.bad_events = macro, epoch, 0
                    self.save(cfg["best_filename"], epoch)
                else:
                    self.bad_events += 1
                log(f"epoch {epoch}: loss={train_loss:.4f} "
                    f"val_macro={macro:.4f} best={self.best_macro:.4f}@{self.best_epoch}")
                if self.bad_events >= cfg["early_stopping_patience_val_events"]:
                    log(f"early stop at epoch {epoch} ({self.bad_events} bad val events)")
                    row["early_stopped"] = True
                    self.history.append(row)
                    break
            self.history.append(row)
            self.save(cfg["last_filename"], epoch)
        self.save(cfg["last_filename"], self.history[-1]["epoch"])
        return {"best_epoch": self.best_epoch, "best_macro": self.best_macro,
                "epochs_run": len(self.history)}
