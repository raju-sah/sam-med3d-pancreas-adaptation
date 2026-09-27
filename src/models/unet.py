"""MONAI 3D U-Net baseline factory (Phase 3). monai imported lazily."""

from __future__ import annotations

from typing import Any


def build_unet(cfg: dict[str, Any]):
    """Construct monai.networks.nets.UNet from config (frozen arch)."""
    from monai.networks.nets import UNet

    return UNet(
        spatial_dims=cfg["spatial_dims"],
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        channels=tuple(cfg["channels"]),
        strides=tuple(cfg["strides"]),
        num_res_units=cfg["num_res_units"],
        norm=cfg["norm"],
    )


def count_parameters(model) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
