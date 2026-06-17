"""Metrics for contact point regression."""

from __future__ import annotations

import torch


def contact_metrics(pred_uv: torch.Tensor, target_uv: torch.Tensor, bbox_wh: torch.Tensor) -> dict[str, float]:
    diff = pred_uv - target_uv
    abs_diff = diff.abs()
    pixel = torch.sqrt(((diff * bbox_wh) ** 2).sum(dim=1))
    return {
        "mae_u": float(abs_diff[:, 0].mean().item()),
        "mae_v": float(abs_diff[:, 1].mean().item()),
        "pixel_mae": float(pixel.mean().item()),
    }


def baseline_uv_like(target_uv: torch.Tensor) -> torch.Tensor:
    baseline = torch.zeros_like(target_uv)
    baseline[:, 0] = 0.5
    baseline[:, 1] = 1.0
    return baseline
