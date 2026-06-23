"""Helpers for CityFlow S02 eval with synchronized videos (sync_manifest.json)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_sync_manifest(gt_root: Path) -> dict | None:
    path = gt_root / "sync_manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def sync_skip_by_cam(manifest: dict | None) -> dict[int, int]:
    if not manifest:
        return {}
    out: dict[int, int] = {}
    for entry in manifest.get("cameras", []):
        out[int(entry["cam_id"])] = int(entry.get("skip_frames", 0))
    return out


def sync_length_frames(manifest: dict | None) -> int | None:
    if not manifest:
        return None
    val = manifest.get("sync_length_frames")
    return int(val) if val is not None else None


def align_gt_to_sync(
    gt: np.ndarray,
    skip_frames: int,
    sync_length: int | None = None,
) -> np.ndarray:
    """Map raw GT timeline to synchronized video frame indices (1-based)."""
    if len(gt) == 0:
        return gt
    skip = int(skip_frames)
    frames = gt[:, 0].astype(int)
    mask = frames > skip
    if sync_length is not None:
        mask &= frames <= skip + int(sync_length)
    out = gt[mask].copy()
    if len(out):
        out[:, 0] = out[:, 0] - skip
    return out


def cap_pred_to_sync_length(pred: np.ndarray, sync_length: int | None) -> np.ndarray:
    if len(pred) == 0 or sync_length is None:
        return pred
    cap = int(sync_length)
    return pred[pred[:, 0].astype(int) <= cap]


def s02_complete_frame_threshold(gt_root: Path) -> int:
    """Frames expected in predictions when using vdo_synch.avi."""
    manifest = load_sync_manifest(gt_root)
    sync_len = sync_length_frames(manifest)
    if sync_len is not None:
        return sync_len
    from scripts.eval_s02 import s02_gt_max_frame

    return s02_gt_max_frame(gt_root)


def apply_sync_alignment(
    gt_by_cam: dict[int, np.ndarray],
    pr_by_cam: dict[int, np.ndarray],
    gt_root: Path,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict | None]:
    manifest = load_sync_manifest(gt_root)
    if manifest is None:
        return gt_by_cam, pr_by_cam, None
    skips = sync_skip_by_cam(manifest)
    sync_len = sync_length_frames(manifest)
    gt_out: dict[int, np.ndarray] = {}
    pr_out: dict[int, np.ndarray] = {}
    for cam, gt in gt_by_cam.items():
        gt_out[cam] = align_gt_to_sync(gt, skips.get(cam, 0), sync_len)
        pr_out[cam] = cap_pred_to_sync_length(pr_by_cam.get(cam, np.empty((0, 10))), sync_len)
    return gt_out, pr_out, manifest
