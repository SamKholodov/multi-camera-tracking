"""CityFlow-aligned evaluation filters for MOT / MCMT metrics.

Official MCMT eval (``datasets/eval/eval.py``) keeps predictions that:
1. Fall inside the camera ROI
2. Use global ``Id`` values seen on at least two cameras

For single-camera tracks with **local** ids (``per_cam_local``), the same
multi-cam id filter is ill-defined. We instead drop predicted tracks that
never overlap a cross-camera GT identity on that camera.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

PredIdMode = Literal["global", "local"]


def infer_pred_id_mode(pred_dir: Path | str) -> PredIdMode:
    """Guess id semantics from output folder name."""
    name = str(pred_dir).replace("\\", "/")
    if "per_cam_local" in name:
        return "local"
    if "per_cam" in name:
        return "global"
    return "local"


def cross_camera_gt_ids(gt_by_cam: dict[int, np.ndarray]) -> set[int]:
    """GT track ids that appear on at least two cameras."""
    id_cams: dict[int, set[int]] = {}
    for cam, gt in gt_by_cam.items():
        if len(gt) == 0:
            continue
        for obj_id in np.unique(gt[:, 1].astype(int)):
            id_cams.setdefault(int(obj_id), set()).add(int(cam))
    return {obj_id for obj_id, cams in id_cams.items() if len(cams) >= 2}


def filter_pred_multi_cam_only(
    pred_by_cam: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Keep only predicted global ids present on >=2 cameras (CityFlow MCMT rule)."""
    id_cams: dict[int, set[int]] = {}
    for cam, pred in pred_by_cam.items():
        if len(pred) == 0:
            continue
        for obj_id in np.unique(pred[:, 1].astype(int)):
            id_cams.setdefault(int(obj_id), set()).add(int(cam))

    keep_ids = {obj_id for obj_id, cams in id_cams.items() if len(cams) >= 2}
    out: dict[int, np.ndarray] = {}
    for cam, pred in pred_by_cam.items():
        if len(pred) == 0:
            out[cam] = pred
            continue
        mask = np.isin(pred[:, 1].astype(int), list(keep_ids))
        out[cam] = pred[mask]
    return out


def _iou_matrix_tlwh(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU for axis-aligned boxes in (x, y, w, h) format."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)))

    ax1 = boxes_a[:, 0]
    ay1 = boxes_a[:, 1]
    ax2 = ax1 + boxes_a[:, 2]
    ay2 = ay1 + boxes_a[:, 3]

    bx1 = boxes_b[:, 0]
    by1 = boxes_b[:, 1]
    bx2 = bx1 + boxes_b[:, 2]
    by2 = by1 + boxes_b[:, 3]

    iou = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    for i in range(len(boxes_a)):
        xx1 = np.maximum(ax1[i], bx1)
        yy1 = np.maximum(ay1[i], by1)
        xx2 = np.minimum(ax2[i], bx2)
        yy2 = np.minimum(ay2[i], by2)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_a = (ax2[i] - ax1[i]) * (ay2[i] - ay1[i])
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        iou[i] = np.where(union > 0, inter / union, 0.0)
    return iou


def _pred_ids_matching_benchmark_gt(
    gt: np.ndarray,
    pred: np.ndarray,
    benchmark_gt_ids: set[int],
    iou_thresh: float,
) -> set[int]:
    """Predicted track ids with at least one IoU match to a benchmark GT id."""
    if len(pred) == 0 or len(gt) == 0 or not benchmark_gt_ids:
        return set()

    bench = np.array(sorted(benchmark_gt_ids), dtype=int)
    matched_pred: set[int] = set()

    gt_frames = gt[:, 0].astype(int)
    pr_frames = pred[:, 0].astype(int)
    for frame in np.intersect1d(np.unique(gt_frames), np.unique(pr_frames)):
        g_mask = gt_frames == frame
        p_mask = pr_frames == frame
        g_rows = gt[g_mask]
        p_rows = pred[p_mask]

        g_ids = g_rows[:, 1].astype(int)
        bench_mask = np.isin(g_ids, bench)
        if not bench_mask.any():
            continue

        g_boxes = g_rows[bench_mask, 2:6]
        p_ids = p_rows[:, 1].astype(int)
        p_boxes = p_rows[:, 2:6]
        ious = _iou_matrix_tlwh(g_boxes, p_boxes)
        if ious.size == 0:
            continue
        for j in range(ious.shape[1]):
            if np.any(ious[:, j] >= iou_thresh):
                matched_pred.add(int(p_ids[j]))

    return matched_pred


def filter_pred_tracks_with_benchmark_gt(
    gt: np.ndarray,
    pred: np.ndarray,
    benchmark_gt_ids: set[int],
    iou_thresh: float = 0.5,
) -> np.ndarray:
    """Drop predicted tracks that never overlap a cross-camera GT identity."""
    if len(pred) == 0:
        return pred
    keep_ids = _pred_ids_matching_benchmark_gt(gt, pred, benchmark_gt_ids, iou_thresh)
    if not keep_ids:
        return np.empty((0, pred.shape[1]))
    mask = np.isin(pred[:, 1].astype(int), list(keep_ids))
    return pred[mask]


def apply_cityflow_filters(
    gt_by_cam: dict[int, np.ndarray],
    pred_by_cam: dict[int, np.ndarray],
    mode: PredIdMode,
    iou_thresh: float = 0.5,
) -> dict[int, np.ndarray]:
    """Apply CityFlow prediction filters per camera."""
    if mode == "global":
        return filter_pred_multi_cam_only(pred_by_cam)

    benchmark_ids = cross_camera_gt_ids(gt_by_cam)
    out: dict[int, np.ndarray] = {}
    for cam, pred in pred_by_cam.items():
        gt = gt_by_cam.get(cam, np.empty((0, 10)))
        out[cam] = filter_pred_tracks_with_benchmark_gt(
            gt, pred, benchmark_ids, iou_thresh=iou_thresh
        )
    return out
