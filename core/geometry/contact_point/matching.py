"""Detection matching utilities for contact point preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DetectionMatch:
    bbox: tuple[float, float, float, float] | None
    score: float
    class_id: int
    iou: float


def iou_xyxy(
    a: tuple[float, float, float, float] | list[float],
    b: tuple[float, float, float, float] | list[float],
) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def best_detection_match(
    ann_bbox: tuple[float, float, float, float] | list[float],
    detections: Iterable[list[float] | tuple[float, ...]],
    *,
    used_indices: set[int] | None = None,
) -> tuple[int, DetectionMatch]:
    best_idx = -1
    best = DetectionMatch(bbox=None, score=0.0, class_id=-1, iou=0.0)
    used = used_indices or set()
    for idx, det in enumerate(detections):
        if idx in used:
            continue
        if len(det) < 6:
            continue
        bbox = tuple(float(v) for v in det[:4])
        score = float(det[4])
        class_id = int(det[5])
        value = iou_xyxy(ann_bbox, bbox)
        if value > best.iou:
            best_idx = idx
            best = DetectionMatch(bbox=bbox, score=score, class_id=class_id, iou=value)
    return best_idx, best
