"""Debug visualization for contact point dataset preparation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


COLOR_ANN = (0, 255, 255)
COLOR_YOLO = (0, 220, 0)
COLOR_CONTACT = (0, 0, 255)
COLOR_BASELINE = (255, 0, 0)
COLOR_TEXT_BG = (0, 0, 0)
COLOR_TEXT = (255, 255, 255)


def _draw_box(img: np.ndarray, bbox, color, label: str) -> None:
    if bbox is None:
        return
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def _draw_cross(img: np.ndarray, x: float, y: float, color, radius: int = 6) -> None:
    px, py = int(round(float(x))), int(round(float(y)))
    cv2.circle(img, (px, py), radius, color, 2)
    cv2.line(img, (px - radius, py), (px + radius, py), color, 2)
    cv2.line(img, (px, py - radius), (px, py + radius), color, 2)


def _put_label_block(img: np.ndarray, lines: list[str]) -> None:
    if not lines:
        return
    line_h = 18
    width = max(220, max(len(line) for line in lines) * 8 + 12)
    height = line_h * len(lines) + 10
    cv2.rectangle(img, (5, 5), (5 + width, 5 + height), COLOR_TEXT_BG, -1)
    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (12, 24 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )


def baseline_point(bbox) -> tuple[float, float] | None:
    if bbox is None:
        return None
    x1, _y1, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) / 2.0, y2


def draw_contact_debug(
    image_bgr: np.ndarray,
    *,
    record: dict,
    output_path: str | Path,
) -> None:
    canvas = image_bgr.copy()
    ann_bbox = record.get("ann_bbox")
    yolo_bbox = record.get("bbox")
    _draw_box(canvas, ann_bbox, COLOR_ANN, "ann")
    _draw_box(canvas, yolo_bbox, COLOR_YOLO, "yolo")

    contact_x = record.get("contact_point_x")
    contact_y = record.get("contact_point_y")
    if contact_x is not None and contact_y is not None:
        _draw_cross(canvas, contact_x, contact_y, COLOR_CONTACT, radius=7)

    base = baseline_point(yolo_bbox or ann_bbox)
    if base is not None:
        _draw_cross(canvas, base[0], base[1], COLOR_BASELINE, radius=5)

    lines = [
        f"veh={record.get('vehicle_id')} cam={record.get('cam_index')} frame={record.get('frame_number')}",
        f"status={record.get('status')} reason={record.get('reject_reason', '-')}",
        f"uv=({record.get('target_u', 0):.3f},{record.get('target_v', 0):.3f}) iou={record.get('match_iou', 0):.3f}",
        "yellow=ann green=yolo red=contact blue=baseline",
    ]
    _put_label_block(canvas, lines)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def select_viz_records(
    accepted: list[dict],
    rejected: list[dict],
    *,
    count: int,
    rejected_fraction: float = 0.2,
) -> list[dict]:
    if count <= 0:
        return []
    rejected_count = min(len(rejected), int(round(count * rejected_fraction)))
    accepted_count = max(0, count - rejected_count)
    return _stride_sample(accepted, accepted_count) + _stride_sample(rejected, rejected_count)


def _stride_sample(rows: list[dict], count: int) -> list[dict]:
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return rows[:]
    step = len(rows) / float(count)
    return [rows[int(i * step)] for i in range(count)]
