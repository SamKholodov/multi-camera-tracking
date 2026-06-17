"""Validation helpers for 2D contact point annotations."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

import cv2

from .annotations import GtaContactRecord


class RejectReason(str, Enum):
    MISSING_IMAGE = "missing_image"
    DEGENERATE_BBOX = "degenerate_bbox"
    BBOX_OUT_OF_BOUNDS = "bbox_out_of_bounds"
    POINT_OUT_OF_BOUNDS = "point_out_of_bounds"
    ABNORMAL_UV = "abnormal_uv"
    LOW_CONFIDENCE = "low_confidence"
    NO_DETECTION = "no_detection"
    LOW_IOU = "low_iou"


@dataclass(frozen=True)
class UVBounds:
    u_min: float = -0.2
    u_max: float = 1.2
    v_min: float = 0.5
    v_max: float = 1.3

    def contains(self, u: float, v: float) -> bool:
        return self.u_min <= u <= self.u_max and self.v_min <= v <= self.v_max


@dataclass(frozen=True)
class ValidationConfig:
    match_iou_threshold: float = 0.5
    min_bbox_size: float = 8.0
    min_confidence: float = 0.0
    uv_bounds: UVBounds = UVBounds()


def compute_uv(
    contact_x: float,
    contact_y: float,
    bbox: tuple[float, float, float, float] | list[float],
) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    return (float(contact_x) - x1) / width, (float(contact_y) - y1) / height


def bbox_wh(bbox: tuple[float, float, float, float] | list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return x2 - x1, y2 - y1


def point_in_image(x: float, y: float, width: int, height: int) -> bool:
    return 0.0 <= float(x) < float(width) and 0.0 <= float(y) < float(height)


def bbox_inside_image(
    bbox: tuple[float, float, float, float] | list[float],
    width: int,
    height: int,
) -> bool:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return x1 >= 0.0 and y1 >= 0.0 and x2 <= float(width) and y2 <= float(height)


def validate_bbox_and_point(
    *,
    bbox: tuple[float, float, float, float] | list[float],
    contact_x: float,
    contact_y: float,
    image_width: int,
    image_height: int,
    confidence: float,
    config: ValidationConfig,
    require_bbox_inside: bool = True,
) -> tuple[RejectReason | None, float, float]:
    width, height = bbox_wh(bbox)
    if width < config.min_bbox_size or height < config.min_bbox_size:
        return RejectReason.DEGENERATE_BBOX, 0.0, 0.0
    if confidence < config.min_confidence:
        return RejectReason.LOW_CONFIDENCE, 0.0, 0.0
    if require_bbox_inside and not bbox_inside_image(bbox, image_width, image_height):
        return RejectReason.BBOX_OUT_OF_BOUNDS, 0.0, 0.0
    if not point_in_image(contact_x, contact_y, image_width, image_height):
        return RejectReason.POINT_OUT_OF_BOUNDS, 0.0, 0.0
    u, v = compute_uv(contact_x, contact_y, bbox)
    if not config.uv_bounds.contains(u, v):
        return RejectReason.ABNORMAL_UV, u, v
    return None, u, v


def validate_annotation_record(
    record: GtaContactRecord,
    *,
    config: ValidationConfig,
) -> tuple[dict, dict | None]:
    out = record.to_json()
    image = cv2.imread(record.image_path)
    if image is None:
        rejected = {**out, "status": "rejected", "reject_reason": RejectReason.MISSING_IMAGE.value}
        return rejected, RejectReason.MISSING_IMAGE.value

    image_height, image_width = image.shape[:2]
    reason, u, v = validate_bbox_and_point(
        bbox=record.ann_bbox,
        contact_x=record.contact_point_x,
        contact_y=record.contact_point_y,
        image_width=image_width,
        image_height=image_height,
        confidence=record.confidence,
        config=config,
    )
    out.update(
        {
            "image_width": image_width,
            "image_height": image_height,
            "target_u": float(u),
            "target_v": float(v),
            "baseline_u": 0.5,
            "baseline_v": 1.0,
        }
    )
    if reason is not None:
        out.update({"status": "rejected", "reject_reason": reason.value})
        return out, reason.value
    out["status"] = "accepted"
    return out, None


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_stats(
    path: str | Path,
    *,
    total: int,
    accepted: int,
    rejected: Iterable[str],
) -> dict:
    by_reason = Counter(rejected)
    payload = {
        "total_candidates": int(total),
        "accepted": int(accepted),
        "rejected": int(sum(by_reason.values())),
        "by_reason": dict(sorted(by_reason.items())),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
