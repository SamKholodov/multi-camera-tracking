"""Dataset preparation for contact point regression."""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from core.detector.detector import Detector

from .annotations import discover_annotation_csvs, load_contact_records
from .matching import best_detection_match
from .validation import (
    RejectReason,
    ValidationConfig,
    compute_uv,
    validate_annotation_record,
    write_jsonl,
    write_stats,
)
from .viz import draw_contact_debug, select_viz_records

LOGGER = logging.getLogger("core.geometry.contact_point.prepare")

DEFAULT_TARGET_CLASSES = [1, 2, 3, 5, 7]


@dataclass(frozen=True)
class PrepareConfig:
    gta_root: Path
    output_dir: Path
    detector_model: str = "models/yolo26l_fine_tune_gta.pt"
    target_classes: tuple[int, ...] = tuple(DEFAULT_TARGET_CLASSES)
    conf_thres: float = 0.2
    imgsz: int = 640
    device: str | int | None = None
    match_iou_threshold: float = 0.5
    viz_count: int = 100
    val_fraction: float = 0.1
    seed: int = 42
    max_images: int | None = None
    skip_detector: bool = False


def _group_records_by_image(records) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for record in records:
        grouped[record.image_path].append(record)
    return grouped


def _build_splits(
    manifest_rows: list[dict],
    *,
    val_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    accepted_by_frame: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, row in enumerate(manifest_rows):
        if row.get("status") != "accepted":
            continue
        accepted_by_frame[(int(row["cam_index"]), int(row["frame_number"]))].append(idx)

    frame_keys = sorted(accepted_by_frame.keys())
    rng = random.Random(seed)
    rng.shuffle(frame_keys)

    val_count = max(1, int(round(len(frame_keys) * val_fraction))) if frame_keys else 0
    val_frames = set(frame_keys[:val_count])

    train: list[int] = []
    val: list[int] = []
    for frame_key, indices in accepted_by_frame.items():
        target = val if frame_key in val_frames else train
        target.extend(indices)

    return {
        "train": sorted(train),
        "val": sorted(val),
        "sanity_cam_holdout": [],
    }


def _apply_detector_match(
    row: dict,
    *,
    detections: list[list[float]],
    used_indices: set[int],
    config: ValidationConfig,
) -> tuple[dict, str | None]:
    if row.get("status") != "accepted":
        return row, row.get("reject_reason")

    ann_bbox = row["ann_bbox"]
    det_idx, match = best_detection_match(ann_bbox, detections, used_indices=used_indices)
    row["match_iou"] = float(match.iou)
    row["det_score"] = float(match.score)
    row["det_class_id"] = int(match.class_id)

    if match.bbox is None:
        row.update({"status": "rejected", "reject_reason": RejectReason.NO_DETECTION.value})
        return row, RejectReason.NO_DETECTION.value

    if match.iou < config.match_iou_threshold:
        row.update({"status": "rejected", "reject_reason": RejectReason.LOW_IOU.value})
        return row, RejectReason.LOW_IOU.value

    used_indices.add(det_idx)
    det_bbox = list(match.bbox)
    u, v = compute_uv(row["contact_point_x"], row["contact_point_y"], det_bbox)
    row["bbox"] = det_bbox
    row["target_u"] = float(u)
    row["target_v"] = float(v)
    row["status"] = "accepted"
    row.pop("reject_reason", None)
    return row, None


def _skip_detector_row(row: dict) -> tuple[dict, str | None]:
    if row.get("status") != "accepted":
        return row, row.get("reject_reason")
    ann_bbox = list(row["ann_bbox"])
    row["bbox"] = ann_bbox
    row["match_iou"] = 1.0
    row["det_score"] = 1.0
    row["det_class_id"] = int(row.get("class_coco_id", -1))
    return row, None


def prepare_contact_point_dataset(config: PrepareConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        det_cfg = config.get("detector", {})
        config = PrepareConfig(
            gta_root=Path(config.get("gta_root", "datasets/gta_mcmt_with_points")),
            output_dir=Path(config.get("output_dir", "datasets/gta_mcmt_with_points/contact_point")),
            detector_model=det_cfg.get("model", "models/yolo26l_fine_tune_gta.pt"),
            target_classes=tuple(det_cfg.get("target_classes", DEFAULT_TARGET_CLASSES)),
            conf_thres=float(det_cfg.get("conf_thres", 0.2)),
            imgsz=int(det_cfg.get("imgsz", 640)),
            device=det_cfg.get("device"),
            match_iou_threshold=float(config.get("match_iou_threshold", 0.5)),
            viz_count=int(config.get("viz_count", 100)),
            val_fraction=float(config.get("val_fraction", 0.1)),
            seed=int(config.get("seed", 42)),
            max_images=config.get("max_images"),
            skip_detector=bool(config.get("skip_detector", False)),
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    validation_config = ValidationConfig(match_iou_threshold=config.match_iou_threshold)

    csv_paths = discover_annotation_csvs(config.gta_root)
    records = load_contact_records(csv_paths)
    grouped = _group_records_by_image(records)
    image_paths = sorted(grouped.keys())
    if config.max_images is not None:
        image_paths = image_paths[: int(config.max_images)]

    detector = None
    if not config.skip_detector:
        detector = Detector(
            model=config.detector_model,
            target_classes=list(config.target_classes),
            conf_thres=config.conf_thres,
            imgsz=config.imgsz,
            device=config.device,
        )

    manifest_rows: list[dict] = []
    rejected_reasons: list[str] = []
    detection_cache: dict[str, list[list[float]]] = {}

    for image_idx, image_path in enumerate(image_paths):
        if image_idx % 100 == 0:
            LOGGER.info("Processing image %d / %d: %s", image_idx + 1, len(image_paths), image_path)

        detections: list[list[float]] = []
        if not config.skip_detector:
            if image_path not in detection_cache:
                image = cv2.imread(image_path)
                if image is None:
                    detection_cache[image_path] = []
                else:
                    detections, _ = detector.detect(image)
                    detection_cache[image_path] = detections
            else:
                detections = detection_cache[image_path]

        used_indices: set[int] = set()
        for record in grouped[image_path]:
            row, ann_reason = validate_annotation_record(record, config=validation_config)
            if config.skip_detector:
                row, det_reason = _skip_detector_row(row)
            else:
                row, det_reason = _apply_detector_match(
                    row,
                    detections=detections,
                    used_indices=used_indices,
                    config=validation_config,
                )

            reason = det_reason or ann_reason
            if reason is not None:
                rejected_reasons.append(reason)
            manifest_rows.append(row)

    manifest_path = config.output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, manifest_rows)

    accepted_rows = [row for row in manifest_rows if row.get("status") == "accepted"]
    rejected_rows = [row for row in manifest_rows if row.get("status") != "accepted"]
    stats = write_stats(
        config.output_dir / "stats.json",
        total=len(manifest_rows),
        accepted=len(accepted_rows),
        rejected=rejected_reasons,
    )

    splits = _build_splits(manifest_rows, val_fraction=config.val_fraction, seed=config.seed)
    splits_path = config.output_dir / "splits.json"
    splits_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    viz_dir = config.output_dir / "debug" / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for old_png in viz_dir.glob("*.png"):
        old_png.unlink()

    viz_records = select_viz_records(accepted_rows, rejected_rows, count=config.viz_count)
    for viz_idx, record in enumerate(viz_records):
        image = cv2.imread(record["image_path"])
        if image is None:
            continue
        out_name = (
            f"{viz_idx:04d}_cam{record.get('cam_index')}_"
            f"f{record.get('frame_number')}_v{record.get('vehicle_id')}.png"
        )
        draw_contact_debug(image, record=record, output_path=viz_dir / out_name)

    LOGGER.info(
        "Prepared %d rows (%d accepted, %d rejected). train=%d val=%d viz=%d",
        len(manifest_rows),
        len(accepted_rows),
        len(rejected_rows),
        len(splits["train"]),
        len(splits["val"]),
        len(viz_records),
    )
    return {
        "manifest_path": str(manifest_path),
        "splits_path": str(splits_path),
        "stats": stats,
        "viz_count": len(viz_records),
    }
