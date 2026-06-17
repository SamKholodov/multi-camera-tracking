"""CSV parsing for 2D contact point annotations."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from core.io.gta_mcmt import center_bbox_to_xyxy


DEFAULT_IMAGE_PATTERN = "{cam_dir}/image_{frame_number}.jpg"


@dataclass(frozen=True)
class GtaContactRecord:
    """One annotated vehicle contact point in image coordinates."""

    image_path: str
    annotation_csv: str
    row_index: int
    cam_index: int
    frame_gta_time: float | None
    frame_number: int
    vehicle_id: int
    class_coco_id: int
    license_plate: str
    ann_bbox: tuple[float, float, float, float]
    contact_point_x: float
    contact_point_y: float
    confidence: float

    @property
    def ann_width(self) -> float:
        return self.ann_bbox[2] - self.ann_bbox[0]

    @property
    def ann_height(self) -> float:
        return self.ann_bbox[3] - self.ann_bbox[1]

    def to_json(self) -> dict:
        data = asdict(self)
        data["ann_bbox"] = list(self.ann_bbox)
        return data


def infer_cam_index(path: str | Path) -> int:
    """Infer cam index from a path containing a ``cam-N`` component."""

    for part in Path(path).parts:
        if part.startswith("cam-"):
            return int(part.split("-", 1)[1])
    stem = Path(path).stem
    if "cam_" in stem:
        return int(stem.rsplit("cam_", 1)[1])
    raise ValueError(f"Cannot infer cam index from path: {path}")


def resolve_image_path(
    *,
    cam_dir: str | Path,
    frame_number: int,
    image_pattern: str = DEFAULT_IMAGE_PATTERN,
) -> str:
    path = image_pattern.format(cam_dir=str(cam_dir), frame_number=frame_number)
    return str(Path(path))


def discover_annotation_csvs(gta_root: str | Path, annotation_csv: str | None = "auto") -> list[Path]:
    root = Path(gta_root)
    if annotation_csv in (None, "", "auto"):
        return sorted(root.glob("cam-*/coords_cam_*.csv"))
    raw = Path(annotation_csv)
    if raw.is_absolute():
        matches = sorted(raw.parent.glob(raw.name)) if any(ch in raw.name for ch in "*?[]") else [raw]
    else:
        pattern = str(raw)
        matches = sorted(root.glob(pattern)) if any(ch in pattern for ch in "*?[]") else [root / raw]
    return [p for p in matches if p.is_file()]


def _get(row: dict[str, str], *names: str, default: str | None = None) -> str:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    if default is not None:
        return default
    raise KeyError(f"Missing required columns: {', '.join(names)}")


def _parse_record(
    row: dict[str, str],
    *,
    csv_path: Path,
    row_index: int,
    cam_index: int,
    image_pattern: str,
) -> GtaContactRecord:
    frame_gta_time_s = row.get("frame_gta_time", "")
    frame_gta_time = float(frame_gta_time_s) if frame_gta_time_s not in ("", None) else None
    frame_number = int(float(_get(row, "frame_number", "cam_id")))
    vehicle_id = int(float(_get(row, "vehicle_id", "obj_id")))
    class_coco_id = int(float(_get(row, "class_coco_id", "obj_class", default="2")))
    license_plate = row.get("license_plate", "")
    cx = float(_get(row, "bb_x_center", "cx"))
    cy = float(_get(row, "bb_y_center", "cy"))
    width = float(_get(row, "width", "w"))
    height = float(_get(row, "height", "h"))
    contact_x = float(_get(row, "bottom_px", "contact_point_x"))
    contact_y = float(_get(row, "bottom_py", "contact_point_y"))
    confidence = float(_get(row, "confidence", default="1.0"))
    cam_dir = csv_path.parent
    image_path = resolve_image_path(
        cam_dir=cam_dir,
        frame_number=frame_number,
        image_pattern=image_pattern,
    )
    return GtaContactRecord(
        image_path=image_path,
        annotation_csv=str(csv_path),
        row_index=row_index,
        cam_index=cam_index,
        frame_gta_time=frame_gta_time,
        frame_number=frame_number,
        vehicle_id=vehicle_id,
        class_coco_id=class_coco_id,
        license_plate=license_plate,
        ann_bbox=center_bbox_to_xyxy(cx, cy, width, height),
        contact_point_x=contact_x,
        contact_point_y=contact_y,
        confidence=confidence,
    )


def _rows_with_header(csv_path: Path) -> Iterable[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        sample = fh.readline()
        fh.seek(0)
        has_header = "frame_number" in sample or "bb_x_center" in sample or "bottom_px" in sample
        if has_header:
            yield from csv.DictReader(fh)
            return

        fieldnames = [
            "frame_gta_time",
            "frame_number",
            "vehicle_id",
            "class_coco_id",
            "license_plate",
            "bb_x_center",
            "bb_y_center",
            "width",
            "height",
            "x_world",
            "y_world",
            "z_world",
            "bottom_world_x",
            "bottom_world_y",
            "bottom_world_z",
            "bottom_px",
            "bottom_py",
            "yaw_vehicle",
            "confidence",
        ]
        yield from csv.DictReader(fh, fieldnames=fieldnames)


def load_contact_records(
    csv_paths: Iterable[str | Path],
    *,
    image_pattern: str = DEFAULT_IMAGE_PATTERN,
) -> list[GtaContactRecord]:
    records: list[GtaContactRecord] = []
    for csv_path_raw in csv_paths:
        csv_path = Path(csv_path_raw)
        cam_index = infer_cam_index(csv_path)
        for row_index, row in enumerate(_rows_with_header(csv_path), start=1):
            try:
                records.append(
                    _parse_record(
                        row,
                        csv_path=csv_path,
                        row_index=row_index,
                        cam_index=cam_index,
                        image_pattern=image_pattern,
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return records
