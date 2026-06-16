"""GTA MCMT synthetic dataset I/O.

CSV columns (0-based):
  0 frame_idx, 1 cam_id (image id), 2 obj_id, 3 obj_class,
  4 license_plate, 5-8 bbox cx/cy/w/h, 9-11 world xyz, 12 confidence.

Images: ``cam-{N}/image_{cam_id}.jpg`` (cam_id from CSV col 1, not frame_idx).
Sync: k-th unique snapshot row in each camera CSV = same moment across cameras.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

NUM_CAMERAS = 4

COL_FRAME_IDX = 0
COL_CAM_ID = 1
COL_OBJ_ID = 2
COL_OBJ_CLASS = 3
COL_BBOX_CX = 5
COL_BBOX_CY = 6
COL_BBOX_W = 7
COL_BBOX_H = 8


def scale_center_bbox(
    cx: float, cy: float, w: float, h: float, scale: float
) -> tuple[float, float, float, float]:
    """Shrink or expand (cx, cy, w, h) around center; scale=1.0 is unchanged."""
    return cx, cy, w * scale, h * scale


def center_bbox_to_xyxy(
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    scale: float = 1.0,
) -> tuple[float, float, float, float]:
    """DatasetCreator stores (cx, cy, w, h); convert to top-left pixel corners."""
    if scale != 1.0:
        cx, cy, w, h = scale_center_bbox(cx, cy, w, h, scale)
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


def center_bbox_to_tlwh(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    x1, y1, _, _ = center_bbox_to_xyxy(cx, cy, w, h)
    return x1, y1, w, h


@dataclass(frozen=True)
class GtaAnnotation:
    obj_id: int
    cx: float
    cy: float
    w: float
    h: float
    obj_class: int
    world_x: float = 0.0
    world_y: float = 0.0
    world_z: float = 0.0
    confidence: float = 1.0


@dataclass(frozen=True)
class GtaSnapshot:
    frame_idx: str
    cam_id: str
    sync_index: int
    annotations: tuple[GtaAnnotation, ...]


def parse_csv_row(parts: list[str]) -> GtaAnnotation | None:
    if len(parts) < 9:
        return None
    try:
        cx, cy, w, h = (
            float(parts[COL_BBOX_CX]),
            float(parts[COL_BBOX_CY]),
            float(parts[COL_BBOX_W]),
            float(parts[COL_BBOX_H]),
        )
        world_x = float(parts[9]) if len(parts) > 9 else 0.0
        world_y = float(parts[10]) if len(parts) > 10 else 0.0
        world_z = float(parts[11]) if len(parts) > 11 else 0.0
        confidence = float(parts[12]) if len(parts) > 12 else 1.0
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return GtaAnnotation(
        obj_id=int(parts[COL_OBJ_ID]),
        cx=cx,
        cy=cy,
        w=w,
        h=h,
        obj_class=int(parts[COL_OBJ_CLASS]),
        world_x=world_x,
        world_y=world_y,
        world_z=world_z,
        confidence=confidence,
    )


def load_snapshots(csv_path: Path) -> list[GtaSnapshot]:
    """Ordered unique snapshots: one entry per (frame_idx, cam_id) pair."""
    order: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    by_key: dict[tuple[str, str], list[GtaAnnotation]] = {}

    with csv_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            key = (parts[COL_FRAME_IDX], parts[COL_CAM_ID])
            if key not in seen:
                seen.add(key)
                order.append(key)
            ann = parse_csv_row(parts)
            if ann is None:
                continue
            by_key.setdefault(key, []).append(ann)

    return [
        GtaSnapshot(
            frame_idx=frame_idx,
            cam_id=cam_id,
            sync_index=k,
            annotations=tuple(by_key.get((frame_idx, cam_id), [])),
        )
        for k, (frame_idx, cam_id) in enumerate(order)
    ]


def cam_index_from_dir(cam_dir: Path) -> int:
    name = cam_dir.name
    if name.startswith("cam-"):
        return int(name.split("-", 1)[1])
    raise ValueError(f"Expected cam-N directory, got: {cam_dir}")


def coords_csv_path(cam_dir: Path) -> Path:
    cam = cam_index_from_dir(cam_dir)
    return cam_dir / f"coords_cam_{cam}.csv"


def image_path_for_cam_dir(cam_dir: Path, snapshot_cam_id: str) -> Path:
    return cam_dir / f"image_{snapshot_cam_id}.jpg"


def is_gta_mcmt_cam_dir(source: str | Path) -> bool:
    path = Path(source)
    if not path.is_dir():
        return False
    csv_path = coords_csv_path(path)
    return csv_path.is_file()


class GtaMcmtFrameSource:
    """OpenCV-like reader for one GTA MCMT camera folder."""

    def __init__(self, cam_dir: str | Path):
        self.cam_dir = Path(cam_dir)
        self.cam_index = cam_index_from_dir(self.cam_dir)
        self.snapshots = load_snapshots(coords_csv_path(self.cam_dir))
        self._cursor = 0
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cursor >= len(self.snapshots):
            return False, None
        snap = self.snapshots[self._cursor]
        image_path = image_path_for_cam_dir(self.cam_dir, snap.cam_id)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Missing GTA MCMT image: {image_path}")
        self._cursor += 1
        return True, frame

    def release(self) -> None:
        self._opened = False


class GtaMcmtDataset:
    """Multi-camera synced access by sync index k."""

    def __init__(self, dataset_root: str | Path, num_cameras: int = NUM_CAMERAS):
        self.root = Path(dataset_root)
        self.num_cameras = num_cameras
        self.cam_dirs = [self.root / f"cam-{cam}" for cam in range(num_cameras)]
        self.snapshots_by_cam: list[list[GtaSnapshot]] = [
            load_snapshots(coords_csv_path(d)) for d in self.cam_dirs
        ]
        self.length = min(len(s) for s in self.snapshots_by_cam) if self.snapshots_by_cam else 0

    def __len__(self) -> int:
        return self.length

    def read_sync(self, sync_index: int) -> list[np.ndarray]:
        if sync_index < 0 or sync_index >= self.length:
            raise IndexError(f"sync_index {sync_index} out of range [0, {self.length})")
        frames: list[np.ndarray] = []
        for cam, snapshots in enumerate(self.snapshots_by_cam):
            snap = snapshots[sync_index]
            path = image_path_for_cam_dir(self.cam_dirs[cam], snap.cam_id)
            frame = cv2.imread(str(path))
            if frame is None:
                raise FileNotFoundError(f"Missing synced image cam-{cam} k={sync_index}: {path}")
            frames.append(frame)
        return frames

    def snapshot(self, cam: int, sync_index: int) -> GtaSnapshot:
        return self.snapshots_by_cam[cam][sync_index]
