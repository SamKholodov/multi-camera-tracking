"""Region-of-interest masks for CityFlow / AICity cameras (roi.jpg)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import cv2
import numpy as np

PointMode = Literal["bottom_center", "center"]
ROISpec = Union[str, Path, np.ndarray, list]


def load_roi_mask(path: Union[str, Path]) -> np.ndarray:
    """Load roi.jpg as a single-channel uint8 mask (non-zero = inside ROI)."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read ROI mask: {path}")
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D ROI mask, got shape {mask.shape} for {path}")
    return np.ascontiguousarray(mask)


def resolve_roi_paths(
    sources: list[str],
    roi_paths: Optional[list[Optional[str]]] = None,
) -> list[Optional[ROISpec]]:
    """Build per-camera ROI paths.

    * ``roi_paths is None`` — no ROI.
    * ``roi_paths == "auto"`` (in YAML as string) handled in run.py.
    * Explicit list — one path per source (``null`` disables ROI for that cam).
    * Missing entry — fallback to ``<video_dir>/roi.jpg``.
    """
    if roi_paths is None:
        return [None] * len(sources)

    resolved: list[Optional[ROISpec]] = []
    for i, src in enumerate(sources):
        if i < len(roi_paths) and roi_paths[i] is not None:
            resolved.append(roi_paths[i])
        else:
            candidate = Path(src).parent / "roi.jpg"
            resolved.append(str(candidate) if candidate.exists() else None)
    return resolved


class ROIFilter:
    """Keep detections/tracks whose reference point lies inside the ROI mask."""

    def __init__(
        self,
        mask: np.ndarray,
        point_mode: PointMode = "bottom_center",
        threshold: int = 0,
    ):
        self.mask = np.asarray(mask)
        if self.mask.ndim != 2:
            self.mask = np.squeeze(self.mask)
        self.point_mode = point_mode
        self.threshold = int(threshold)
        self._h, self._w = int(self.mask.shape[0]), int(self.mask.shape[1])

    @classmethod
    def from_path(
        cls,
        path: Union[str, Path],
        point_mode: PointMode = "bottom_center",
        threshold: int = 0,
    ) -> "ROIFilter":
        return cls(load_roi_mask(path), point_mode=point_mode, threshold=threshold)

    @classmethod
    def from_spec(
        cls,
        spec: ROISpec,
        point_mode: PointMode = "bottom_center",
        threshold: int = 0,
    ) -> "ROIFilter":
        if isinstance(spec, (str, Path)):
            return cls.from_path(spec, point_mode=point_mode, threshold=threshold)
        return cls(np.asarray(spec), point_mode=point_mode, threshold=threshold)

    def _reference_point(self, x1: float, y1: float, x2: float, y2: float) -> tuple[int, int]:
        if self.point_mode == "center":
            px = int(round((x1 + x2) / 2.0))
            py = int(round((y1 + y2) / 2.0))
        else:
            px = int(round((x1 + x2) / 2.0))
            py = int(round(y2))
        return px, py

    def contains_bbox(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        px, py = self._reference_point(x1, y1, x2, y2)
        if px < 0 or py < 0 or px >= self._w or py >= self._h:
            return False
        return bool(self.mask[py, px] > self.threshold)

    def _inside_mask(self, x1, y1, x2, y2) -> np.ndarray:
        """Vectorized inside test for arrays x1,y1,x2,y2."""
        x1 = np.asarray(x1, dtype=np.float64)
        y1 = np.asarray(y1, dtype=np.float64)
        x2 = np.asarray(x2, dtype=np.float64)
        y2 = np.asarray(y2, dtype=np.float64)
        if self.point_mode == "center":
            px = np.round((x1 + x2) / 2.0).astype(np.int64)
            py = np.round((y1 + y2) / 2.0).astype(np.int64)
        else:
            px = np.round((x1 + x2) / 2.0).astype(np.int64)
            py = np.round(y2).astype(np.int64)
        inside = np.zeros(px.shape, dtype=bool)
        valid = (px >= 0) & (py >= 0) & (px < self._w) & (py < self._h)
        if np.any(valid):
            inside[valid] = self.mask[py[valid], px[valid]] > self.threshold
        return inside

    def filter_xyxy_array(self, boxes: np.ndarray) -> np.ndarray:
        """Filter rows with bbox in first four columns (x1,y1,x2,y2)."""
        if boxes is None or len(boxes) == 0:
            return boxes
        arr = np.atleast_2d(np.asarray(boxes, dtype=np.float64))
        keep = self._inside_mask(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])
        return arr[keep]

    def filter_mot(self, mot: np.ndarray) -> np.ndarray:
        """Filter MOT16 rows ``frame,id,x,y,w,h,...`` using bottom-center of box."""
        if mot is None or len(mot) == 0:
            return mot
        arr = np.atleast_2d(np.asarray(mot, dtype=np.float64))
        if arr.shape[1] < 6:
            return np.empty((0, arr.shape[1] if arr.ndim == 2 else 0))
        x, y, w, h = arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]
        keep = self._inside_mask(x, y, x + w, y + h)
        return arr[keep]
