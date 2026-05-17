"""Load CityFlow / MOTChallenge detection files for tracker input.

Format (datasets/ReadMe.txt)::

    frame, -1, left, top, width, height, conf, -1, -1, -1

Frames are 1-based in the file. Tracker pipelines query by 1-based frame id.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np


class MotDetectionStore:
    """Per-frame detections as ``Nx6`` xyxy + score + class (class defaults to 0)."""

    def __init__(
        self,
        path: str | Path,
        conf_thres: float = 0.0,
        default_class: int = 0,
    ):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Detection file not found: {self.path}")

        self.conf_thres = float(conf_thres)
        self.default_class = int(default_class)
        self._by_frame: dict[int, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        by_frame: dict[int, list[list[float]]] = defaultdict(list)
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 7:
                    continue
                frame_id = int(float(parts[0]))
                left = float(parts[2])
                top = float(parts[3])
                width = float(parts[4])
                height = float(parts[5])
                conf = float(parts[6])
                if conf < self.conf_thres:
                    continue
                if width <= 0 or height <= 0:
                    continue
                x1, y1 = left, top
                x2, y2 = left + width, top + height
                by_frame[frame_id].append(
                    [x1, y1, x2, y2, conf, float(self.default_class)]
                )

        self._by_frame = {
            f: np.asarray(rows, dtype=np.float32)
            for f, rows in by_frame.items()
        }
        self.max_frame = max(self._by_frame) if self._by_frame else 0

    def get(self, frame_id_1based: int) -> np.ndarray:
        """Return detections for MOT frame index (1-based)."""
        rows = self._by_frame.get(int(frame_id_1based))
        if rows is None or len(rows) == 0:
            return np.empty((0, 6), dtype=np.float32)
        return rows


def resolve_cityflow_det_paths(
    sources: list[str],
    det_files: list[str] | None = None,
    det_basename: str = "det_mask_rcnn.txt",
) -> list[str]:
    """Map each video source to ``<cam_dir>/det/<basename>`` unless explicit paths given."""
    if det_files is not None:
        if len(det_files) != len(sources):
            raise ValueError("det_files length must match sources length")
        return [str(Path(p)) for p in det_files]

    out = []
    for src in sources:
        det_path = Path(src).parent / "det" / det_basename
        if not det_path.is_file():
            raise FileNotFoundError(f"CityFlow detection file missing: {det_path}")
        out.append(str(det_path))
    return out
