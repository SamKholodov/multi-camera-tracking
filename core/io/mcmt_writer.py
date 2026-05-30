"""Writer for multi-camera multi-target tracking results.

Outputs three artefacts in ``output_dir``:

* ``track_results.txt`` — AICity Challenge Track-1 single-file format::

      cam_id obj_id frame_id xmin ymin width height xworld yworld

* ``per_cam/c{cam_id:03d}.txt`` — MOT16 format with **global** track ids
  (suitable for cross-camera ID-aware metrics per camera)::

      frame, id, x, y, w, h, conf, -1, -1, -1

* ``per_cam_local/c{cam_id:03d}.txt`` — MOT16 format with **local** (SCT)
  track ids, useful as a single-camera baseline.

The writer is incremental: ``add_frame`` is called after cross-camera
association on every step, ``finalize`` flushes results to disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np

from core.mot.types import WORLD_COORD_MISSING, homography_valid, world_point_from_row

from .calibration import project_bbox_bottom_center


class MCMTResultWriter:
    def __init__(
        self,
        output_dir: Union[str, Path],
        cam_ids: Iterable[int],
        homographies_image_to_world: Iterable[np.ndarray],
    ):
        self.output_dir = Path(output_dir)
        self.per_cam_dir = self.output_dir / "per_cam"
        self.per_cam_local_dir = self.output_dir / "per_cam_local"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.per_cam_dir.mkdir(parents=True, exist_ok=True)
        self.per_cam_local_dir.mkdir(parents=True, exist_ok=True)

        self.cam_ids = list(cam_ids)
        self.homographies_i2w = [
            np.asarray(h, dtype=np.float64) for h in homographies_image_to_world
        ]
        if len(self.cam_ids) != len(self.homographies_i2w):
            raise ValueError("cam_ids and homographies length mismatch")

        # Buffers: list[(cam_id, gid, frame_id, x, y, w, h, xw, yw)] and
        # per-cam dicts of MOT rows.
        self._aicity_rows: list[tuple[int, int, int, float, float, float, float, float, float]] = []
        self._per_cam_global: dict[int, list[tuple[int, int, float, float, float, float, float]]] = {
            cid: [] for cid in self.cam_ids
        }
        self._per_cam_local: dict[int, list[tuple[int, int, float, float, float, float, float]]] = {
            cid: [] for cid in self.cam_ids
        }

    def add_frame(
        self,
        cam_index: int,
        frame_id: int,
        tracks: np.ndarray,
        local_to_global: dict,
    ) -> None:
        """Record one frame's tracks for a single camera.

        Args:
            cam_index: position of the camera in ``cam_ids`` / ``homographies``.
            frame_id: 1-based frame number to write into the MOT/AICity files.
            tracks: ndarray (N, >=8) with columns
                ``[x1, y1, x2, y2, local_tid, conf, det_idx, has_detection]``
                as returned by the tracker wrappers in this project.
            local_to_global: mapping ``(cam_index, local_tid) -> global_id`` —
                pass ``MultiCameraTrackingPipeline.local_to_global`` directly.
        """
        if tracks is None or len(tracks) == 0:
            return

        cam_id = self.cam_ids[cam_index]
        H_i2w = self.homographies_i2w[cam_index]
        # MOT: one global id per frame -> one row; on conflict keep the max conf.
        best_global: dict[int, tuple] = {}

        for row in tracks:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            local_tid = int(row[4])
            conf = float(row[5]) if len(row) > 5 else 1.0

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            wpt = world_point_from_row(row)
            if wpt is not None:
                xw, yw = wpt
            elif homography_valid(H_i2w):
                xw, yw = project_bbox_bottom_center(H_i2w, x1, y1, x2, y2)
            else:
                xw, yw = WORLD_COORD_MISSING, WORLD_COORD_MISSING

            self._per_cam_local[cam_id].append(
                (frame_id, local_tid, x1, y1, w, h, conf)
            )

            gid = local_to_global.get((cam_index, local_tid))
            if gid is None:
                continue

            gid = int(gid)
            cand = (cam_id, gid, frame_id, x1, y1, w, h, xw, yw, conf)
            prev = best_global.get(gid)
            if prev is None or conf > prev[-1]:
                best_global[gid] = cand

        for cam_id, gid, frame_id, x1, y1, w, h, xw, yw, conf in best_global.values():
            self._aicity_rows.append(
                (cam_id, gid, frame_id, x1, y1, w, h, xw, yw)
            )
            self._per_cam_global[cam_id].append(
                (frame_id, gid, x1, y1, w, h, conf)
            )

    def finalize(self) -> dict:
        """Flush all buffered rows to disk. Returns a dict of created paths."""
        out_paths: dict = {"per_cam": {}, "per_cam_local": {}}

        track_file = self.output_dir / "track_results.txt"
        with track_file.open("w", encoding="utf-8") as f:
            for cam_id, gid, frame_id, x, y, w, h, xw, yw in self._aicity_rows:
                f.write(
                    f"{cam_id} {gid} {frame_id} "
                    f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} "
                    f"{xw:.6f} {yw:.6f}\n"
                )
        out_paths["aicity"] = track_file

        for cam_id, rows in self._per_cam_global.items():
            path = self.per_cam_dir / f"c{cam_id:03d}.txt"
            self._write_mot(path, rows)
            out_paths["per_cam"][cam_id] = path

        for cam_id, rows in self._per_cam_local.items():
            path = self.per_cam_local_dir / f"c{cam_id:03d}.txt"
            self._write_mot(path, rows)
            out_paths["per_cam_local"][cam_id] = path

        return out_paths

    @staticmethod
    def _write_mot(path: Path, rows: list) -> None:
        with path.open("w", encoding="utf-8") as f:
            for frame_id, tid, x, y, w, h, conf in rows:
                f.write(
                    f"{frame_id},{tid},{x:.2f},{y:.2f},"
                    f"{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
                )
