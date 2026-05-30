"""Pipeline wrapper for classic SORT (``core.sort``)."""
from __future__ import annotations

import numpy as np

from core.sort import Sort, iou_batch


class SortTracker:
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        det_thresh: float = 0.3,
        **kwargs,
    ):
        self.det_thresh = float(det_thresh)
        self.tracker = Sort(
            max_age=int(max_age),
            min_hits=int(min_hits),
            iou_threshold=float(iou_threshold),
        )

    @staticmethod
    def _filter_detections(detections: np.ndarray, det_thresh: float) -> np.ndarray:
        if detections is None or len(detections) == 0:
            return np.empty((0, 6), dtype=np.float32)
        dets = np.asarray(detections, dtype=np.float32)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        if dets.shape[1] < 5:
            return np.empty((0, 6), dtype=np.float32)
        keep = dets[:, 4] >= det_thresh
        return dets[keep]

    @staticmethod
    def _match_det_indices(dets_xyxy: np.ndarray, track_xyxy: np.ndarray, iou_threshold: float):
        n_tracks = len(track_xyxy)
        if n_tracks == 0:
            return []
        if len(dets_xyxy) == 0:
            return [-1] * n_tracks

        ious = iou_batch(dets_xyxy[:, :4], track_xyxy[:, :4])
        det_indices = []
        for j in range(n_tracks):
            best_i = int(np.argmax(ious[:, j]))
            best_iou = float(ious[best_i, j])
            det_indices.append(best_i if best_iou >= iou_threshold else -1)
        return det_indices

    def update(self, detections, frame, embs=None):
        del frame, embs  # SORT is motion-only; frame kept for API compatibility.

        dets = self._filter_detections(detections, self.det_thresh)
        if len(dets) == 0:
            sort_out = self.tracker.update(
                np.empty((0, 5), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        else:
            labels = (
                dets[:, 5].astype(np.float32)
                if dets.shape[1] > 5
                else np.zeros(len(dets), dtype=np.float32)
            )
            sort_out = self.tracker.update(dets[:, :5].astype(np.float32), labels)

        if sort_out is None or len(sort_out) == 0:
            return np.empty((0, 8), dtype=np.float32)

        sort_out = np.asarray(sort_out, dtype=np.float32)
        track_boxes = sort_out[:, :4]
        det_indices = self._match_det_indices(
            dets[:, :4] if len(dets) else np.empty((0, 4)),
            track_boxes,
            self.tracker.iou_threshold,
        )

        out = []
        for i, row in enumerate(sort_out):
            x1, y1, x2, y2, tid = row[:5]
            tid = int(tid)
            det_idx = int(det_indices[i])
            conf = float(dets[det_idx, 4]) if det_idx >= 0 else 1.0
            has_detection = 1.0 if det_idx >= 0 else 0.0
            out.append([x1, y1, x2, y2, tid, conf, det_idx, has_detection])

        return np.asarray(out, dtype=np.float32)

    def get_track_feature_map(self):
        return {}
