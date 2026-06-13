import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.eval.cityflow_protocol import (
    apply_cityflow_filters,
    cross_camera_gt_ids,
    filter_pred_multi_cam_only,
    filter_pred_tracks_with_benchmark_gt,
    infer_pred_id_mode,
)


def _mot_row(frame: int, obj_id: int, x: float, y: float, w: float, h: float) -> list[float]:
    return [frame, obj_id, x, y, w, h, 1.0, -1, -1, -1]


class TestCityFlowProtocol(unittest.TestCase):
    def test_infer_pred_id_mode(self):
        self.assertEqual(infer_pred_id_mode("outputs/run/per_cam_local"), "local")
        self.assertEqual(infer_pred_id_mode("outputs/run/per_cam"), "global")

    def test_cross_camera_gt_ids(self):
        gt_by_cam = {
            6: np.array([_mot_row(1, 10, 0, 0, 10, 10)], dtype=float),
            7: np.array([_mot_row(1, 10, 0, 0, 10, 10)], dtype=float),
            8: np.array([_mot_row(1, 20, 0, 0, 10, 10)], dtype=float),
        }
        ids = cross_camera_gt_ids(gt_by_cam)
        self.assertEqual(ids, {10})

    def test_filter_pred_multi_cam_only(self):
        pred_by_cam = {
            6: np.array([_mot_row(1, 1, 0, 0, 5, 5), _mot_row(1, 2, 10, 10, 5, 5)], dtype=float),
            7: np.array([_mot_row(1, 1, 0, 0, 5, 5)], dtype=float),
        }
        out = filter_pred_multi_cam_only(pred_by_cam)
        self.assertEqual(len(out[6]), 1)
        self.assertEqual(int(out[6][0, 1]), 1)
        self.assertEqual(len(out[7]), 1)

    def test_filter_pred_tracks_without_gt_overlap_removed(self):
        gt = np.array([_mot_row(1, 100, 0, 0, 10, 10)], dtype=float)
        pred = np.array(
            [
                _mot_row(1, 5, 0, 0, 10, 10),
                _mot_row(1, 6, 100, 100, 5, 5),
            ],
            dtype=float,
        )
        out = filter_pred_tracks_with_benchmark_gt(gt, pred, {100}, iou_thresh=0.5)
        self.assertEqual(len(out), 1)
        self.assertEqual(int(out[0, 1]), 5)

    def test_apply_cityflow_filters_local_mode(self):
        gt_by_cam = {
            6: np.array([_mot_row(1, 10, 0, 0, 10, 10)], dtype=float),
            7: np.array([_mot_row(1, 10, 0, 0, 10, 10)], dtype=float),
        }
        pred_by_cam = {
            6: np.array([_mot_row(1, 1, 0, 0, 10, 10), _mot_row(1, 2, 50, 50, 5, 5)], dtype=float),
            7: np.array([_mot_row(1, 3, 0, 0, 10, 10)], dtype=float),
        }
        out = apply_cityflow_filters(gt_by_cam, pred_by_cam, mode="local", iou_thresh=0.5)
        self.assertEqual(len(out[6]), 1)
        self.assertEqual(int(out[6][0, 1]), 1)
        self.assertEqual(len(out[7]), 1)


if __name__ == "__main__":
    unittest.main()
