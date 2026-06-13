import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_spec = importlib.util.spec_from_file_location(
    "pipeline",
    _ROOT / "pipeline.py",
)
assert _spec and _spec.loader
_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline)

MultiCameraTrackingPipeline = _pipeline.MultiCameraTrackingPipeline


def _make_box_result(x1, y1, x2, y2, score=0.9, cls=2):
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32),
            conf=torch.tensor([score], dtype=torch.float32),
            cls=torch.tensor([cls], dtype=torch.float32),
        )
    )


class TestDetectorBatch(unittest.TestCase):
    def test_detect_batch_single_forward_and_index_mapping(self):
        from core.detector.detector import Detector

        frames = [
            np.zeros((64, 64, 3), dtype=np.uint8),
            None,
            np.ones((64, 64, 3), dtype=np.uint8),
            np.full((64, 64, 3), 2, dtype=np.uint8),
        ]
        mock_results = [
            _make_box_result(0, 0, 10, 10, score=0.91, cls=2),
            _make_box_result(1, 1, 11, 11, score=0.92, cls=3),
            _make_box_result(2, 2, 12, 12, score=0.93, cls=5),
        ]

        with patch.object(Detector, "__init__", lambda self, *a, **k: None):
            det = Detector()
            det.conf_thres = 0.5
            det.target_classes = [2, 3, 5, 7]
            det.imgsz = 960
            det.model = MagicMock(return_value=mock_results)

            out = det.detect_batch(frames)

        det.model.assert_called_once()
        batched_frames = det.model.call_args[0][0]
        self.assertEqual(len(batched_frames), 3)

        self.assertEqual(len(out[0][0]), 1)
        box0 = [float(v) for v in out[0][0][0]]
        self.assertEqual(box0[:4], [0.0, 0.0, 10.0, 10.0])
        self.assertAlmostEqual(box0[4], 0.91, places=5)
        self.assertEqual(box0[5], 2.0)
        self.assertEqual(out[1], ([], []))
        box2 = [float(v) for v in out[2][0][0]]
        self.assertEqual(box2[:4], [1.0, 1.0, 11.0, 11.0])
        self.assertAlmostEqual(box2[4], 0.92, places=5)
        self.assertEqual(box2[5], 3.0)
        box3 = [float(v) for v in out[3][0][0]]
        self.assertEqual(box3[:4], [2.0, 2.0, 12.0, 12.0])
        self.assertAlmostEqual(box3[4], 0.93, places=5)
        self.assertEqual(box3[5], 5.0)

    def test_detect_batch_all_none(self):
        from core.detector.detector import Detector

        with patch.object(Detector, "__init__", lambda self, *a, **k: None):
            det = Detector()
            det.model = MagicMock()
            out = det.detect_batch([None, None])
        det.model.assert_not_called()
        self.assertEqual(out, [([], []), ([], [])])


class TestSharedDetectorWiring(unittest.TestCase):
    def test_mcmt_uses_one_shared_detector(self):
        shared = MagicMock()
        shared.detect_batch.return_value = [
            ([[0, 0, 1, 1, 0.9, 2]], [2]),
            ([], []),
        ]

        p = object.__new__(MultiCameraTrackingPipeline)
        p.shared_detector = shared
        p.shared_reid_model = None
        p._last_batch_reid_ms = None
        p.frame_idx = 0
        p.per_cam_pipelines = [
            MagicMock(),
            MagicMock(),
        ]
        for cam in p.per_cam_pipelines:
            cam._filter_detections_roi.side_effect = lambda dets: dets
        p.per_cam_pipelines[0].process_frame.return_value = np.empty(
            (0, 8), dtype=np.float32
        )
        p.per_cam_pipelines[1].process_frame.return_value = np.empty(
            (0, 8), dtype=np.float32
        )

        frames = [np.zeros((8, 8, 3), dtype=np.uint8), None]
        tracks = p._step_sct(frames)

        shared.detect_batch.assert_called_once_with(frames)
        self.assertEqual(len(tracks), 2)
        p.per_cam_pipelines[0].process_frame.assert_called_once()
        p.per_cam_pipelines[1].process_frame.assert_called_once()

if __name__ == "__main__":
    unittest.main()
