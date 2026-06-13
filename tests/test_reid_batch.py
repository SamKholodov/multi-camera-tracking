import sys
import unittest
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.mot.reid_batch import batch_reid_features


class _FakeReidBackend:
    def __init__(self):
        self.forward_calls: list[int] = []

    def get_crops(self, boxes, frame):
        n = len(boxes)
        return torch.zeros((n, 3, 8, 8), dtype=torch.float32)

    def inference_preprocess(self, crops):
        return crops

    def forward(self, im_batch):
        self.forward_calls.append(int(im_batch.shape[0]))
        return torch.ones((im_batch.shape[0], 4), dtype=torch.float32)

    def inference_postprocess(self, features):
        return features


class TestReidBatch(unittest.TestCase):
    def test_chunks_large_multicam_batch(self):
        backend = _FakeReidBackend()
        frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(2)]
        detections = [
            np.array([[0, 0, 1, 1, 0.9, 2, 0]], dtype=np.float32),
            np.array(
                [[0, 0, 1, 1, 0.9, 2, 0]] * 5,
                dtype=np.float32,
            ),
        ]

        out = batch_reid_features(
            backend,
            frames,
            detections,
            max_batch_size=3,
        )

        self.assertEqual(backend.forward_calls, [3, 3])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (1, 4))
        self.assertEqual(out[1].shape, (5, 4))

    def test_skips_low_confidence_for_inference(self):
        backend = _FakeReidBackend()
        frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
        detections = [
            np.array(
                [
                    [0, 0, 1, 1, 0.9, 2, 0],
                    [1, 1, 2, 2, 0.4, 2, 0],
                    [2, 2, 3, 3, 0.2, 2, 0],
                ],
                dtype=np.float32,
            )
        ]

        out = batch_reid_features(
            backend,
            frames,
            detections,
            max_batch_size=8,
            conf_thresh=0.6,
        )

        self.assertEqual(backend.forward_calls, [1])
        self.assertEqual(out[0].shape, (3, 4))
        self.assertGreater(np.linalg.norm(out[0][0]), 0.99)
        self.assertTrue(np.allclose(out[0][1:], 0.0))


if __name__ == "__main__":
    unittest.main()
