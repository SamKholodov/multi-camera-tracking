import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_spec = importlib.util.spec_from_file_location(
    "appearance",
    _ROOT / "core" / "mot" / "appearance.py",
)
assert _spec and _spec.loader
_appearance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_appearance)

cosine_distance = _appearance.cosine_distance
global_appearance_distance = _appearance.global_appearance_distance
cross_camera_appearance_distance = _appearance.cross_camera_appearance_distance
normalize_l2 = _appearance.normalize_l2
normalized_for_matching = _appearance.normalized_for_matching
update_appearance = _appearance.update_appearance


class TestAppearance(unittest.TestCase):
    def test_aaf_first_update(self):
        f = np.array([3.0, 4.0], dtype=np.float32)
        out = update_appearance(None, f, mode="aaf")
        expected = normalize_l2(f)
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_aaf_accumulates_normalized_features(self):
        f1 = np.array([1.0, 0.0], dtype=np.float32)
        f2 = np.array([0.0, 1.0], dtype=np.float32)
        acc = update_appearance(None, f1, mode="aaf")
        acc = update_appearance(acc, f2, mode="aaf")
        np.testing.assert_allclose(acc, np.array([1.0, 1.0], dtype=np.float32))

    def test_aaf_normalized_for_matching(self):
        f1 = np.array([2.0, 0.0], dtype=np.float32)
        f2 = np.array([0.0, 2.0], dtype=np.float32)
        acc = update_appearance(None, f1, mode="aaf")
        acc = update_appearance(acc, f2, mode="aaf")
        norm = normalized_for_matching(acc, "aaf")
        np.testing.assert_allclose(norm, normalize_l2(acc))
        self.assertAlmostEqual(float(np.linalg.norm(norm)), 1.0, places=5)

    def test_ema_update_normalizes(self):
        f1 = np.array([1.0, 0.0], dtype=np.float32)
        f2 = np.array([0.0, 1.0], dtype=np.float32)
        emb = update_appearance(None, f1, mode="ema", alpha=0.9)
        emb = update_appearance(emb, f2, mode="ema", alpha=0.9)
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=5)

    def test_cosine_distance_identical(self):
        v = normalize_l2(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        self.assertAlmostEqual(cosine_distance(v, v), 0.0, places=6)

    def test_global_appearance_distance_mean(self):
        q = normalize_l2(np.array([1.0, 0.0], dtype=np.float32))
        local_feats = {
            (0, 1): normalize_l2(np.array([1.0, 0.0], dtype=np.float32)),
            (1, 2): normalize_l2(np.array([0.0, 1.0], dtype=np.float32)),
        }
        dist = global_appearance_distance(q, local_feats, mode="aaf")
        self.assertAlmostEqual(dist, 0.5, places=5)

    def test_global_appearance_distance_empty(self):
        q = normalize_l2(np.array([1.0, 0.0], dtype=np.float32))
        self.assertIsNone(global_appearance_distance(q, {}, mode="aaf"))

    def test_cross_camera_appearance_min_other_active(self):
        q = normalize_l2(np.array([1.0, 0.0], dtype=np.float32))
        same = normalize_l2(np.array([1.0, 0.0], dtype=np.float32))
        orth = normalize_l2(np.array([0.0, 1.0], dtype=np.float32))
        local_feats = {
            (6, 1): same,
            (7, 2): orth,
            (8, 3): same,
        }
        dist = cross_camera_appearance_distance(
            q, local_feats, query_cam=6, active_cameras={6, 7, 8}, last_seen_cam=None, mode="aaf"
        )
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_cross_camera_appearance_excludes_query_cam(self):
        q = normalize_l2(np.array([1.0, 0.0], dtype=np.float32))
        orth = normalize_l2(np.array([0.0, 1.0], dtype=np.float32))
        local_feats = {(6, 1): orth}
        dist = cross_camera_appearance_distance(
            q, local_feats, query_cam=6, active_cameras={6}, last_seen_cam=None, mode="aaf"
        )
        self.assertIsNone(dist)

    def test_cross_camera_appearance_fallback_last_seen(self):
        q = normalize_l2(np.array([1.0, 0.0], dtype=np.float32))
        same = normalize_l2(np.array([1.0, 0.0], dtype=np.float32))
        local_feats = {(7, 2): same}
        dist = cross_camera_appearance_distance(
            q,
            local_feats,
            query_cam=6,
            active_cameras=set(),
            last_seen_cam=7,
            mode="aaf",
            cam_last_frame={7: 95},
            frame_idx=100,
            max_gap_frames=30,
        )
        self.assertAlmostEqual(dist, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
