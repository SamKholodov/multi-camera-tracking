import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

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

# S02 reference GPS; offsets in meters via ~111.32 m/deg latitude.
_REF_LAT = 42.492
_REF_LON = -90.723
_M_PER_DEG_LAT = 111_320.0


def _gps_offset_m(north_m: float, east_m: float = 0.0) -> tuple[float, float]:
    lat = _REF_LAT + north_m / _M_PER_DEG_LAT
    lon = _REF_LON + east_m / (_M_PER_DEG_LAT * 0.737)  # cos(42.49°) ≈ 0.737
    return lat, lon


def _bare_pipeline(**overrides):
    p = object.__new__(MultiCameraTrackingPipeline)
    p.frame_idx = overrides.get("frame_idx", 100)
    p.max_cross_cam_gap_frames = overrides.get("max_cross_cam_gap_frames", 30)
    p.geometry_max_distance = 25.0
    p.association_reid_weight = 0.5
    p.appearance_update = "aaf"
    p.local_to_global = overrides.get("local_to_global", {})
    p.global_tracks = overrides.get("global_tracks", {})
    return p


class TestMcmtAssociation(unittest.TestCase):
    def test_refresh_active_cameras(self):
        p = _bare_pipeline(
            local_to_global={(0, 1): 10, (1, 2): 10, (2, 3): 20},
            global_tracks={
                10: {
                    "active_cameras": {0, 1},
                    "cam_world": {0: (0.0, 0.0), 1: (5.0, 5.0)},
                    "cam_last_frame": {0: 99, 1: 98},
                    "last_seen_cam": None,
                    "last_seen_world": None,
                    "local_appearance": {},
                },
                20: {"active_cameras": set(), "cam_world": {}, "cam_last_frame": {}},
            },
        )
        per_cam = [
            np.array([[0, 0, 1, 1, 1, 0.9, 0, 1]], dtype=np.float32),
            np.empty((0, 10), dtype=np.float32),
            np.empty((0, 10), dtype=np.float32),
        ]
        p._refresh_active_cameras(per_cam)
        self.assertEqual(p.global_tracks[10]["active_cameras"], {0})
        self.assertEqual(p.global_tracks[10]["last_seen_cam"], 1)
        self.assertEqual(p.global_tracks[10]["last_seen_world"], (5.0, 5.0))

    def test_geometry_min_active_other_cam(self):
        p = _bare_pipeline()
        gmeta = {
            "active_cameras": {0, 1},
            "cam_world": {0: _gps_offset_m(0), 1: _gps_offset_m(10)},
            "cam_last_frame": {0: 100, 1: 100},
            "last_seen_cam": None,
            "last_seen_world": None,
        }
        cost = p._geometry_cost_for_match(0, _gps_offset_m(9), gmeta)
        self.assertIsNotNone(cost)
        self.assertAlmostEqual(cost, 0.04, places=2)

    def test_geometry_fallback_last_seen(self):
        p = _bare_pipeline()
        gmeta = {
            "active_cameras": set(),
            "cam_world": {1: _gps_offset_m(10)},
            "cam_last_frame": {1: 95},
            "last_seen_cam": 1,
            "last_seen_world": _gps_offset_m(10),
        }
        cost = p._geometry_cost_for_match(0, _gps_offset_m(12), gmeta)
        self.assertIsNotNone(cost)
        self.assertLess(cost, 0.2)

    def test_candidate_globals_skips_active_on_query_cam(self):
        p = _bare_pipeline(
            global_tracks={
                1: {
                    "active_cameras": {0},
                    "cam_last_frame": {0: 100},
                },
                2: {
                    "active_cameras": {1},
                    "cam_last_frame": {1: 100},
                },
            }
        )
        cands = p._candidate_globals(0)
        self.assertEqual(cands, [2])

    def test_valid_candidates_stub(self):
        p = _bare_pipeline()
        self.assertEqual(p._valid_candidates(0, [1, 2, 3]), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
