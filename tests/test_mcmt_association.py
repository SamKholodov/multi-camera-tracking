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
CamObservation = _pipeline.CamObservation
GlobalTrackStore = _pipeline.GlobalTrackStore
CrossCameraAssociationConfig = _pipeline.CrossCameraAssociationConfig
association_cost_for_match = _pipeline.association_cost_for_match
geometry_penalty = _pipeline.geometry_penalty
passes_gates = _pipeline.passes_gates
passes_hard_gates = _pipeline.passes_hard_gates
temporal_penalty = _pipeline.temporal_penalty

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
    p.assoc_cfg = overrides.get(
        "assoc_cfg",
        CrossCameraAssociationConfig(
            geometry_max_distance_m=overrides.get("geometry_max_distance_m", 25.0),
            max_cross_cam_gap_frames=p.max_cross_cam_gap_frames,
            reid_cost_threshold=0.25,
        ),
    )
    p.geometry_max_distance = p.assoc_cfg.geometry_max_distance_m
    p.association_cost_threshold = p.assoc_cfg.reid_cost_threshold
    p.association_reid_weight = 0.5
    p.appearance_update = "aaf"
    p.local_to_global = overrides.get("local_to_global", {})
    p.global_tracks = overrides.get("global_tracks", {})
    p._global_track_store = GlobalTrackStore(p.global_tracks)
    return p


def _global_track(gid: int, entries, *, active_cameras=None):
    tracks = {}
    store = GlobalTrackStore(tracks)
    first = entries[0]
    store.create(
        gid,
        first["frame"],
        first["cam_id"],
        CamObservation(
            local_tid=first.get("local_tid", first["cam_id"]),
            wpt=first.get("wpt"),
            reid_raw=first.get("reid_raw"),
        ),
    )
    for entry in entries[1:]:
        store.append_observation(
            gid,
            entry["frame"],
            entry["cam_id"],
            CamObservation(
                local_tid=entry.get("local_tid", entry["cam_id"]),
                wpt=entry.get("wpt"),
                reid_raw=entry.get("reid_raw"),
            ),
        )
    track = tracks[gid]
    track.active_cameras = set(active_cameras or set())
    return track


class TestMcmtAssociation(unittest.TestCase):
    def test_refresh_active_cameras(self):
        p = _bare_pipeline(
            local_to_global={(0, 1): 10, (1, 2): 10, (2, 3): 20},
            global_tracks={
                10: _global_track(
                    10,
                    [
                        {"frame": 99, "cam_id": 0, "local_tid": 1, "wpt": (0.0, 0.0)},
                        {"frame": 98, "cam_id": 1, "local_tid": 2, "wpt": (5.0, 5.0)},
                    ],
                    active_cameras={0, 1},
                ),
                20: _global_track(
                    20,
                    [{"frame": 90, "cam_id": 2, "local_tid": 3, "wpt": None}],
                ),
            },
        )
        per_cam = [
            np.array([[0, 0, 1, 1, 1, 0.9, 0, 1]], dtype=np.float32),
            np.empty((0, 10), dtype=np.float32),
            np.empty((0, 10), dtype=np.float32),
        ]
        p._refresh_active_cameras(per_cam)
        self.assertEqual(p.global_tracks[10].active_cameras, {0})
        self.assertEqual(p.global_tracks[10].last_seen_cam, 1)
        self.assertEqual(p.global_tracks[10].last_seen_world, (5.0, 5.0))

    def test_geometry_min_active_other_cam_haversine(self):
        p = _bare_pipeline(geometry_max_distance_m=8.0)
        gmeta = _global_track(
            10,
            [
                {"frame": 100, "cam_id": 0, "wpt": _gps_offset_m(0.0)},
                {"frame": 100, "cam_id": 1, "wpt": _gps_offset_m(10.0)},
            ],
            active_cameras={0, 1},
        )
        cost = p._geometry_cost_for_match(0, _gps_offset_m(9.0), gmeta)
        self.assertIsNotNone(cost)
        self.assertAlmostEqual(cost, 1.0 / 8.0, places=2)

    def test_geometry_gate_applies_only_to_overlap(self):
        p = _bare_pipeline(
            frame_idx=100,
            assoc_cfg=CrossCameraAssociationConfig(
                gate_mode="hard",
                geometry_max_distance_m=8.0,
                max_cross_cam_gap_frames=30,
            ),
        )
        overlap = _global_track(
            10,
            [{"frame": 100, "cam_id": 1, "wpt": _gps_offset_m(10.0)}],
            active_cameras={1},
        )
        handoff = _global_track(
            20,
            [{"frame": 95, "cam_id": 1, "wpt": _gps_offset_m(10.0)}],
            active_cameras=set(),
        )
        handoff.last_seen_cam = 1
        handoff.last_seen_world = _gps_offset_m(10.0)

        self.assertFalse(passes_gates(p.assoc_cfg, 0, _gps_offset_m(30.0), overlap, 100))
        self.assertTrue(passes_gates(p.assoc_cfg, 0, _gps_offset_m(30.0), handoff, 100))

    def test_geometry_gate_can_be_disabled_for_ablation(self):
        cfg = CrossCameraAssociationConfig(gate_mode="hard", geometry_overlap=False)
        gmeta = _global_track(
            10,
            [{"frame": 100, "cam_id": 1, "wpt": _gps_offset_m(10.0)}],
            active_cameras={1},
        )
        self.assertTrue(passes_gates(cfg, 0, _gps_offset_m(30.0), gmeta, 100))

    def test_temporal_gate_rejects_stale_handoff(self):
        cfg = CrossCameraAssociationConfig(gate_mode="hard", max_cross_cam_gap_frames=3)
        gmeta = _global_track(
            10,
            [{"frame": 95, "cam_id": 1, "wpt": _gps_offset_m(10.0)}],
            active_cameras=set(),
        )
        gmeta.last_seen_cam = 1
        self.assertFalse(passes_gates(cfg, 0, _gps_offset_m(10.0), gmeta, 100))

    def test_geometry_penalty_close_far_and_missing(self):
        cfg = CrossCameraAssociationConfig(
            geometry_max_distance_m=8.0,
            geometry_far_penalty=10.0,
            geometry_missing_penalty=7.0,
        )
        gmeta = _global_track(
            10,
            [{"frame": 100, "cam_id": 1, "wpt": _gps_offset_m(10.0)}],
            active_cameras={1},
        )
        self.assertEqual(geometry_penalty(cfg, _gps_offset_m(9.0), gmeta, 0), 1.0)
        self.assertEqual(geometry_penalty(cfg, _gps_offset_m(30.0), gmeta, 0), 10.0)
        self.assertEqual(geometry_penalty(cfg, None, gmeta, 0), 7.0)

    def test_temporal_penalty_stale_handoff_is_soft_in_mixed(self):
        cfg = CrossCameraAssociationConfig(max_cross_cam_gap_frames=3, temporal_far_penalty=9.0)
        gmeta = _global_track(
            10,
            [{"frame": 95, "cam_id": 1, "wpt": _gps_offset_m(10.0)}],
            active_cameras=set(),
        )
        gmeta.last_seen_cam = 1
        self.assertTrue(passes_hard_gates(cfg, 0, _gps_offset_m(10.0), gmeta, 100))
        self.assertEqual(temporal_penalty(cfg, gmeta, 100, 0), 9.0)

    def test_hard_mode_still_rejects_geometry_in_association_cost(self):
        cfg = CrossCameraAssociationConfig(
            gate_mode="hard",
            geometry_max_distance_m=8.0,
        )
        query = np.array([1.0, 0.0], dtype=np.float32)
        gmeta = _global_track(
            10,
            [
                {
                    "frame": 100,
                    "cam_id": 1,
                    "wpt": _gps_offset_m(10.0),
                    "reid_raw": query,
                }
            ],
            active_cameras={1},
        )
        cost = association_cost_for_match(
            cfg,
            0,
            _gps_offset_m(30.0),
            query,
            gmeta,
            100,
            appearance_update="aaf",
        )
        self.assertIsNone(cost)

    def test_association_cost_mixed_applies_geometry_penalty(self):
        cfg = CrossCameraAssociationConfig(
            gate_mode="mixed",
            geometry_max_distance_m=8.0,
            geometry_far_penalty=10.0,
        )
        query = np.array([1.0, 0.0], dtype=np.float32)
        # Cosine distance from [1, 0] to [0.9, sqrt(1 - 0.9^2)] is 0.1.
        stored = np.array([0.9, np.sqrt(1.0 - 0.9**2)], dtype=np.float32)
        gmeta = _global_track(
            10,
            [
                {
                    "frame": 100,
                    "cam_id": 1,
                    "wpt": _gps_offset_m(10.0),
                    "reid_raw": stored,
                }
            ],
            active_cameras={1},
        )
        cost = association_cost_for_match(
            cfg,
            0,
            _gps_offset_m(30.0),
            query,
            gmeta,
            100,
            appearance_update="aaf",
        )
        self.assertIsNotNone(cost)
        self.assertAlmostEqual(cost, 1.0, places=5)

    def test_hard_gates_exclude_same_cam_in_mixed(self):
        cfg = CrossCameraAssociationConfig(gate_mode="mixed")
        gmeta = _global_track(
            10,
            [{"frame": 100, "cam_id": 0, "wpt": _gps_offset_m(0.0)}],
            active_cameras={0},
        )
        self.assertFalse(passes_hard_gates(cfg, 0, _gps_offset_m(0.0), gmeta, 100))

    def test_candidate_globals_skips_active_on_query_cam(self):
        p = _bare_pipeline(
            global_tracks={
                1: _global_track(
                    1,
                    [{"frame": 100, "cam_id": 0, "wpt": (0.0, 0.0)}],
                    active_cameras={0},
                ),
                2: _global_track(
                    2,
                    [{"frame": 100, "cam_id": 1, "wpt": (0.0, 0.0)}],
                    active_cameras={1},
                ),
            }
        )
        cands = p._candidate_globals(0)
        self.assertEqual(cands, [2])

    def test_valid_candidates_stub(self):
        p = _bare_pipeline()
        self.assertEqual(p._valid_candidates(0, [1, 2, 3]), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
