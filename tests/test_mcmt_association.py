import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.mot.association.cross_camera import (
    CrossCameraAssociationConfig,
    association_cost_for_match,
    geometry_cost_adjustment,
    passes_temporal_gate,
    passes_hard_gates,
    temporal_cost_adjustment,
    _uses_geometry_tiers,
)
from core.mot.association.kinematic import speed_cost_adjustment
from core.mot.association.trajectory import trajectory_cost_adjustment
from core.mot.global_track import CamObservation, GlobalTrackStore


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


def _candidate_globals(global_tracks, frame_idx, max_gap):
    out = []
    for gid, meta in global_tracks.items():
        if meta.state == "lost":
            continue
        last_f = meta.last_frame
        if last_f is None:
            continue
        if frame_idx - last_f > max_gap:
            continue
        out.append(gid)
    return out


class TestMcmtAssociation(unittest.TestCase):
    def test_geometry_tier_close_mid_and_reject(self):
        cfg = CrossCameraAssociationConfig(
            geometry_t_min_m=14.0,
            geometry_t_distant_m=38.0,
            geometry_mid_penalty=0.15,
            geometry_distance_metric="plane",
        )
        gmeta = _global_track(
            10,
            [{"frame": 100, "cam_id": 1, "wpt": (10.0, 0.0)}],
            active_cameras={1},
        )
        self.assertEqual(
            geometry_cost_adjustment(cfg, (10.0, 0.0), gmeta, 0), 0.0
        )
        self.assertEqual(
            geometry_cost_adjustment(cfg, (30.0, 0.0), gmeta, 0), 0.15
        )
        self.assertIsNone(
            geometry_cost_adjustment(cfg, (60.0, 0.0), gmeta, 0)
        )

    def test_association_cost_tiered_rejects_distant(self):
        cfg = CrossCameraAssociationConfig(
            geometry_t_min_m=14.0,
            geometry_t_distant_m=38.0,
        )
        query = np.array([1.0, 0.0], dtype=np.float32)
        gmeta = _global_track(
            10,
            [
                {
                    "frame": 100,
                    "cam_id": 1,
                    "wpt": (10.0, 0.0),
                    "reid_raw": query,
                }
            ],
            active_cameras={1},
        )
        cost = association_cost_for_match(
            cfg,
            0,
            (60.0, 0.0),
            query,
            gmeta,
            100,
            appearance_update="aaf",
        )
        self.assertIsNone(cost)

    def test_association_cost_tiered_adds_mid_penalty(self):
        cfg = CrossCameraAssociationConfig(
            geometry_t_min_m=14.0,
            geometry_t_distant_m=38.0,
            geometry_mid_penalty=0.15,
        )
        query = np.array([1.0, 0.0], dtype=np.float32)
        stored = np.array([0.9, np.sqrt(1.0 - 0.9**2)], dtype=np.float32)
        gmeta = _global_track(
            10,
            [
                {
                    "frame": 100,
                    "cam_id": 1,
                    "wpt": (10.0, 0.0),
                    "reid_raw": stored,
                }
            ],
            active_cameras={1},
        )
        cost = association_cost_for_match(
            cfg,
            0,
            (30.0, 0.0),
            query,
            gmeta,
            100,
            appearance_update="aaf",
        )
        self.assertIsNotNone(cost)
        self.assertAlmostEqual(cost, 0.1 + 0.15, places=5)

    def test_hard_gates_allow_without_legacy_geometry_gate(self):
        cfg = CrossCameraAssociationConfig()
        gmeta = _global_track(
            10,
            [{"frame": 100, "cam_id": 1, "wpt": (10.0, 0.0)}],
            active_cameras={1},
        )
        self.assertTrue(
            passes_hard_gates(cfg, 0, (30.0, 0.0), gmeta, 100)
        )

    def test_candidate_globals_skips_lost_and_stale(self):
        global_tracks = {
            1: _global_track(
                1,
                [{"frame": 100, "cam_id": 0, "wpt": (0.0, 0.0)}],
                active_cameras={0},
            ),
            2: _global_track(
                2,
                [{"frame": 90, "cam_id": 1, "wpt": (0.0, 0.0)}],
                active_cameras={1},
            ),
        }
        cands = _candidate_globals(global_tracks, 100, 5)
        self.assertEqual(cands, [1])

    def test_candidate_globals_allows_same_cam(self):
        global_tracks = {
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
        cands = _candidate_globals(global_tracks, 100, 300)
        self.assertEqual(cands, [1, 2])

    def test_config_defaults_match_phase0(self):
        cfg = CrossCameraAssociationConfig()
        self.assertEqual(cfg.geometry_t_min_m, 14.0)
        self.assertEqual(cfg.geometry_t_distant_m, 38.0)
        self.assertEqual(cfg.geometry_distance_metric, "plane")
        self.assertEqual(cfg.reid_strong_reject_threshold, 0.50)
        self.assertEqual(cfg.temporal_mode, "off")
        self.assertEqual(cfg.global_delete_after_frames, 600)

    def test_from_yaml_legacy_geometry_max_distance_disables_tiers(self):
        cfg = CrossCameraAssociationConfig.from_yaml(
            {
                "association_cost_threshold": 0.25,
                "geometry_max_distance": 0.0,
                "max_cross_cam_gap_frames": 100,
            }
        )
        self.assertEqual(cfg.geometry_t_distant_m, 0.0)
        self.assertEqual(cfg.reid_cost_threshold, 0.25)
        self.assertFalse(_uses_geometry_tiers(cfg))

    def test_from_yaml_legacy_geometry_max_distance_maps_tiers(self):
        cfg = CrossCameraAssociationConfig.from_yaml(
            {"geometry_max_distance": 8.0}
        )
        self.assertEqual(cfg.geometry_t_distant_m, 8.0)
        self.assertAlmostEqual(cfg.geometry_t_min_m, 3.0)

    def test_temporal_strict_gate_rejects_stale_handoff(self):
        cfg = CrossCameraAssociationConfig(
            temporal_mode="strict",
            max_cross_cam_gap_frames=30,
        )
        gmeta = _global_track(
            10,
            [{"frame": 100, "cam_id": 1, "wpt": (0.0, 0.0)}],
            active_cameras=set(),
        )
        gmeta.last_seen_world = (0.0, 0.0)
        self.assertFalse(passes_temporal_gate(cfg, gmeta, 131, 0))
        self.assertTrue(passes_temporal_gate(cfg, gmeta, 130, 0))

    def test_temporal_penalty_only_adds_mid_penalty(self):
        cfg = CrossCameraAssociationConfig(
            temporal_mode="penalty_only",
            max_cross_cam_gap_frames=30,
            temporal_mid_penalty=0.2,
        )
        gmeta = _global_track(
            10,
            [{"frame": 100, "cam_id": 1, "wpt": (0.0, 0.0)}],
            active_cameras=set(),
        )
        gmeta.last_seen_world = (0.0, 0.0)
        self.assertEqual(temporal_cost_adjustment(cfg, gmeta, 130, 0), 0.0)
        self.assertEqual(temporal_cost_adjustment(cfg, gmeta, 131, 0), 0.2)

    def test_speed_hard_rejects_impossible_handoff(self):
        cfg = CrossCameraAssociationConfig(
            speed_limit_enabled=True,
            speed_limit_mode="hard",
            speed_v_max_mps=25.0,
            speed_margin=0.2,
            video_fps=10.0,
        )
        gmeta = _global_track(
            10,
            [{"frame": 100, "cam_id": 1, "wpt": (0.0, 0.0)}],
            active_cameras=set(),
        )
        gmeta.last_seen_world = (0.0, 0.0)
        self.assertIsNone(speed_cost_adjustment(cfg, (40.0, 0.0), gmeta, 110, 0))

    def test_speed_penalty_scales_with_ratio(self):
        cfg = CrossCameraAssociationConfig(
            speed_limit_enabled=True,
            speed_limit_mode="penalty",
            speed_v_max_mps=25.0,
            speed_penalty_scale=0.5,
            video_fps=10.0,
        )
        gmeta = _global_track(
            10,
            [{"frame": 100, "cam_id": 1, "wpt": (0.0, 0.0)}],
            active_cameras=set(),
        )
        gmeta.last_seen_world = (0.0, 0.0)
        self.assertAlmostEqual(speed_cost_adjustment(cfg, (50.0, 0.0), gmeta, 110, 0), 0.5)

    def test_trajectory_linear_adds_soft_penalty(self):
        cfg = CrossCameraAssociationConfig(
            trajectory_enabled=True,
            trajectory_mode="linear",
            trajectory_history_k=3,
            trajectory_threshold_m=2.0,
            trajectory_penalty_scale=0.5,
        )
        gmeta = _global_track(
            10,
            [
                {"frame": 100, "cam_id": 1, "wpt": (0.0, 0.0)},
                {"frame": 101, "cam_id": 1, "wpt": (1.0, 0.0)},
            ],
            active_cameras=set(),
        )
        # Prediction at frame 102 is (2, 0); query at (6, 0) has error 4m.
        self.assertAlmostEqual(
            trajectory_cost_adjustment(cfg, (6.0, 0.0), gmeta, 102),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
