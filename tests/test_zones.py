import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.io.zones import ZoneMap
from core.mot.local_zone_state import LocalZoneTracker
from core.mot.association.cross_camera import (
    CrossCameraAssociationConfig,
    passes_hard_gates,
    passes_zone_transition,
)
from core.mot.global_track import CamObservation, GlobalTrack, GlobalTrackStore


class ZoneMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zone_map = ZoneMap.from_yaml(_ROOT / "config" / "s02_zone_polygons.yaml")

    def test_loads_all_cameras(self):
        self.assertEqual(set(self.zone_map.zones.keys()), {6, 7, 8, 9})

    def test_transition_graph(self):
        self.assertEqual(set(self.zone_map.transitions[2]), {4, 6, 8})
        self.assertEqual(self.zone_map.transitions[1], frozenset())
        self.assertTrue(self.zone_map.allows_transition(3, 1))
        self.assertFalse(self.zone_map.allows_transition(3, 2))
        self.assertTrue(self.zone_map.allows_transition(None, 4))
        self.assertTrue(self.zone_map.allows_transition(2, None))

    def test_same_zone_cross_cam_allowed_even_for_terminal(self):
        # Terminal zones (empty outgoing list) still allow cross-camera handoff
        # when both sides resolve to the same global zone id.
        self.assertEqual(self.zone_map.transitions[4], frozenset())
        self.assertTrue(self.zone_map.allows_transition(4, 4))
        self.assertTrue(self.zone_map.allows_transition(6, 6))

    def test_geometry_soft_zones(self):
        self.assertEqual(self.zone_map.geometry_soft_zones, frozenset({1, 2}))
        self.assertEqual(
            self.zone_map.geometry_max_distance_m(9.0, query_zone=1, active_zones=set()),
            15.0,
        )
        self.assertEqual(
            self.zone_map.geometry_max_distance_m(9.0, query_zone=4, active_zones={2}),
            15.0,
        )
        self.assertEqual(
            self.zone_map.geometry_max_distance_m(9.0, query_zone=4, active_zones={7}),
            9.0,
        )

    def test_zone_at_bbox_bottom_center(self):
        # Point inside c007 Z1 polygon (near image center of zone).
        zone = self.zone_map.zone_at_bbox(7, (900.0, 250.0, 980.0, 360.0))
        self.assertIn(zone, {1, 2})

    def test_resolve_stable_zone_requires_full_window(self):
        self.assertEqual(self.zone_map.resolve_stable_zone([2, 2, 2, 2, 2]), 2)
        self.assertIsNone(self.zone_map.resolve_stable_zone([2, 2, 2, 2]))
        self.assertIsNone(self.zone_map.resolve_stable_zone([2, 2, 2, 2, 4]))
        self.assertIsNone(self.zone_map.resolve_stable_zone([2, 2, None, 2, 2]))


class ZoneTransitionGateTests(unittest.TestCase):
    _poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)

    def test_handoff_requires_allowed_transition(self):
        cfg = CrossCameraAssociationConfig(zone_transition=True)
        zone_map = ZoneMap(
            zones={7: {2: self._poly}, 8: {4: self._poly}},
            transitions={2: frozenset({4})},
        )
        gmeta = type(
            "G",
            (),
            {
                "active_cameras": set(),
                "last_seen_cam": 7,
                "last_seen_zone": 2,
            },
        )()
        bbox = (2.0, 2.0, 8.0, 8.0)
        self.assertTrue(
            passes_zone_transition(cfg, 8, bbox, gmeta, zone_map)
        )
        zone_map_bad = ZoneMap(
            zones={7: {2: self._poly}, 8: {4: self._poly}},
            transitions={2: frozenset({8})},
        )
        self.assertFalse(
            passes_zone_transition(cfg, 8, bbox, gmeta, zone_map_bad)
        )

    def test_overlap_skips_zone_gate(self):
        cfg = CrossCameraAssociationConfig(zone_transition=True)
        zone_map = ZoneMap(
            zones={8: {4: np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)}},
            transitions={2: frozenset()},
        )
        gmeta = type(
            "G",
            (),
            {
                "active_cameras": {7},
                "last_seen_cam": 7,
                "last_seen_zone": 2,
            },
        )()
        self.assertTrue(
            passes_zone_transition(cfg, 8, (2.0, 2.0, 8.0, 8.0), gmeta, zone_map)
        )

    def test_tracklet_mode_uses_exit_and_entry(self):
        cfg = CrossCameraAssociationConfig(zone_transition=True)
        zone_map = ZoneMap(
            zones={},
            transitions={2: frozenset({4, 6, 8})},
            mode="tracklet",
        )
        gmeta = type(
            "G",
            (),
            {"active_cameras": set(), "zone_exit": 2, "zone_entry": 2},
        )()
        self.assertTrue(
            passes_zone_transition(
                cfg, 8, None, gmeta, zone_map, query_zone_entry=6
            )
        )
        self.assertFalse(
            passes_zone_transition(
                cfg, 8, None, gmeta, zone_map, query_zone_entry=7
            )
        )
        self.assertTrue(
            passes_zone_transition(
                cfg, 8, None, gmeta, zone_map, query_zone_entry=None
            )
        )

    def test_tracklet_same_zone_terminal_allowed(self):
        cfg = CrossCameraAssociationConfig(zone_transition=True)
        zone_map = ZoneMap(
            zones={},
            transitions={4: frozenset()},
            mode="tracklet",
        )
        gmeta = type(
            "G",
            (),
            {"active_cameras": set(), "zone_exit": None, "zone_entry": 4},
        )()
        self.assertTrue(
            passes_zone_transition(
                cfg, 9, None, gmeta, zone_map, query_zone_entry=4
            )
        )


class LocalZoneTrackerTests(unittest.TestCase):
    def test_stable_entry_and_exit(self):
        poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
        poly2 = np.array([[20, 0], [30, 0], [30, 10], [20, 10]], dtype=np.float64)
        zone_map = ZoneMap(
            zones={7: {2: poly, 4: poly2}},
            transitions={2: frozenset({4})},
            mode="tracklet",
            stabilize_frames=5,
        )
        tracker = LocalZoneTracker(zone_map)
        key = (0, 10)
        for _ in range(4):
            tracker.update(key, 7, (2, 2, 8, 8))
        self.assertIsNone(tracker.entry_zone(key))
        tracker.update(key, 7, (2, 2, 8, 8))
        self.assertEqual(tracker.entry_zone(key), 2)
        for _ in range(5):
            tracker.update(key, 7, (22, 2, 28, 8))
        self.assertEqual(tracker.effective_out(key), 4)


class GlobalTrackZoneRefreshTests(unittest.TestCase):
    def test_refresh_active_sets_last_seen_zone(self):
        zone_map = ZoneMap(
            zones={
                0: {1: np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=np.float64)},
                1: {4: np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=np.float64)},
            },
            transitions={1: frozenset({4})},
        )
        tracks = {}
        store = GlobalTrackStore(tracks)
        store.create(
            1,
            10,
            0,
            CamObservation(local_tid=1, bbox=(2.0, 2.0, 8.0, 18.0), wpt=(42.492, -90.723)),
        )
        local_to_global = {(0, 1): 1, (1, 2): 1}
        per_cam_tracks = [None, np.array([[2, 2, 8, 18, 2, 0.9, -1, 1]], dtype=np.float32)]
        store.refresh_active(local_to_global, per_cam_tracks, zone_map=zone_map)
        self.assertEqual(tracks[1].last_seen_zone, 1)
        self.assertEqual(tracks[1].active_zones, {4})

    def test_hard_gate_blocks_disallowed_handoff(self):
        cfg = CrossCameraAssociationConfig(zone_transition=True)
        zone_map = ZoneMap(
            zones={
                0: {1: np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=np.float64)},
                1: {4: np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=np.float64)},
            },
            transitions={1: frozenset({4})},
        )
        gmeta = GlobalTrack(global_id=1, start_frame=10)
        gmeta.active_cameras = set()
        gmeta.last_seen_zone = 1
        bbox = (2.0, 2.0, 8.0, 18.0)
        blocked = ZoneMap(
            zones=zone_map.zones,
            transitions={1: frozenset({8})},
        )
        self.assertFalse(
            passes_hard_gates(
                cfg,
                1,
                (42.492, -90.723),
                gmeta,
                20,
                query_bbox=bbox,
                zone_map=blocked,
            )
        )
        self.assertTrue(
            passes_hard_gates(
                cfg,
                1,
                (42.492, -90.723),
                gmeta,
                20,
                query_bbox=bbox,
                zone_map=zone_map,
            )
        )


if __name__ == "__main__":
    unittest.main()
