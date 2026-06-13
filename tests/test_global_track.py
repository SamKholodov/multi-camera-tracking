import unittest
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.mot.global_track import CamObservation, GlobalTrackStore


def _obs(
    local_tid: int,
    *,
    bbox=(0.0, 1.0, 2.0, 3.0),
    wpt=(10.0, 20.0),
    reid_raw=None,
    conf=0.9,
    has_detection=1,
) -> CamObservation:
    if reid_raw is None:
        reid_raw = np.ones(4, dtype=np.float32) * local_tid
    return CamObservation(
        local_tid=local_tid,
        bbox=bbox,
        wpt=wpt,
        reid_raw=reid_raw,
        conf=conf,
        has_detection=has_detection,
    )


class TestGlobalTrack(unittest.TestCase):
    def test_create_append_and_merge_two_cameras_on_same_frame(self):
        tracks = {}
        store = GlobalTrackStore(tracks)

        track = store.create(1, 10, 0, _obs(5, wpt=(1.0, 2.0)))
        store.append_observation(1, 10, 1, _obs(7, wpt=(3.0, 4.0)))

        self.assertIs(track, tracks[1])
        self.assertEqual(track.frames, [10])
        self.assertEqual(set(track.cam_observations[0]), {0, 1})
        self.assertEqual(track.has_detection, [1])
        self.assertEqual(track.cam_world, {0: (1.0, 2.0), 1: (3.0, 4.0)})

    def test_placeholder_does_not_move_last_frame(self):
        tracks = {}
        store = GlobalTrackStore(tracks)
        track = store.create(1, 10, 0, _obs(5))

        store.append_missed(1, 11)

        self.assertEqual(track.frames, [10, 11])
        self.assertEqual(track.cam_observations[1], {})
        self.assertEqual(track.has_detection, [1, 0])
        self.assertEqual(track.last_frame, 10)

    def test_properties_use_latest_observations(self):
        tracks = {}
        store = GlobalTrackStore(tracks)
        store.create(1, 10, 0, _obs(5, wpt=(1.0, 2.0), reid_raw=np.array([1, 0])))
        store.append_observation(
            1,
            12,
            0,
            _obs(5, wpt=(5.0, 6.0), reid_raw=np.array([2, 0])),
        )
        store.append_observation(
            1,
            12,
            1,
            _obs(8, wpt=(7.0, 8.0), reid_raw=np.array([0, 3])),
        )

        track = tracks[1]
        self.assertEqual(track.cam_world, {0: (5.0, 6.0), 1: (7.0, 8.0)})
        self.assertEqual(track.cam_last_frame, {0: 12, 1: 12})
        self.assertTrue(np.array_equal(track.local_appearance[(0, 5)], np.array([2, 0])))
        self.assertTrue(np.array_equal(track.local_appearance[(1, 8)], np.array([0, 3])))

    def test_lifecycle_lost_then_deleted(self):
        tracks = {}
        store = GlobalTrackStore(tracks)
        track = store.create(1, 10, 0, _obs(5))

        store.manage_states(14, lost_after=3, delete_after=300)
        self.assertEqual(track.state, "lost")
        self.assertIn(1, tracks)

        store.manage_states(311, lost_after=3, delete_after=300)
        deleted = store.prune_deleted()
        self.assertEqual(deleted, [1])
        self.assertNotIn(1, tracks)

    def test_refresh_active_updates_last_seen(self):
        tracks = {}
        store = GlobalTrackStore(tracks)
        track = store.create(10, 99, 0, _obs(1, wpt=(0.0, 0.0)))
        store.append_observation(10, 98, 1, _obs(2, wpt=(5.0, 5.0)))
        track.active_cameras = {0, 1}

        per_cam = [
            np.array([[0, 0, 1, 1, 1, 0.9, 0, 1]], dtype=np.float32),
            np.empty((0, 10), dtype=np.float32),
        ]
        store.refresh_active({(0, 1): 10, (1, 2): 10}, per_cam)

        self.assertEqual(track.active_cameras, {0})
        self.assertEqual(track.last_seen_cam, 1)
        self.assertEqual(track.last_seen_world, (5.0, 5.0))


if __name__ == "__main__":
    unittest.main()
