from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass
class CamObservation:
    local_tid: int
    bbox: tuple[float, float, float, float] | None = None
    wpt: tuple[float, float] | None = None
    reid_raw: np.ndarray | None = None
    conf: float = 0.0
    has_detection: int = 1

    def __post_init__(self) -> None:
        self.local_tid = int(self.local_tid)
        self.conf = float(self.conf)
        self.has_detection = int(self.has_detection)
        if self.bbox is not None:
            self.bbox = tuple(float(v) for v in self.bbox)
        if self.wpt is not None:
            self.wpt = (float(self.wpt[0]), float(self.wpt[1]))
        if self.reid_raw is not None:
            self.reid_raw = np.asarray(self.reid_raw, dtype=np.float32).copy()


@dataclass
class GlobalTrack:
    global_id: int
    start_frame: int
    frames: list[int] = field(default_factory=list)
    cam_observations: list[dict[int, CamObservation]] = field(default_factory=list)
    has_detection: list[int] = field(default_factory=list)
    state: str = "pending"
    active_cameras: set[int] = field(default_factory=set)
    last_seen_cam: int | None = None
    last_seen_world: tuple[float, float] | None = None
    last_seen_zone: int | None = None
    zone_entry: int | None = None
    zone_exit: int | None = None
    active_zones: set[int] = field(default_factory=set)
    _removed_local_keys: set[tuple[int, int]] = field(default_factory=set)
    _deleted: bool = False

    def append_observation(
        self,
        frame_idx: int,
        cam_id: int,
        observation: CamObservation,
    ) -> None:
        slot = self._ensure_frame_slot(frame_idx)
        slot[int(cam_id)] = observation
        self.has_detection[-1] = int(
            any(obs.has_detection for obs in self.cam_observations[-1].values())
        )
        self._removed_local_keys.discard((int(cam_id), int(observation.local_tid)))

    def append_missed(self, frame_idx: int) -> None:
        self._ensure_frame_slot(frame_idx)

    def mark_local_key_removed(self, key: tuple[int, int]) -> None:
        self._removed_local_keys.add((int(key[0]), int(key[1])))

    @property
    def length(self) -> int:
        return len(self.frames)

    @property
    def last_frame(self) -> int | None:
        for frame_idx, detected in zip(reversed(self.frames), reversed(self.has_detection)):
            if detected:
                return frame_idx
        return None

    @property
    def local_appearance(self) -> dict[tuple[int, int], np.ndarray]:
        out: dict[tuple[int, int], np.ndarray] = {}
        seen: set[tuple[int, int]] = set()
        for slot in reversed(self.cam_observations):
            for cam_id, obs in slot.items():
                key = (int(cam_id), int(obs.local_tid))
                if key in seen or key in self._removed_local_keys:
                    continue
                seen.add(key)
                if obs.reid_raw is not None:
                    out[key] = obs.reid_raw
        return out

    @property
    def cam_world(self) -> dict[int, tuple[float, float]]:
        out: dict[int, tuple[float, float]] = {}
        for slot in reversed(self.cam_observations):
            for cam_id, obs in slot.items():
                if cam_id not in out and obs.wpt is not None and obs.has_detection:
                    out[int(cam_id)] = obs.wpt
        return out

    @property
    def cam_last_frame(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for frame_idx, slot in zip(reversed(self.frames), reversed(self.cam_observations)):
            for cam_id, obs in slot.items():
                if cam_id not in out and obs.has_detection:
                    out[int(cam_id)] = int(frame_idx)
        return out

    def last_bbox_on_cam(self, cam_id: int) -> tuple[float, float, float, float] | None:
        for slot in reversed(self.cam_observations):
            obs = slot.get(int(cam_id))
            if obs is not None and obs.has_detection and obs.bbox is not None:
                return obs.bbox
        return None

    def recent_world_observations(
        self,
        k: int,
    ) -> list[tuple[int, tuple[float, float]]]:
        out: list[tuple[int, tuple[float, float]]] = []
        for frame_idx, slot in zip(reversed(self.frames), reversed(self.cam_observations)):
            points = [obs.wpt for obs in slot.values() if obs.has_detection and obs.wpt is not None]
            if not points:
                continue
            x = sum(float(p[0]) for p in points) / len(points)
            y = sum(float(p[1]) for p in points) / len(points)
            out.append((int(frame_idx), (x, y)))
            if len(out) >= int(k):
                break
        return list(reversed(out))

    def _ensure_frame_slot(self, frame_idx: int) -> dict[int, CamObservation]:
        frame_idx = int(frame_idx)
        if self.frames and self.frames[-1] == frame_idx:
            return self.cam_observations[-1]
        self.frames.append(frame_idx)
        self.cam_observations.append({})
        self.has_detection.append(0)
        return self.cam_observations[-1]


class GlobalTrackStore:
    def __init__(self, tracks: dict[int, GlobalTrack]):
        self.tracks = tracks

    def create(
        self,
        gid: int,
        frame_idx: int,
        cam_id: int,
        obs: CamObservation,
    ) -> GlobalTrack:
        track = GlobalTrack(global_id=int(gid), start_frame=int(frame_idx))
        track.append_observation(frame_idx, cam_id, obs)
        track.active_cameras = {int(cam_id)}
        self.tracks[int(gid)] = track
        return track

    def append_observation(
        self,
        gid: int,
        frame_idx: int,
        cam_id: int,
        obs: CamObservation,
    ) -> None:
        self.tracks[int(gid)].append_observation(frame_idx, cam_id, obs)

    def append_missed(self, gid: int, frame_idx: int) -> None:
        self.tracks[int(gid)].append_missed(frame_idx)

    def refresh_active(
        self,
        local_to_global: Mapping[tuple[int, int], int],
        per_cam_tracks,
        *,
        zone_map=None,
        cam_ids=None,
        local_zone_tracker=None,
    ) -> None:
        active_local: dict[int, set[int]] = {}
        for cam_id, tracks in enumerate(per_cam_tracks):
            if tracks is None or len(tracks) == 0:
                active_local[cam_id] = set()
                continue
            active_local[cam_id] = {int(row[4]) for row in tracks}

        gid_to_cams: dict[int, set[int]] = {}
        for (cam_id, local_tid), gid in local_to_global.items():
            if gid not in self.tracks:
                continue
            if local_tid in active_local.get(cam_id, set()):
                gid_to_cams.setdefault(int(gid), set()).add(int(cam_id))

        for gid, track in self.tracks.items():
            prev_active = set(track.active_cameras)
            new_active = gid_to_cams.get(gid, set())
            dropped = prev_active - new_active
            if dropped:
                cam_last = track.cam_last_frame
                last_cam = max(dropped, key=lambda c: cam_last.get(c, -1))
                track.last_seen_cam = last_cam
                cam_world = track.cam_world
                if last_cam in cam_world:
                    track.last_seen_world = cam_world[last_cam]
                if zone_map is not None:
                    real_cam = (
                        int(cam_ids[last_cam])
                        if cam_ids is not None and last_cam < len(cam_ids)
                        else int(last_cam)
                    )
                    track.last_seen_zone = zone_map.zone_at_bbox(
                        real_cam, track.last_bbox_on_cam(last_cam)
                    )
                if local_zone_tracker is not None:
                    local_tids = [
                        local_tid
                        for (cam_id, local_tid), mapped_gid in local_to_global.items()
                        if int(cam_id) == int(last_cam) and int(mapped_gid) == int(gid)
                    ]
                    if local_tids:
                        key = (int(last_cam), int(local_tids[0]))
                        state = local_zone_tracker.get(key)
                        if state is not None:
                            track.zone_entry = state.zone_entry
                            track.zone_exit = state.effective_out
            track.active_cameras = new_active
            if zone_map is None:
                track.active_zones = set()
                continue
            active_zones: set[int] = set()
            for cam_idx in new_active:
                tracks = per_cam_tracks[cam_idx] if cam_idx < len(per_cam_tracks) else None
                if tracks is None or len(tracks) == 0:
                    continue
                for row in tracks:
                    local_tid = int(row[4])
                    if local_to_global.get((cam_idx, local_tid)) != gid:
                        continue
                    real_cam = (
                        int(cam_ids[cam_idx])
                        if cam_ids is not None and cam_idx < len(cam_ids)
                        else int(cam_idx)
                    )
                    zone_id = zone_map.zone_at_bbox(
                        real_cam, tuple(float(v) for v in row[:4])
                    )
                    if zone_id is not None:
                        active_zones.add(zone_id)
            track.active_zones = active_zones

    def manage_states(
        self,
        frame_idx: int,
        *,
        lost_after: int,
        delete_after: int,
    ) -> None:
        for track in self.tracks.values():
            last_f = track.last_frame
            gap = frame_idx - last_f if last_f is not None else frame_idx - track.start_frame
            if gap > delete_after:
                track._deleted = True
            elif gap > lost_after:
                track.state = "lost"
            elif track.length < 5:
                track.state = "pending"
            else:
                track.state = "confirmed"

    def prune_deleted(self) -> list[int]:
        deleted = [gid for gid, track in self.tracks.items() if track._deleted]
        for gid in deleted:
            self.tracks.pop(gid, None)
        return deleted

    def remove_local_appearance_key(self, gid: int, cam_id: int, local_tid: int) -> None:
        track = self.tracks.get(int(gid))
        if track is not None:
            track.mark_local_key_removed((cam_id, local_tid))

    def merge(self, gid_keep: int, gid_drop: int) -> None:
        """Merge gid_drop history into gid_keep and remove gid_drop."""
        gid_keep = int(gid_keep)
        gid_drop = int(gid_drop)
        if gid_keep == gid_drop or gid_drop not in self.tracks:
            return
        keep = self.tracks[gid_keep]
        drop = self.tracks.pop(gid_drop)
        keep.start_frame = min(keep.start_frame, drop.start_frame)
        keep._removed_local_keys |= drop._removed_local_keys
        for frame_idx, slot in zip(drop.frames, drop.cam_observations):
            for cam_id, obs in slot.items():
                keep.append_observation(frame_idx, cam_id, obs)
        if drop.last_seen_cam is not None:
            keep_last = keep.cam_last_frame.get(keep.last_seen_cam, -1) if keep.last_seen_cam is not None else -1
            drop_last = drop.cam_last_frame.get(drop.last_seen_cam, -1)
            if drop_last >= keep_last:
                keep.last_seen_cam = drop.last_seen_cam
                if drop.last_seen_world is not None:
                    keep.last_seen_world = drop.last_seen_world
                if drop.last_seen_zone is not None:
                    keep.last_seen_zone = drop.last_seen_zone
        keep.active_cameras |= drop.active_cameras
        keep.active_zones |= drop.active_zones

    def finalize_frame(
        self,
        frame_idx: int,
        per_cam_tracks,
        local_to_global: Mapping[tuple[int, int], int],
        *,
        lost_after: int,
        delete_after: int,
        zone_map=None,
        cam_ids=None,
        local_zone_tracker=None,
    ) -> list[int]:
        self.refresh_active(
            local_to_global,
            per_cam_tracks,
            zone_map=zone_map,
            cam_ids=cam_ids,
            local_zone_tracker=local_zone_tracker,
        )
        for gid, track in list(self.tracks.items()):
            if not track.frames or track.frames[-1] != int(frame_idx):
                track.append_missed(frame_idx)
        self.manage_states(frame_idx, lost_after=lost_after, delete_after=delete_after)
        return self.prune_deleted()
