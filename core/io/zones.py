"""Zone polygons and handoff graph for MCMT association."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class ZoneMap:
    """Per-camera zone polygons and global zone-to-zone handoff rules."""

    zones: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)
    transitions: dict[int, frozenset[int]] = field(default_factory=dict)
    mode: str = "snapshot"
    stabilize_frames: int = 5
    fail_open: bool = True
    geometry_soft_zones: frozenset[int] = frozenset()
    geometry_soft_max_distance_m: float | None = None

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ZoneMap":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        zones_raw = data.get("zones") or {}
        zones: dict[int, dict[int, np.ndarray]] = {}
        for cam_s, zone_dict in zones_raw.items():
            cam = int(str(cam_s).lstrip("cC"))
            zones[cam] = {}
            for zone_s, pts in (zone_dict or {}).items():
                zone_id = int(str(zone_s).lstrip("zZ"))
                arr = np.asarray(pts, dtype=np.float64)
                if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3:
                    raise ValueError(
                        f"zones[{cam_s}][{zone_s}] must be a polygon with >=3 points"
                    )
                zones[cam][zone_id] = arr

        transitions_raw = data.get("transitions") or {}
        transitions: dict[int, frozenset[int]] = {}
        for zone_s, dsts in transitions_raw.items():
            zone_id = int(str(zone_s).lstrip("zZ"))
            if dsts is None:
                transitions[zone_id] = frozenset()
            elif isinstance(dsts, (list, tuple, set)):
                transitions[zone_id] = frozenset(int(str(d).lstrip("zZ")) for d in dsts)
            else:
                transitions[zone_id] = frozenset({int(str(dsts).lstrip("zZ"))})

        mode = str(data.get("mode", "snapshot")).lower().strip()
        if mode not in {"snapshot", "tracklet"}:
            raise ValueError("zone mode must be one of: snapshot, tracklet")

        tracklet = data.get("tracklet") or {}
        soft = data.get("geometry_soft") or {}
        soft_zones = soft.get("zones") or []
        soft_max = soft.get("max_distance_m")
        return cls(
            zones=zones,
            transitions=transitions,
            mode=mode,
            stabilize_frames=int(tracklet.get("stabilize_frames", 5)),
            fail_open=bool(tracklet.get("fail_open", True)),
            geometry_soft_zones=frozenset(int(z) for z in soft_zones),
            geometry_soft_max_distance_m=float(soft_max) if soft_max is not None else None,
        )

    def zone_at_bbox(
        self,
        cam: int,
        bbox: tuple[float, float, float, float] | None,
    ) -> int | None:
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        px = (float(x1) + float(x2)) / 2.0
        py = float(y2)
        return self.zone_at_point(int(cam), px, py)

    def zone_at_point(self, cam: int, x: float, y: float) -> int | None:
        cam_zones = self.zones.get(int(cam))
        if not cam_zones:
            return None
        hits = [
            zone_id
            for zone_id, poly in cam_zones.items()
            if _point_in_polygon(x, y, poly)
        ]
        if not hits:
            return None
        if len(hits) == 1:
            return hits[0]
        # Overlapping zone labels: prefer the smallest area polygon.
        areas = [_polygon_area(cam_zones[z]) for z in hits]
        return hits[int(np.argmin(areas))]

    def allows_transition(self, src_zone: int | None, dst_zone: int | None) -> bool:
        # Polygons do not cover the full ROI; unknown zone → do not block handoff.
        if src_zone is None or dst_zone is None:
            return self.fail_open
        # Same global zone on different cameras (e.g. Z4 on c007 ↔ Z4 on c009).
        if int(src_zone) == int(dst_zone):
            return True
        allowed = self.transitions.get(int(src_zone))
        if allowed is None:
            return self.fail_open
        return int(dst_zone) in allowed

    def allows_tracklet_handoff(
        self,
        src_exit_or_entry: int | None,
        dst_entry: int | None,
    ) -> bool:
        return self.allows_transition(src_exit_or_entry, dst_entry)

    def resolve_stable_zone(
        self,
        labels: list[int | None],
        *,
        window: int | None = None,
    ) -> int | None:
        n = int(window if window is not None else self.stabilize_frames)
        sample = labels[-n:]
        if len(sample) < n:
            return None
        if any(z is None for z in sample):
            return None
        first = int(sample[0])
        if all(int(z) == first for z in sample):
            return first
        return None

    def geometry_max_distance_m(
        self,
        default_m: float,
        *,
        query_zone: int | None,
        active_zones: set[int] | None = None,
    ) -> float:
        if self.geometry_soft_max_distance_m is None:
            return default_m
        soft = self.geometry_soft_zones
        if query_zone is not None and int(query_zone) in soft:
            return self.geometry_soft_max_distance_m
        if active_zones and soft.intersection({int(z) for z in active_zones}):
            return self.geometry_soft_max_distance_m
        return default_m


def _point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        ):
            inside = not inside
    return inside


def _polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
