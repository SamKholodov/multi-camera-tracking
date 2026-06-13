from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from core.io.calibration import world_gps_distance_m
from core.mot.appearance import cross_camera_appearance_distance


@dataclass(frozen=True)
class CrossCameraAssociationConfig:
    gate_mode: str = "mixed"
    different_cam: bool = True
    temporal: bool = True
    geometry_overlap: bool = True
    cam_transition: bool = False
    zone_transition: bool = False
    reid_matching: bool = True
    geometry_max_distance_m: float = 8.0
    max_cross_cam_gap_frames: int = 300
    reid_cost_threshold: float = 0.25
    geometry_far_penalty: float = 10.0
    geometry_missing_penalty: float = 10.0
    temporal_far_penalty: float = 10.0
    temporal_missing_penalty: float = 10.0
    cam_transitions: Mapping[int, set[int]] = field(default_factory=dict)

    zones_path: str | None = None

    @classmethod
    def from_yaml(cls, multi_cfg=None) -> "CrossCameraAssociationConfig":
        if isinstance(multi_cfg, cls):
            return multi_cfg
        multi_cfg = dict(multi_cfg or {})
        assoc = dict(multi_cfg.get("association") or {})
        gates = dict(assoc.get("gates") or {})

        geometry_max_distance_m = assoc.get(
            "geometry_max_distance_m",
            multi_cfg.get("geometry_max_distance", 8.0),
        )
        max_cross_cam_gap_frames = assoc.get(
            "max_cross_cam_gap_frames",
            multi_cfg.get("max_cross_cam_gap_frames", 300),
        )
        reid_cost_threshold = assoc.get(
            "reid_cost_threshold",
            multi_cfg.get("association_cost_threshold", 0.25),
        )

        gate_mode = str(assoc.get("gate_mode", "mixed")).lower().strip()
        if gate_mode not in {"mixed", "hard", "soft"}:
            raise ValueError("association.gate_mode must be one of: mixed, hard, soft")

        return cls(
            gate_mode=gate_mode,
            different_cam=bool(gates.get("different_cam", True)),
            temporal=bool(gates.get("temporal", True)),
            geometry_overlap=bool(gates.get("geometry_overlap", True)),
            cam_transition=bool(gates.get("cam_transition", False)),
            zone_transition=bool(gates.get("zone_transition", False)),
            reid_matching=bool(assoc.get("reid_matching", True)),
            geometry_max_distance_m=float(geometry_max_distance_m),
            max_cross_cam_gap_frames=int(max_cross_cam_gap_frames),
            reid_cost_threshold=float(reid_cost_threshold),
            geometry_far_penalty=float(assoc.get("geometry_far_penalty", 10.0)),
            geometry_missing_penalty=float(assoc.get("geometry_missing_penalty", 10.0)),
            temporal_far_penalty=float(assoc.get("temporal_far_penalty", 10.0)),
            temporal_missing_penalty=float(assoc.get("temporal_missing_penalty", 10.0)),
            cam_transitions=_normalize_cam_transitions(assoc.get("cam_transitions")),
            zones_path=assoc.get("zones_path"),
        )


def _normalize_cam_transitions(raw) -> dict[int, set[int]]:
    if not raw:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("association.cam_transitions must be a mapping")
    out: dict[int, set[int]] = {}
    for src, dsts in raw.items():
        if dsts is None:
            out[int(src)] = set()
        elif isinstance(dsts, (list, tuple, set)):
            out[int(src)] = {int(dst) for dst in dsts}
        else:
            out[int(src)] = {int(dsts)}
    return out


def classify_scenario(active_cameras, query_cam: int) -> str:
    other_active = {int(c) for c in (active_cameras or set()) if int(c) != int(query_cam)}
    return "overlap" if other_active else "handoff"


def min_overlap_distance_m(query_wpt, gmeta, query_cam: int) -> float | None:
    if query_wpt is None:
        return None
    cam_world = gmeta.cam_world or {}
    refs = [
        cam_world[c]
        for c in (gmeta.active_cameras or set())
        if int(c) != int(query_cam) and c in cam_world and cam_world[c] is not None
    ]
    if not refs:
        return None
    return float(min(world_gps_distance_m(query_wpt, ref) for ref in refs))


def passes_cam_transition(
    config: CrossCameraAssociationConfig,
    query_cam: int,
    gmeta,
) -> bool:
    if not config.cam_transition:
        return True
    if not config.cam_transitions:
        return True
    src_cam = gmeta.last_seen_cam
    if src_cam is None:
        other_active = [
            int(c)
            for c in (gmeta.active_cameras or set())
            if int(c) != int(query_cam)
        ]
        src_cam = other_active[0] if other_active else None
    if src_cam is None:
        return False
    return int(query_cam) in config.cam_transitions.get(int(src_cam), set())


def passes_zone_transition(
    config: CrossCameraAssociationConfig,
    query_cam: int,
    query_bbox,
    gmeta,
    zone_map,
    *,
    zone_cam: int | None = None,
    query_zone_entry: int | None = None,
) -> bool:
    if not config.zone_transition:
        return True
    if zone_map is None:
        return True
    if classify_scenario(gmeta.active_cameras or set(), query_cam) != "handoff":
        return True
    if getattr(zone_map, "mode", "snapshot") == "tracklet":
        src_zone = getattr(gmeta, "zone_exit", None)
        if src_zone is None:
            src_zone = getattr(gmeta, "zone_entry", None)
        return zone_map.allows_tracklet_handoff(src_zone, query_zone_entry)
    src_zone = getattr(gmeta, "last_seen_zone", None)
    cam_for_zone = int(zone_cam if zone_cam is not None else query_cam)
    dst_zone = zone_map.zone_at_bbox(cam_for_zone, query_bbox)
    return zone_map.allows_transition(src_zone, dst_zone)


def passes_gates(
    config: CrossCameraAssociationConfig,
    query_cam: int,
    query_wpt,
    gmeta,
    frame_idx: int,
    *,
    query_bbox=None,
    zone_map=None,
    zone_cam: int | None = None,
    query_zone_entry: int | None = None,
) -> bool:
    if not passes_hard_gates(
        config,
        query_cam,
        query_wpt,
        gmeta,
        frame_idx,
        query_bbox=query_bbox,
        zone_map=zone_map,
        zone_cam=zone_cam,
        query_zone_entry=query_zone_entry,
    ):
        return False
    return _passes_soft_as_hard(
        config,
        query_cam,
        query_wpt,
        gmeta,
        frame_idx,
        query_bbox=query_bbox,
        zone_map=zone_map,
        zone_cam=zone_cam,
    )


def passes_hard_gates(
    config: CrossCameraAssociationConfig,
    query_cam: int,
    query_wpt,
    gmeta,
    frame_idx: int,
    *,
    query_bbox=None,
    zone_map=None,
    zone_cam: int | None = None,
    query_zone_entry: int | None = None,
) -> bool:
    scenario = classify_scenario(gmeta.active_cameras or set(), query_cam)

    if config.different_cam:
        if int(query_cam) in (gmeta.active_cameras or set()):
            return False
        if scenario == "handoff" and gmeta.last_seen_cam == int(query_cam):
            return False

    if not passes_cam_transition(config, query_cam, gmeta):
        return False
    return passes_zone_transition(
        config,
        query_cam,
        query_bbox,
        gmeta,
        zone_map,
        zone_cam=zone_cam,
        query_zone_entry=query_zone_entry,
    )


def _passes_soft_as_hard(
    config: CrossCameraAssociationConfig,
    query_cam: int,
    query_wpt,
    gmeta,
    frame_idx: int,
    *,
    query_bbox=None,
    zone_map=None,
    zone_cam: int | None = None,
) -> bool:
    scenario = classify_scenario(gmeta.active_cameras or set(), query_cam)

    last_f = gmeta.last_frame
    if config.temporal and scenario == "handoff":
        if last_f is None:
            return False
        if int(frame_idx) - int(last_f) > config.max_cross_cam_gap_frames:
            return False

    if config.geometry_overlap and scenario == "overlap":
        dist_m = min_overlap_distance_m(query_wpt, gmeta, query_cam)
        if dist_m is None:
            return False
        max_m = config.geometry_max_distance_m
        if zone_map is not None:
            cam_for_zone = int(zone_cam if zone_cam is not None else query_cam)
            query_zone = zone_map.zone_at_bbox(cam_for_zone, query_bbox)
            active_zones = set(getattr(gmeta, "active_zones", set()) or set())
            max_m = zone_map.geometry_max_distance_m(
                config.geometry_max_distance_m,
                query_zone=query_zone,
                active_zones=active_zones,
            )
        if dist_m > max_m:
            return False

    return True


def geometry_penalty(
    config: CrossCameraAssociationConfig,
    query_wpt,
    gmeta,
    query_cam: int,
    *,
    query_bbox=None,
    zone_map=None,
    zone_cam: int | None = None,
) -> float:
    if not config.geometry_overlap:
        return 1.0
    if classify_scenario(gmeta.active_cameras or set(), query_cam) != "overlap":
        return 1.0
    dist_m = min_overlap_distance_m(query_wpt, gmeta, query_cam)
    if dist_m is None:
        return config.geometry_missing_penalty
    max_m = config.geometry_max_distance_m
    if zone_map is not None:
        cam_for_zone = int(zone_cam if zone_cam is not None else query_cam)
        query_zone = zone_map.zone_at_bbox(cam_for_zone, query_bbox)
        active_zones = set(getattr(gmeta, "active_zones", set()) or set())
        max_m = zone_map.geometry_max_distance_m(
            config.geometry_max_distance_m,
            query_zone=query_zone,
            active_zones=active_zones,
        )
    if dist_m <= max_m:
        return 1.0
    return config.geometry_far_penalty


def temporal_penalty(
    config: CrossCameraAssociationConfig,
    gmeta,
    frame_idx: int,
    query_cam: int,
) -> float:
    if not config.temporal:
        return 1.0
    if classify_scenario(gmeta.active_cameras or set(), query_cam) != "handoff":
        return 1.0
    last_f = gmeta.last_frame
    if last_f is None:
        return config.temporal_missing_penalty
    if int(frame_idx) - int(last_f) <= config.max_cross_cam_gap_frames:
        return 1.0
    return config.temporal_far_penalty


def reid_cost_for_match(
    config: CrossCameraAssociationConfig,
    query_feat,
    gmeta,
    query_cam: int,
    *,
    appearance_update: str,
    frame_idx: int,
) -> float | None:
    if not config.reid_matching or query_feat is None:
        return None
    return cross_camera_appearance_distance(
        query_feat,
        gmeta.local_appearance,
        query_cam,
        gmeta.active_cameras or set(),
        gmeta.last_seen_cam,
        mode=appearance_update,
        cam_last_frame=gmeta.cam_last_frame,
        frame_idx=frame_idx,
        max_gap_frames=config.max_cross_cam_gap_frames,
    )


def association_cost_for_match(
    config: CrossCameraAssociationConfig,
    query_cam: int,
    query_wpt,
    query_feat,
    gmeta,
    frame_idx: int,
    *,
    appearance_update: str,
    query_bbox=None,
    zone_map=None,
    zone_cam: int | None = None,
    query_zone_entry: int | None = None,
) -> float | None:
    if not passes_hard_gates(
        config,
        query_cam,
        query_wpt,
        gmeta,
        frame_idx,
        query_bbox=query_bbox,
        zone_map=zone_map,
        zone_cam=zone_cam,
        query_zone_entry=query_zone_entry,
    ):
        return None

    reid = reid_cost_for_match(
        config,
        query_feat,
        gmeta,
        query_cam,
        appearance_update=appearance_update,
        frame_idx=frame_idx,
    )
    if reid is None:
        return None

    if config.gate_mode == "hard":
        if not _passes_soft_as_hard(
            config,
            query_cam,
            query_wpt,
            gmeta,
            frame_idx,
            query_bbox=query_bbox,
            zone_map=zone_map,
            zone_cam=zone_cam,
        ):
            return None
        return reid

    geo_pen = geometry_penalty(
        config,
        query_wpt,
        gmeta,
        query_cam,
        query_bbox=query_bbox,
        zone_map=zone_map,
        zone_cam=zone_cam,
    )
    temp_pen = temporal_penalty(config, gmeta, frame_idx, query_cam)
    return float(reid * geo_pen * temp_pen)

