from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from core.io.calibration import world_distance
from core.mot.appearance import cross_camera_appearance_distance
from core.mot.association.kinematic import speed_cost_adjustment
from core.mot.association.trajectory import trajectory_cost_adjustment


@dataclass(frozen=True)
class CrossCameraAssociationConfig:
    temporal: bool = False
    temporal_mode: str = "off"
    cam_transition: bool = False
    zone_transition: bool = False
    reid_matching: bool = True
    geometry_distance_metric: str = "plane"
    geometry_t_min_m: float = 14.0
    geometry_t_distant_m: float = 38.0
    geometry_mid_penalty: float = 0.15
    same_cam_cost_add: float = 0.25
    max_cross_cam_gap_frames: int = 300
    reid_cost_threshold: float = 0.25
    reid_strong_reject_threshold: float = 0.50
    same_frame_linking: bool = False
    geometry_missing_penalty: float = 10.0
    temporal_missing_penalty: float = 10.0
    temporal_mid_penalty: float = 0.10
    global_delete_after_frames: int = 600
    video_fps: float = 10.0
    speed_limit_enabled: bool = False
    speed_limit_mode: str = "hard"
    speed_v_max_mps: float = 25.0
    speed_margin: float = 0.2
    speed_penalty_scale: float = 0.15
    trajectory_enabled: bool = False
    trajectory_mode: str = "off"
    trajectory_history_k: int = 3
    trajectory_threshold_m: float = 10.0
    trajectory_penalty_scale: float = 0.15
    cam_transitions: Mapping[int, set[int]] = field(default_factory=dict)
    zones_path: str | None = None

    @classmethod
    def from_yaml(cls, multi_cfg=None) -> "CrossCameraAssociationConfig":
        if isinstance(multi_cfg, cls):
            return multi_cfg
        multi_cfg = dict(multi_cfg or {})
        assoc = dict(multi_cfg.get("association") or {})
        gates = dict(assoc.get("gates") or {})

        max_cross_cam_gap_frames = assoc.get(
            "max_cross_cam_gap_frames",
            multi_cfg.get("max_cross_cam_gap_frames", 300),
        )
        global_delete_after_frames = assoc.get(
            "global_delete_after_frames",
            multi_cfg.get("global_delete_after_frames", max(600, int(max_cross_cam_gap_frames))),
        )
        reid_cost_threshold = assoc.get(
            "reid_cost_threshold",
            multi_cfg.get("association_cost_threshold", 0.25),
        )
        temporal_mode = str(assoc.get("temporal_mode", "off")).lower().strip()
        if temporal_mode not in {"off", "strict", "penalty_only"}:
            raise ValueError("association.temporal_mode must be off, strict, or penalty_only")
        speed_mode = str(assoc.get("speed_limit_mode", "hard")).lower().strip()
        if speed_mode not in {"hard", "penalty"}:
            raise ValueError("association.speed_limit_mode must be hard or penalty")
        trajectory_mode = str(assoc.get("trajectory_mode", "off")).lower().strip()
        if trajectory_mode not in {"off", "linear"}:
            raise ValueError("association.trajectory_mode must be off or linear")

        t_min = _optional_float(assoc.get("geometry_t_min_m"))
        t_distant = _optional_float(assoc.get("geometry_t_distant_m"))
        if t_min is None:
            t_min = 14.0
        if t_distant is None:
            t_distant = 38.0

        return cls(
            temporal=bool(gates.get("temporal", False)) or temporal_mode != "off",
            temporal_mode=temporal_mode,
            cam_transition=bool(gates.get("cam_transition", False)),
            zone_transition=bool(gates.get("zone_transition", False)),
            reid_matching=bool(assoc.get("reid_matching", True)),
            geometry_distance_metric=str(
                assoc.get("geometry_distance_metric", "plane")
            ).lower().strip(),
            geometry_t_min_m=float(t_min),
            geometry_t_distant_m=float(t_distant),
            geometry_mid_penalty=float(assoc.get("geometry_mid_penalty", 0.15)),
            same_cam_cost_add=float(assoc.get("same_cam_cost_add", 0.25)),
            max_cross_cam_gap_frames=int(max_cross_cam_gap_frames),
            reid_cost_threshold=float(reid_cost_threshold),
            reid_strong_reject_threshold=float(
                assoc.get("reid_strong_reject_threshold", 0.50)
            ),
            same_frame_linking=bool(assoc.get("same_frame_linking", False)),
            geometry_missing_penalty=float(assoc.get("geometry_missing_penalty", 10.0)),
            temporal_missing_penalty=float(assoc.get("temporal_missing_penalty", 10.0)),
            temporal_mid_penalty=float(assoc.get("temporal_mid_penalty", 0.10)),
            global_delete_after_frames=int(global_delete_after_frames),
            video_fps=float(assoc.get("video_fps", multi_cfg.get("video_fps", 10.0))),
            speed_limit_enabled=bool(assoc.get("speed_limit_enabled", False)),
            speed_limit_mode=speed_mode,
            speed_v_max_mps=float(assoc.get("speed_v_max_mps", 25.0)),
            speed_margin=float(assoc.get("speed_margin", 0.2)),
            speed_penalty_scale=float(assoc.get("speed_penalty_scale", 0.15)),
            trajectory_enabled=bool(assoc.get("trajectory_enabled", False)),
            trajectory_mode=trajectory_mode,
            trajectory_history_k=int(assoc.get("trajectory_history_k", 3)),
            trajectory_threshold_m=float(assoc.get("trajectory_threshold_m", 10.0)),
            trajectory_penalty_scale=float(assoc.get("trajectory_penalty_scale", 0.15)),
            cam_transitions=_normalize_cam_transitions(assoc.get("cam_transitions")),
            zones_path=assoc.get("zones_path"),
        )


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _uses_geometry_tiers(config: CrossCameraAssociationConfig) -> bool:
    return config.geometry_t_distant_m > 0


uses_geometry_tiers = _uses_geometry_tiers


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


def min_overlap_distance_m(
    query_wpt,
    gmeta,
    query_cam: int,
    *,
    metric: str = "plane",
) -> float | None:
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
    return float(min(world_distance(query_wpt, ref, metric=metric) for ref in refs))


def geometry_match_distance_m(
    query_wpt,
    gmeta,
    query_cam: int,
    *,
    metric: str = "plane",
) -> float | None:
    """World distance for association geometry tiers (overlap, same-cam, handoff)."""
    if query_wpt is None:
        return None
    scenario = classify_scenario(gmeta.active_cameras or set(), query_cam)
    cam_world = gmeta.cam_world or {}

    if int(query_cam) in (gmeta.active_cameras or set()):
        ref = cam_world.get(int(query_cam))
        if ref is not None:
            return float(world_distance(query_wpt, ref, metric=metric))
        return None

    if scenario == "overlap":
        return min_overlap_distance_m(
            query_wpt,
            gmeta,
            query_cam,
            metric=metric,
        )

    if gmeta.last_seen_world is not None:
        return float(world_distance(query_wpt, gmeta.last_seen_world, metric=metric))
    refs = [
        cam_world[c]
        for c in (gmeta.active_cameras or set())
        if c in cam_world and cam_world[c] is not None
    ]
    if not refs:
        return None
    return float(min(world_distance(query_wpt, ref, metric=metric) for ref in refs))


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
    del query_wpt, frame_idx
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


def geometry_cost_adjustment(
    config: CrossCameraAssociationConfig,
    query_wpt,
    gmeta,
    query_cam: int,
    *,
    query_bbox=None,
    zone_map=None,
    zone_cam: int | None = None,
) -> float | None:
    """Additive geometry cost; None means hard reject (tiered mode only)."""
    del query_bbox, zone_map, zone_cam
    if not _uses_geometry_tiers(config):
        return 0.0
    dist_m = geometry_match_distance_m(
        query_wpt,
        gmeta,
        query_cam,
        metric=config.geometry_distance_metric,
    )
    if dist_m is None:
        return config.geometry_missing_penalty
    t_distant = float(config.geometry_t_distant_m)
    if dist_m > t_distant:
        return None
    t_min = float(config.geometry_t_min_m)
    if dist_m > t_min:
        return config.geometry_mid_penalty
    return 0.0


def same_cam_cost_adjustment(
    config: CrossCameraAssociationConfig,
    query_cam: int,
    gmeta,
) -> float:
    if not _uses_geometry_tiers(config):
        return 0.0
    if int(query_cam) in (gmeta.active_cameras or set()):
        return config.same_cam_cost_add
    return 0.0


def temporal_cost_adjustment(
    config: CrossCameraAssociationConfig,
    gmeta,
    frame_idx: int,
    query_cam: int,
) -> float:
    if config.temporal_mode == "off" or not _uses_geometry_tiers(config):
        return 0.0
    if classify_scenario(gmeta.active_cameras or set(), query_cam) != "handoff":
        return 0.0
    last_f = gmeta.last_frame
    if last_f is None:
        return config.temporal_missing_penalty
    gap = int(frame_idx) - int(last_f)
    if config.temporal_mode == "strict":
        return 0.0
    if gap <= config.max_cross_cam_gap_frames:
        return 0.0
    return config.temporal_mid_penalty


def passes_temporal_gate(
    config: CrossCameraAssociationConfig,
    gmeta,
    frame_idx: int,
    query_cam: int,
) -> bool:
    if config.temporal_mode != "strict":
        return True
    if classify_scenario(gmeta.active_cameras or set(), query_cam) != "handoff":
        return True
    last_f = gmeta.last_frame
    if last_f is None:
        return False
    return int(frame_idx) - int(last_f) <= config.max_cross_cam_gap_frames


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
    if not passes_temporal_gate(config, gmeta, frame_idx, query_cam):
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

    if _uses_geometry_tiers(config):
        geo_add = geometry_cost_adjustment(
            config,
            query_wpt,
            gmeta,
            query_cam,
            query_bbox=query_bbox,
            zone_map=zone_map,
            zone_cam=zone_cam,
        )
        if geo_add is None:
            return None
        cost = float(reid) + geo_add
        cost += same_cam_cost_adjustment(config, query_cam, gmeta)
        cost += temporal_cost_adjustment(config, gmeta, frame_idx, query_cam)
        speed_add = speed_cost_adjustment(config, query_wpt, gmeta, frame_idx, query_cam)
        if speed_add is None:
            return None
        cost += speed_add
        cost += trajectory_cost_adjustment(config, query_wpt, gmeta, frame_idx)
        return cost

    return float(reid)
