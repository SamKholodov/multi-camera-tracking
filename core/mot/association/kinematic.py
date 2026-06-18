from __future__ import annotations

from core.io.calibration import world_distance


def _classify_scenario(active_cameras, query_cam: int) -> str:
    other_active = {int(c) for c in (active_cameras or set()) if int(c) != int(query_cam)}
    return "overlap" if other_active else "handoff"


def implied_speed_mps(
    query_wpt,
    ref_wpt,
    delta_frames: int,
    fps: float,
    *,
    metric: str = "plane",
) -> float | None:
    if query_wpt is None or ref_wpt is None:
        return None
    if delta_frames <= 0 or fps <= 0:
        return None
    dist_m = world_distance(query_wpt, ref_wpt, metric=metric)
    return float(dist_m) / (float(delta_frames) / float(fps))


def _speed_reference(config, query_wpt, gmeta, frame_idx: int, query_cam: int):
    scenario = _classify_scenario(gmeta.active_cameras or set(), query_cam)
    metric = config.geometry_distance_metric
    if scenario == "handoff":
        last_f = gmeta.last_frame
        if last_f is None:
            return None
        return implied_speed_mps(
            query_wpt,
            gmeta.last_seen_world,
            int(frame_idx) - int(last_f),
            config.video_fps,
            metric=metric,
        )

    best_speed = None
    cam_world = gmeta.cam_world or {}
    cam_last_frame = gmeta.cam_last_frame or {}
    for cam_id in gmeta.active_cameras or set():
        if int(cam_id) == int(query_cam):
            continue
        ref_wpt = cam_world.get(int(cam_id))
        ref_frame = cam_last_frame.get(int(cam_id))
        if ref_wpt is None or ref_frame is None:
            continue
        speed = implied_speed_mps(
            query_wpt,
            ref_wpt,
            int(frame_idx) - int(ref_frame),
            config.video_fps,
            metric=metric,
        )
        if speed is None:
            continue
        if best_speed is None or speed < best_speed:
            best_speed = speed
    return best_speed


def speed_cost_adjustment(config, query_wpt, gmeta, frame_idx: int, query_cam: int):
    if not config.speed_limit_enabled:
        return 0.0
    speed = _speed_reference(config, query_wpt, gmeta, frame_idx, query_cam)
    if speed is None:
        return 0.0
    v_max = max(float(config.speed_v_max_mps), 1e-6)
    ratio = float(speed) / v_max
    if config.speed_limit_mode == "hard":
        if ratio > 1.0 + float(config.speed_margin):
            return None
        return 0.0
    return float(config.speed_penalty_scale) * max(0.0, ratio - 1.0)
