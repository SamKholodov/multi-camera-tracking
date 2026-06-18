from __future__ import annotations

from core.io.calibration import world_distance


def linear_prediction(observations, frame_idx: int):
    """Predict world point at frame_idx from oldest-to-newest observations."""
    if len(observations) < 1:
        return None
    if len(observations) == 1:
        return observations[-1][1]
    f0, p0 = observations[-2]
    f1, p1 = observations[-1]
    dt = int(f1) - int(f0)
    if dt <= 0:
        return p1
    gap = int(frame_idx) - int(f1)
    vx = (float(p1[0]) - float(p0[0])) / float(dt)
    vy = (float(p1[1]) - float(p0[1])) / float(dt)
    return (float(p1[0]) + vx * gap, float(p1[1]) + vy * gap)


def trajectory_cost_adjustment(config, query_wpt, gmeta, frame_idx: int) -> float:
    if not config.trajectory_enabled or config.trajectory_mode == "off":
        return 0.0
    if query_wpt is None:
        return 0.0
    if not hasattr(gmeta, "recent_world_observations"):
        return 0.0
    observations = gmeta.recent_world_observations(config.trajectory_history_k)
    pred = linear_prediction(observations, frame_idx)
    if pred is None:
        return 0.0
    err = world_distance(query_wpt, pred, metric=config.geometry_distance_metric)
    threshold = max(float(config.trajectory_threshold_m), 1e-6)
    ratio = float(err) / threshold
    return float(config.trajectory_penalty_scale) * max(0.0, ratio - 1.0)
