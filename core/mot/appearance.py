"""Appearance feature update (EMA / AAF) and distance helpers for MOT/MCMT."""
from __future__ import annotations

from typing import Mapping

import numpy as np

_APPEARANCE_MODES = frozenset({"ema", "aaf"})


def normalize_appearance_mode(mode: str) -> str:
    m = str(mode).lower().strip()
    if m not in _APPEARANCE_MODES:
        raise ValueError(f"appearance_update must be one of {_APPEARANCE_MODES}, got {mode!r}")
    return m


def normalize_l2(f: np.ndarray) -> np.ndarray:
    """L2-normalize a feature vector."""
    arr = np.asarray(f, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        return arr.copy()
    return (arr / norm).astype(np.float32)


def normalized_for_matching(f: np.ndarray, mode: str) -> np.ndarray:
    """Return the vector used in cosine similarity / distance."""
    mode = normalize_appearance_mode(mode)
    arr = np.asarray(f, dtype=np.float32).reshape(-1)
    if mode == "ema":
        return arr.copy()
    return normalize_l2(arr)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine dissimilarity for (possibly unnormalized) vectors."""
    a_n = normalize_l2(a)
    b_n = normalize_l2(b)
    if not np.isfinite(a_n).all() or not np.isfinite(b_n).all():
        return 1.0
    return float(1.0 - np.clip(np.dot(a_n, b_n), -1.0, 1.0))


def update_appearance(
    current: np.ndarray | None,
    new_feat: np.ndarray,
    *,
    mode: str = "aaf",
    alpha: float = 0.9,
) -> np.ndarray:
    """Update stored appearance vector with a new detection feature.

    AAF: F = F + f / ||f||  (unnormalized sum in storage).
    EMA: emb = alpha * emb + (1 - alpha) * f; then L2-normalize.
    """
    mode = normalize_appearance_mode(mode)
    feat = np.asarray(new_feat, dtype=np.float32).reshape(-1)

    if mode == "aaf":
        f_norm = normalize_l2(feat)
        if current is None:
            return f_norm.copy()
        cur = np.asarray(current, dtype=np.float32).reshape(-1)
        return (cur + f_norm).astype(np.float32)

    if current is None:
        out = feat.copy()
    else:
        cur = np.asarray(current, dtype=np.float32).reshape(-1)
        out = alpha * cur + (1.0 - alpha) * feat
    return normalize_l2(out)


def global_appearance_distance(
    query_norm: np.ndarray | None,
    local_feats: Mapping[tuple[int, int], np.ndarray | None],
    mode: str = "aaf",
) -> float | None:
    """Mean cosine distance from query to all local appearance entries in a global track."""
    if query_norm is None or not local_feats:
        return None

    q = normalized_for_matching(query_norm, mode)
    dists: list[float] = []
    for feat in local_feats.values():
        if feat is None:
            continue
        dists.append(cosine_distance(q, normalized_for_matching(feat, mode)))

    if not dists:
        return None
    return float(sum(dists) / len(dists))


def _appearance_feats_for_cam(
    local_feats: Mapping[tuple[int, int], np.ndarray | None],
    cam: int,
    mode: str,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for (c, _tid), feat in local_feats.items():
        if c != cam or feat is None:
            continue
        out.append(normalized_for_matching(feat, mode))
    return out


def cross_camera_appearance_distance(
    query_norm: np.ndarray | None,
    local_feats: Mapping[tuple[int, int], np.ndarray | None],
    query_cam: int,
    active_cameras: set[int] | frozenset[int],
    last_seen_cam: int | None,
    mode: str = "aaf",
    *,
    cam_last_frame: Mapping[int, int] | None = None,
    frame_idx: int | None = None,
    max_gap_frames: int | None = None,
) -> float | None:
    """Min cosine distance to appearance on other *active* cameras.

  Fallback: embedding from ``last_seen_cam`` when no other active cameras remain
  (if still within ``max_gap_frames`` of ``frame_idx``).
    """
    if query_norm is None or not local_feats:
        return None

    q = normalized_for_matching(query_norm, mode)
    dists: list[float] = []
    for cam in active_cameras:
        if cam == query_cam:
            continue
        for feat_n in _appearance_feats_for_cam(local_feats, cam, mode):
            dists.append(cosine_distance(q, feat_n))

    if dists:
        return float(min(dists))

    if last_seen_cam is None or last_seen_cam == query_cam:
        return None

    if (
        cam_last_frame is not None
        and frame_idx is not None
        and max_gap_frames is not None
    ):
        last_f = cam_last_frame.get(last_seen_cam)
        if last_f is None or frame_idx - last_f > max_gap_frames:
            return None

    fallback_feats = _appearance_feats_for_cam(local_feats, last_seen_cam, mode)
    if not fallback_feats:
        return None
    return float(min(cosine_distance(q, f) for f in fallback_feats))
