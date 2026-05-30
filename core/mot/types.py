"""Track row layout and world-coordinate enrichment for SCT / MCMT pipelines.

Track row layout (float32)::

    [x1, y1, x2, y2, local_tid, conf, det_idx, has_detection, xworld, yworld]

Tracker wrappers return 8 columns. ``enrich_tracks_world`` always produces
``(N, 10)``: with a valid ``H_image_to_world``, cols 8–9 hold projected
bottom-center world coordinates; otherwise ``(-1, -1)``.
"""
from __future__ import annotations

import numpy as np

TRACK_COL_XWORLD = 8
TRACK_COL_YWORLD = 9
TRACK_NCOLS = 10
WORLD_COORD_MISSING = -1.0


def homography_valid(H_image_to_world) -> bool:
    """True when ``H`` is a non-identity 3x3 image-to-world matrix."""
    if H_image_to_world is None:
        return False
    H = np.asarray(H_image_to_world, dtype=np.float64)
    if H.shape != (3, 3):
        return False
    return not np.allclose(H, np.eye(3))


def enrich_tracks_world(
    tracks: np.ndarray,
    H_image_to_world: np.ndarray,
) -> np.ndarray:
    """Pad to ``(N, 10)`` and fill xworld/yworld or ``(-1, -1)``."""
    if tracks is None or len(tracks) == 0:
        return np.empty((0, TRACK_NCOLS), dtype=np.float32)
    tracks = np.asarray(tracks, dtype=np.float32)
    n = tracks.shape[0]
    ncol = min(tracks.shape[1], TRACK_NCOLS)
    out = np.zeros((n, TRACK_NCOLS), dtype=np.float32)
    out[:, :ncol] = tracks[:, :ncol]
    if not homography_valid(H_image_to_world):
        out[:, TRACK_COL_XWORLD] = WORLD_COORD_MISSING
        out[:, TRACK_COL_YWORLD] = WORLD_COORD_MISSING
        return out

    from core.io.calibration import project_bbox_bottom_center

    H = np.asarray(H_image_to_world, dtype=np.float64)
    for i in range(n):
        xw, yw = project_bbox_bottom_center(
            H, out[i, 0], out[i, 1], out[i, 2], out[i, 3]
        )
        out[i, TRACK_COL_XWORLD] = xw
        out[i, TRACK_COL_YWORLD] = yw
    return out


def world_point_from_row(row) -> tuple[float, float] | None:
    """Read valid world coords from an enriched row, or None."""
    if row is None or len(row) < TRACK_NCOLS:
        return None
    xw, yw = float(row[TRACK_COL_XWORLD]), float(row[TRACK_COL_YWORLD])
    if np.isnan(xw) or np.isnan(yw):
        return None
    if xw == WORLD_COORD_MISSING and yw == WORLD_COORD_MISSING:
        return None
    return (xw, yw)
