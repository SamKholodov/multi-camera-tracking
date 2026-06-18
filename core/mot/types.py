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
    *,
    world_anchor: str = "bottom_center",
    contact_uv_by_det: np.ndarray | None = None,
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

    from core.io.calibration import project_bbox_bottom_center, project_point
    from core.geometry.contact_point.model import uv_to_pixel

    H = np.asarray(H_image_to_world, dtype=np.float64)
    use_contact = str(world_anchor).lower().strip() == "contact_point"
    for i in range(n):
        x1, y1, x2, y2 = map(float, out[i, :4])
        has_detection = int(out[i, 7]) if out.shape[1] > 7 else 1
        det_idx = int(out[i, 6]) if out.shape[1] > 6 else -1

        xw, yw = project_bbox_bottom_center(H, x1, y1, x2, y2)
        if (
            use_contact
            and has_detection == 1
            and contact_uv_by_det is not None
            and 0 <= det_idx < len(contact_uv_by_det)
        ):
            uv = contact_uv_by_det[det_idx]
            if np.all(np.isfinite(uv)):
                px, py = uv_to_pixel(uv, (x1, y1, x2, y2))
                xw, yw = project_point(H, px, py)

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
