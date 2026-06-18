"""Batch contact point inference across synchronized camera frames."""

from __future__ import annotations

import numpy as np
import torch


def batch_contact_point_uv(
    contact_model,
    frames,
    detections_per_cam,
    *,
    max_batch_size: int | None = None,
    conf_thresh: float = 0.0,
) -> list[np.ndarray | None]:
    """Predict per-detection ``(u, v)`` for each camera.

    Returns one array per camera with shape ``(N_dets, 2)``. Rows below
  ``conf_thresh`` are left as ``nan`` so world enrichment can fall back to
    bbox bottom-center.
    """
    thresh = float(conf_thresh)
    counts: list[int] = []
    dets_per_cam: list[np.ndarray | None] = []
    crop_batches: list[torch.Tensor] = []
    selected: list[tuple[int, int]] = []

    for cam_idx, (frame, dets) in enumerate(zip(frames, detections_per_cam)):
        if frame is None or dets is None or len(dets) == 0:
            counts.append(0)
            dets_per_cam.append(None)
            continue

        dets = np.asarray(dets, dtype=np.float32)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)

        counts.append(int(dets.shape[0]))
        dets_per_cam.append(dets)

        conf_col = dets[:, 4] if dets.shape[1] > 4 else np.ones(dets.shape[0], dtype=np.float32)
        keep = conf_col >= thresh
        if not np.any(keep):
            continue

        crops = contact_model.get_crops(dets[keep, :4], frame)
        crop_batches.append(crops)
        for det_idx in np.flatnonzero(keep):
            selected.append((cam_idx, int(det_idx)))

    out: list[np.ndarray | None] = [None for _ in counts]
    if not counts or not crop_batches:
        return out

    crops = torch.cat(crop_batches, dim=0)
    uv_chunks: list[np.ndarray] = []
    with torch.no_grad():
        if max_batch_size is None or int(max_batch_size) <= 0:
            uv_chunks.append(contact_model.predict_uv_batch(crops))
        else:
            chunk_size = max(1, int(max_batch_size))
            for start in range(0, crops.shape[0], chunk_size):
                uv_chunks.append(contact_model.predict_uv_batch(crops[start : start + chunk_size]))

    computed = (
        np.concatenate(uv_chunks, axis=0)
        if uv_chunks
        else np.empty((0, 2), dtype=np.float32)
    )
    if computed.size == 0:
        return out

    per_cam_uv: dict[int, list[tuple[int, np.ndarray]]] = {}
    for (cam_idx, det_idx), uv in zip(selected, computed):
        per_cam_uv.setdefault(cam_idx, []).append((det_idx, uv))

    for cam_idx, count in enumerate(counts):
        if count == 0:
            continue
        arr = np.full((count, 2), np.nan, dtype=np.float32)
        for det_idx, uv in per_cam_uv.get(cam_idx, []):
            arr[det_idx] = uv
        out[cam_idx] = arr
    return out
