"""Batch ReID feature extraction across synchronized camera frames."""
from __future__ import annotations

import numpy as np
import torch


def _empty_features() -> np.ndarray:
    return np.empty((0, 0), dtype=np.float32)


def _normalize_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    out = features.copy()
    mask = norms.squeeze(-1) > 1e-12
    out[mask] /= norms[mask]
    return out


def batch_reid_features(
    reid_backend,
    frames,
    detections_per_cam,
    *,
    max_batch_size: int | None = None,
    conf_thresh: float | None = None,
) -> list[np.ndarray | None]:
    """Run ReID forward pass for detections, then split by camera.

    Only detections with ``conf >= conf_thresh`` are sent through the ReID model
    (same gate as ``reid_accum_conf_thresh`` in DeepOcSort). Low-confidence rows
    get zero embeddings so association falls back to motion/IoU for those dets.

    When ``max_batch_size`` is set, inference is chunked to limit GPU memory.
    """
    thresh = 0.0 if conf_thresh is None else float(conf_thresh)
    counts: list[int] = []
    dets_per_cam: list[np.ndarray | None] = []
    crop_batches: list[torch.Tensor] = []
    # (camera index, detection row index within that camera)
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

        crops = reid_backend.get_crops(dets[keep, :4], frame)
        crop_batches.append(crops)
        det_indices = np.flatnonzero(keep)
        for det_idx in det_indices:
            selected.append((cam_idx, int(det_idx)))

    out: list[np.ndarray | None] = [None for _ in counts]
    if not counts or not crop_batches:
        return out

    crops = torch.cat(crop_batches, dim=0)
    crops = reid_backend.inference_preprocess(crops)
    feature_chunks: list[np.ndarray] = []
    with torch.no_grad():
        if max_batch_size is None or int(max_batch_size) <= 0:
            feats = reid_backend.forward(crops)
            feats = reid_backend.inference_postprocess(feats)
            feature_chunks.append(np.asarray(feats, dtype=np.float32))
        else:
            chunk_size = max(1, int(max_batch_size))
            for start in range(0, crops.shape[0], chunk_size):
                chunk = crops[start : start + chunk_size]
                feats = reid_backend.forward(chunk)
                feats = reid_backend.inference_postprocess(feats)
                feature_chunks.append(np.asarray(feats, dtype=np.float32))
    computed = (
        np.concatenate(feature_chunks, axis=0)
        if feature_chunks
        else _empty_features()
    )
    if computed.size == 0:
        return out

    computed = _normalize_rows(computed)
    emb_dim = int(computed.shape[1])

    per_cam_features: dict[int, list[tuple[int, np.ndarray]]] = {}
    for (cam_idx, det_idx), feat in zip(selected, computed):
        per_cam_features.setdefault(cam_idx, []).append((det_idx, feat))

    for cam_idx, count in enumerate(counts):
        if count == 0:
            continue
        arr = np.zeros((count, emb_dim), dtype=np.float32)
        for det_idx, feat in per_cam_features.get(cam_idx, []):
            arr[det_idx] = feat
        out[cam_idx] = arr
    return out
