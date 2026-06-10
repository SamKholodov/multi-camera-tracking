"""Batch ReID feature extraction across synchronized camera frames."""
from __future__ import annotations

import numpy as np
import torch


def _empty_features() -> np.ndarray:
    return np.empty((0, 0), dtype=np.float32)


def batch_reid_features(reid_backend, frames, detections_per_cam) -> list[np.ndarray | None]:
    """Run one ReID forward pass for all detections, then split by camera."""
    crop_batches = []
    counts: list[int] = []

    for frame, dets in zip(frames, detections_per_cam):
        if frame is None or dets is None or len(dets) == 0:
            counts.append(0)
            continue

        dets = np.asarray(dets, dtype=np.float32)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)

        crops = reid_backend.get_crops(dets[:, :4], frame)
        crop_batches.append(crops)
        counts.append(int(crops.shape[0]))

    out: list[np.ndarray | None] = []
    if not crop_batches:
        return [None for _ in counts]

    crops = torch.cat(crop_batches, dim=0)
    crops = reid_backend.inference_preprocess(crops)
    features = reid_backend.forward(crops)
    features = reid_backend.inference_postprocess(features)
    features = np.asarray(features, dtype=np.float32)

    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    features = features / np.clip(norms, 1e-12, None)

    offset = 0
    for count in counts:
        if count == 0:
            out.append(None)
            continue
        out.append(features[offset : offset + count])
        offset += count
    return out
