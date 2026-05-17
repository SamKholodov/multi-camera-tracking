from typing import Any

import numpy as np

from core.mot.tracker.deepocsort import DeepOcSort


class DeepOcSortTracker:
    def __init__(
        self,
        reid_model: Any | None = None,
        use_embeddings: bool = False,
        custom_reid_extractor=None,
        **tracker_kwargs,
    ):
        self.use_embeddings = bool(use_embeddings)
        self.custom_reid_extractor = custom_reid_extractor

        self.tracker = DeepOcSort(
            reid_model=reid_model,
            embedding_off=not self.use_embeddings,
            **tracker_kwargs,
        )

    @staticmethod
    def _normalize_embeddings(features, expected_count):
        if features is None:
            return None

        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != expected_count:
            return None

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return features / norms

    def _compute_embeddings(self, dets, frame):
        if callable(self.custom_reid_extractor):
            custom_features = self.custom_reid_extractor(dets=dets, frame=frame)
            return self._normalize_embeddings(custom_features, len(dets))
        return None

    def update(self, detections, frame, embs=None):
        if detections is None:
            dets = np.empty((0, 6), dtype=np.float32)
        else:
            dets = np.asarray(detections, dtype=np.float32)
            if dets.size == 0:
                dets = np.empty((0, 6), dtype=np.float32)

        if len(dets) > 0 and self.use_embeddings:
            if embs is not None:
                embs = self._normalize_embeddings(embs, len(dets))
            else:
                embs = self._compute_embeddings(dets, frame)
        else:
            embs = None

        tracks = self.tracker.update(dets, frame, embs=embs)

        if tracks is None or len(tracks) == 0:
            return np.empty((0, 8), dtype=np.float32)

        out = []
        current_det_inds = set(int(i) for i in range(len(dets)))

        for t in tracks:
            x1, y1, x2, y2 = t[:4]
            tid = int(t[4])
            conf = t[5]
            det_idx = t[7]
            has_detection = 1.0 if int(det_idx) in current_det_inds else 0.0
            out.append([x1, y1, x2, y2, tid, conf, det_idx, has_detection])

        return np.array(out, dtype=np.float32)

    def get_track_feature_map(self):
        feature_map = {}
        inner = getattr(self, "tracker", None)
        if inner is None:
            return feature_map

        for obj in getattr(inner, "active_tracks", []):
            tid = int(getattr(obj, "id", -1))
            if tid < 0:
                continue
            feat = getattr(obj, "emb", None)
            if feat is None:
                continue
            try:
                feat = np.asarray(feat, dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if feat.size > 0:
                feature_map[tid] = feat

        return feature_map
