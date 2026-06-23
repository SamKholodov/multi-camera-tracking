from typing import Any, Optional, Type

import numpy as np

from core.mot.appearance import normalize_appearance_mode, normalized_for_matching


def _import_boxmot_botsort():
    """BoxMOT moved BotSort between releases; try known entry points."""
    candidates = (
        "boxmot.trackers.bbox.botsort.botsort",  # boxmot >= 19
        "boxmot.trackers.botsort.botsort",  # older layout (e.g. system site-packages)
        "boxmot.trackers",  # re-export
        "boxmot",
    )
    for module_path in candidates:
        try:
            mod = __import__(module_path, fromlist=["BotSort"])
            cls = getattr(mod, "BotSort", None)
            if cls is not None:
                return cls
        except ImportError:
            continue
    return None


_BoxmotBotSort = _import_boxmot_botsort()


def _finite_feature_vector(feat) -> np.ndarray | None:
    try:
        arr = np.asarray(feat, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    return arr


class BotSortTracker:
    def __init__(
        self,
        reid_model=None,
        use_default_reid=True,
        custom_reid_extractor=None,
        appearance_update: str = "aaf",
        **tracker_kwargs,
    ):
        self.use_default_reid = bool(use_default_reid)
        self.custom_reid_extractor = custom_reid_extractor
        self.appearance_update = normalize_appearance_mode(appearance_update)

        common = dict(
            track_high_thresh=0.5,
            track_low_thresh=0.15,
            new_track_thresh=0.4,
            track_buffer=300,
            match_thresh=0.7,
            proximity_thresh=0.7,
            appearance_thresh=0.35,
            with_reid=self.use_default_reid,
        )
        common.update(tracker_kwargs)

        if _BoxmotBotSort is None:
            raise ImportError(
                "BotSort requires the 'boxmot' package. Install it with: pip install boxmot\n"
                "Or switch tracker.type to 'deepocsort' in your YAML (no boxmot needed)."
            )

        if self.use_default_reid:
            if reid_model is None:
                raise ValueError(
                    "BotSort with use_default_reid=true requires a ReID model. "
                    "Set tracker.reid_weights or use_default_reid: false."
                )
            self.tracker = _BoxmotBotSort(reid_model=reid_model, **common)
        else:
            self.tracker = _BoxmotBotSort(**common)

    @staticmethod
    def _normalize_embeddings(features, expected_count):
        if features is None:
            return None

        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != expected_count:
            return None

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        out = features / norms
        bad = ~np.isfinite(out).all(axis=1)
        if bad.any():
            out = out.copy()
            out[bad] = 0.0
        return out

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

        if len(dets) > 0:
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

    def _iter_active_track_objects(self):
        inner = getattr(self, "tracker", None)
        if inner is None:
            return
        for attr in ("active_tracks", "tracked_stracks", "stracks", "tracks"):
            obj_list = getattr(inner, attr, None)
            if not obj_list:
                continue
            for obj in obj_list:
                tid = None
                for id_attr in ("id", "track_id", "tid"):
                    if hasattr(obj, id_attr):
                        tid = int(getattr(obj, id_attr))
                        break
                if tid is not None:
                    yield tid, obj
            return

    @staticmethod
    def _extract_track_feature(obj):
        for feat_attr in ("curr_feat", "smooth_feat", "feat", "features"):
            if not hasattr(obj, feat_attr):
                continue
            value = getattr(obj, feat_attr)
            if value is None:
                continue
            if isinstance(value, list) and len(value) > 0:
                value = value[-1]
            feat = _finite_feature_vector(value)
            if feat is not None:
                return feat
        return None

    def get_track_feature_map(self):
        feature_map = {}
        for tid, obj in self._iter_active_track_objects():
            feat = self._extract_track_feature(obj)
            if feat is None:
                continue
            feature_map[tid] = normalized_for_matching(feat, self.appearance_update)
        return feature_map

    def get_track_appearance_raw_map(self):
        feature_map = {}
        for tid, obj in self._iter_active_track_objects():
            feat = self._extract_track_feature(obj)
            if feat is not None:
                feature_map[tid] = feat.copy()
        return feature_map
