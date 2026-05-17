from collections import defaultdict

import numpy as np


class TrackletSaver:
    def __init__(self):
        self.tracklets = defaultdict(
            lambda: {
                "frames": [],
                "boxes": [],
                "footpoints": [],
                "centers": [],
                "embeddings": [],
                "start_frame": None,
                "end_frame": None,
            }
        )

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            return vec
        return vec / norm

    def update(self, frame_id, tracks, track_feature_map=None):
        if track_feature_map is None:
            track_feature_map = {}
        if tracks is None:
            return

        for tr in tracks:
            tr = np.asarray(tr).reshape(-1)
            if tr.size < 5:
                continue

            x1, y1, x2, y2 = map(float, tr[:4])
            tid = int(tr[4])

            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            foot_x = cx
            foot_y = y2

            item = self.tracklets[tid]
            item["frames"].append(int(frame_id))
            item["boxes"].append([x1, y1, x2, y2])
            item["centers"].append([cx, cy])
            item["footpoints"].append([foot_x, foot_y])

            if item["start_frame"] is None:
                item["start_frame"] = int(frame_id)
            item["end_frame"] = int(frame_id)

            feat = track_feature_map.get(tid, None)
            if feat is not None:
                feat = np.asarray(feat, dtype=np.float32).reshape(-1)
                if feat.size > 0:
                    feat = self._l2_normalize(feat)
                    item["embeddings"].append(feat)

    def finalize(self):
        for item in self.tracklets.values():
            item["frames"] = np.asarray(item["frames"], dtype=np.int32)
            item["boxes"] = np.asarray(item["boxes"], dtype=np.float32)
            item["centers"] = np.asarray(item["centers"], dtype=np.float32)
            item["footpoints"] = np.asarray(item["footpoints"], dtype=np.float32)

            if len(item["embeddings"]) > 0:
                item["embeddings"] = np.asarray(item["embeddings"], dtype=np.float32)
                item["mean_embedding"] = item["embeddings"].mean(axis=0).astype(np.float32)
                norm = np.linalg.norm(item["mean_embedding"])
                if norm > 1e-12:
                    item["mean_embedding"] /= norm
            else:
                item["embeddings"] = np.empty((0,), dtype=np.float32)
                item["mean_embedding"] = np.empty((0,), dtype=np.float32)

            if len(item["boxes"]) > 0:
                wh = item["boxes"][:, 2:4] - item["boxes"][:, 0:2]
                item["mean_box_wh"] = wh.mean(axis=0).astype(np.float32)
            else:
                item["mean_box_wh"] = np.zeros(2, dtype=np.float32)

            item["length"] = int(len(item["frames"]))

    def save(self, path):
        np.savez_compressed(path, tracklets=dict(self.tracklets))
