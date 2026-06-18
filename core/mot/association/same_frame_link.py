from __future__ import annotations

from core.io.calibration import world_distance
from core.mot.appearance import cosine_distance, normalized_for_matching


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def components(self) -> list[list[int]]:
        groups: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            groups.setdefault(root, []).append(i)
        return list(groups.values())


def cross_cam_reid_distance(
    feat_a,
    feat_b,
    *,
    appearance_mode: str,
) -> float | None:
    if feat_a is None or feat_b is None:
        return None
    qa = normalized_for_matching(feat_a, appearance_mode)
    qb = normalized_for_matching(feat_b, appearance_mode)
    return cosine_distance(qa, qb)


def find_same_frame_links(
    detections: list[tuple[int, int, tuple[float, float] | None, object]],
    *,
    t_min_m: float,
    reid_threshold: float,
    metric: str,
    appearance_mode: str,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Return cross-camera (cam, local_tid) pairs linked on the same frame."""
    links: list[tuple[tuple[int, int], tuple[int, int]]] = []
    n = len(detections)
    for i in range(n):
        cam_a, tid_a, wpt_a, feat_a = detections[i]
        for j in range(i + 1, n):
            cam_b, tid_b, wpt_b, feat_b = detections[j]
            if cam_a == cam_b:
                continue
            if wpt_a is None or wpt_b is None:
                continue
            dist_m = world_distance(wpt_a, wpt_b, metric=metric)
            if dist_m >= t_min_m:
                continue
            reid_d = cross_cam_reid_distance(
                feat_a, feat_b, appearance_mode=appearance_mode
            )
            if reid_d is None or reid_d >= reid_threshold:
                continue
            links.append(((cam_a, tid_a), (cam_b, tid_b)))
    return links
