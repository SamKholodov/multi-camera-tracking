"""Shared helpers for duplicate overlapping track analysis."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


def iou_xyxy(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def load_mot_by_frame(path: Path) -> dict[int, list[tuple[int, list[float], float]]]:
    by_frame: dict[int, list[tuple[int, list[float], float]]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6])
            by_frame[frame].append((tid, [x, y, x + w, y + h], conf))
    return by_frame


def load_track_stats(path: Path) -> dict[int, dict]:
    stats: dict[int, dict] = defaultdict(lambda: {"frames": [], "confs": []})
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame, tid = int(parts[0]), int(parts[1])
            conf = float(parts[6])
            stats[tid]["frames"].append(frame)
            stats[tid]["confs"].append(conf)
    for tid in stats:
        frames = stats[tid]["frames"]
        confs = stats[tid]["confs"]
        stats[tid]["start"] = min(frames)
        stats[tid]["end"] = max(frames)
        stats[tid]["len"] = len(frames)
        stats[tid]["mean_conf"] = sum(confs) / len(confs)
        stats[tid]["frame_set"] = set(frames)
    return stats


def local_to_global(
    per_cam_local_dir: Path,
    cam: int,
    local_tid: int,
    frame: int,
) -> int | None:
    local_path = per_cam_local_dir / f"c{cam:03d}.txt"
    global_path = per_cam_local_dir.parent / "per_cam" / f"c{cam:03d}.txt"
    local_box = None
    with local_path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if int(parts[0]) != frame or int(parts[1]) != local_tid:
                continue
            x, y, w, h = map(float, parts[2:6])
            local_box = (x, y, w, h)
            break
    if local_box is None:
        return None
    lx, ly, lw, lh = local_box
    with global_path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if int(parts[0]) != frame:
                continue
            x, y, w, h = map(float, parts[2:6])
            if (
                abs(x - lx) < 0.05
                and abs(y - ly) < 0.05
                and abs(w - lw) < 0.05
                and abs(h - lh) < 0.05
            ):
                return int(parts[1])
    return None


@dataclass
class DuplicatePair:
    cam: int
    local_tid1: int
    local_tid2: int
    global_tid1: int
    global_tid2: int
    overlap_frames: int
    mean_iou: float
    loser_global: int
    winner_global: int


def find_duplicate_pairs(
    run_dir: Path,
    *,
    iou_thresh: float = 0.5,
    min_overlap_frames: int = 3,
) -> list[DuplicatePair]:
    per_cam_local = run_dir / "per_cam_local"
    pairs_out: list[DuplicatePair] = []

    for cam_file in sorted(per_cam_local.glob("c*.txt")):
        cam = int(cam_file.stem[1:])
        by_frame = load_mot_by_frame(cam_file)
        stats = load_track_stats(cam_file)

        pair_frames: dict[tuple[int, int], list[float]] = defaultdict(list)
        for frame, dets in by_frame.items():
            for i in range(len(dets)):
                for j in range(i + 1, len(dets)):
                    tid1, box1, _ = dets[i]
                    tid2, box2, _ = dets[j]
                    if tid1 == tid2:
                        continue
                    iv = iou_xyxy(box1, box2)
                    if iv >= iou_thresh:
                        key = (min(tid1, tid2), max(tid1, tid2))
                        pair_frames[key].append(iv)

        for (t1, t2), ious in pair_frames.items():
            if len(ious) < min_overlap_frames:
                continue
            co_frames = sorted(stats[t1]["frame_set"] & stats[t2]["frame_set"])
            ref = co_frames[0]
            g1 = local_to_global(per_cam_local, cam, t1, ref)
            g2 = local_to_global(per_cam_local, cam, t2, ref)
            if g1 is None or g2 is None or g1 == g2:
                continue

            s1, s2 = stats[t1], stats[t2]
            if (s1["len"], s1["mean_conf"]) >= (s2["len"], s2["mean_conf"]):
                winner, loser = g1, g2
            else:
                winner, loser = g2, g1

            pairs_out.append(
                DuplicatePair(
                    cam=cam,
                    local_tid1=t1,
                    local_tid2=t2,
                    global_tid1=g1,
                    global_tid2=g2,
                    overlap_frames=len(ious),
                    mean_iou=sum(ious) / len(ious),
                    loser_global=loser,
                    winner_global=winner,
                )
            )

    pairs_out.sort(key=lambda p: (-p.overlap_frames, -p.mean_iou))
    return pairs_out


def global_ids_to_remove(pairs: list[DuplicatePair]) -> dict[int, set[int]]:
    remove: dict[int, set[int]] = defaultdict(set)
    for pair in pairs:
        remove[pair.cam].add(pair.loser_global)
    return remove
