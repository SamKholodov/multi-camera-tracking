"""Build cross-camera global ID demonstration frames (correct=green, error=red).

Finds 2-3 sync moments per dataset where the MCMT pipeline correctly links
vehicles across cameras (same pred global_id) and shows one association error.

Usage:
    python scripts/build_mcmt_association_demo.py --dataset both --num-examples 3
    python scripts/build_mcmt_association_demo.py --dataset gta \\
        --run-dir outputs/configs_gta/geo_ablation/geo_tight
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.eval.cityflow_protocol import cross_camera_gt_ids
from core.io.calibration import (
    load_homography_image_to_world,
    project_bbox_bottom_center,
    world_distance,
)
from core.io.gta_mcmt import GtaMcmtDataset, image_path_for_cam_dir
from scripts.cityflow_ablation_common import GT_ROOT as CF_GT_ROOT, S02_CAM_IDS
from scripts.cityflow_sync_eval import apply_sync_alignment
from scripts.duplicate_track_utils import iou_xyxy, load_mot_by_frame
from scripts.eval_s02 import load_mot
from scripts.visualize_gta_mcmt import clip_box, stack_camera_views

DEFAULT_RUNS = {
    "gta": _ROOT / "outputs/configs_gta/geo_ablation/geo_tight",
    "cityflow": _ROOT / "outputs/cityflow_ablation_yolo26m/temporal_N100",
}

GTA_CAMERAS = [0, 1, 2, 3]
GTA_GT_ROOT = _ROOT / "datasets/gta_mcmt"

CF_VIDEO = {
    6: CF_GT_ROOT / "c006/vdo_synch.avi",
    7: CF_GT_ROOT / "c007/vdo_synch.avi",
    8: CF_GT_ROOT / "c008/vdo_synch.avi",
    9: CF_GT_ROOT / "c009/vdo_synch.avi",
}

COLOR_CORRECT = (0, 255, 0)
COLOR_ERROR = (0, 0, 255)
COLOR_LABEL = (255, 255, 255)
COLOR_LABEL_OUTLINE = (0, 0, 0)

SUPPLEMENT_GLOBAL_BASE = 900_000
CF_GT_SHRINK = 0.85
GTA_GT_SHRINK = 0.85
CF_FRAME_WIDTH = 1920.0
CF_FRAME_HEIGHT = 1080.0

GEO_METRIC = {"gta": "plane", "cityflow": "gps"}
MAX_ERROR_GT_PAIR_M = {"gta": 12.0, "cityflow": 30.0}
MAX_ERROR_PRED_GT_M = {"gta": 4.0, "cityflow": 15.0}

LEGEND_CORRECT = "Корректная ассоциация"
LEGEND_ERROR = "Ошибка ассоциации"
LEGEND_ID_HELP = "L — локальный ID, G — глобальный ID"


@dataclass
class TrackMatch:
    cam: int
    frame: int
    local_id: int
    global_id: int
    gt_id: int
    bbox_xyxy: list[float]
    iou: float

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def min_side(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return min(x2 - x1, y2 - y1)


@dataclass
class ClassifiedMatch(TrackMatch):
    status: str = "correct"  # correct | error
    bbox_source: str = "pred"  # pred | gt_shrunk


@dataclass
class FrameCandidate:
    frame: int
    matches: list[ClassifiedMatch]
    correct_gt_ids: list[int] = field(default_factory=list)
    error_gt_ids: list[int] = field(default_factory=list)
    error_global_id: int | None = None
    score: float = 0.0
    gt_supplemented: bool = False
    correct_cams: list[int] = field(default_factory=list)
    error_cams: list[int] = field(default_factory=list)
    error_cam: int | None = None  # primary error cam (legacy); use error_cams
    error_gt_pair_dist_m: float | None = None
    error_max_gt_pair_dist_m: float | None = None
    error_pred_gt_dist_m: float | None = None

    @property
    def n_correct(self) -> int:
        return len(self.correct_gt_ids)

    @property
    def n_error(self) -> int:
        return len(self.error_gt_ids)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["gta", "cityflow", "both"], default="both")
    ap.add_argument("--num-examples", type=int, default=6)
    ap.add_argument("--scan-stride", type=int, default=None)
    ap.add_argument("--min-correct", type=int, default=4)
    ap.add_argument("--max-correct", type=int, default=5)
    ap.add_argument("--min-error", type=int, default=1)
    ap.add_argument("--min-sync-gap", type=int, default=None)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "outputs/demos/mcmt_association")
    ap.add_argument("--run-dir", type=Path, default=None, help="Override run dir (single dataset only)")
    ap.add_argument(
        "--cameras",
        type=int,
        nargs="+",
        default=None,
        metavar="CAM",
        help="Restrict CityFlow demo to these cameras (e.g. 6 7)",
    )
    ap.add_argument(
        "--example-name",
        type=str,
        default=None,
        help="Output subdir name instead of example_NNN (e.g. cam67_pair)",
    )
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def tlwh_to_xyxy(x: float, y: float, w: float, h: float) -> list[float]:
    return [x, y, x + w, y + h]


def bbox_ok(area: float, min_side: float, *, min_area: float, min_side_px: float) -> bool:
    return area >= min_area and min_side >= min_side_px


def global_id_for_box(
    global_by_frame: dict[int, list[tuple[int, list[float], float]]],
    frame: int,
    box_xyxy: list[float],
) -> int | None:
    for gid, gbox, _ in global_by_frame.get(frame, []):
        if (
            abs(gbox[0] - box_xyxy[0]) < 0.05
            and abs(gbox[1] - box_xyxy[1]) < 0.05
            and abs(gbox[2] - box_xyxy[2]) < 0.05
            and abs(gbox[3] - box_xyxy[3]) < 0.05
        ):
            return gid
    best_gid, best_iou = None, 0.0
    for gid, gbox, _ in global_by_frame.get(frame, []):
        iv = iou_xyxy(box_xyxy, gbox)
        if iv > best_iou:
            best_iou, best_gid = iv, gid
    return best_gid if best_iou >= 0.95 else None


def greedy_match_frame(
    gt_rows: list[tuple[int, list[float], float]],
    local_rows: list[tuple[int, list[float], float]],
    global_by_frame: dict[int, list[tuple[int, list[float], float]]],
    *,
    cam: int,
    frame: int,
    iou_thresh: float,
) -> list[TrackMatch]:
    matches: list[TrackMatch] = []
    used_gt: set[int] = set()
    for local_tid, pred_box, _conf in sorted(local_rows, key=lambda r: -(r[1][2] - r[1][0]) * (r[1][3] - r[1][1])):
        gid = global_id_for_box(global_by_frame, frame, pred_box)
        if gid is None:
            continue
        best_gt, best_iou = None, 0.0
        for gt_id, gt_box, _ in gt_rows:
            if gt_id in used_gt:
                continue
            iv = iou_xyxy(pred_box, gt_box)
            if iv > best_iou:
                best_iou, best_gt = iv, gt_id
        if best_gt is None or best_iou < iou_thresh:
            continue
        used_gt.add(best_gt)
        matches.append(
            TrackMatch(
                cam=cam,
                frame=frame,
                local_id=local_tid,
                global_id=gid,
                gt_id=best_gt,
                bbox_xyxy=pred_box,
                iou=best_iou,
            )
        )
    return matches


def classify_cross_camera(matches: list[TrackMatch]) -> list[ClassifiedMatch]:
    by_gt: dict[int, list[TrackMatch]] = {}
    by_gid: dict[int, list[TrackMatch]] = {}
    for m in matches:
        by_gt.setdefault(m.gt_id, []).append(m)
        by_gid.setdefault(m.global_id, []).append(m)

    status: dict[tuple[int, int], str] = {}

    for gt_id, group in by_gt.items():
        cams = {m.cam for m in group}
        if len(cams) < 2:
            for m in group:
                status[(m.cam, m.local_id)] = "correct"
            continue
        gids = {m.global_id for m in group}
        st = "correct" if len(gids) == 1 else "error"
        for m in group:
            status[(m.cam, m.local_id)] = st

    for gid, group in by_gid.items():
        gt_ids = {m.gt_id for m in group}
        if len(gt_ids) <= 1:
            continue
        cams = {m.cam for m in group}
        if len(cams) < 2:
            continue
        for m in group:
            status[(m.cam, m.local_id)] = "error"

    out: list[ClassifiedMatch] = []
    for m in matches:
        st = status.get((m.cam, m.local_id), "correct")
        out.append(ClassifiedMatch(**m.__dict__, status=st))
    return out


def load_gta_data(
    run_dir: Path,
) -> tuple[dict[int, np.ndarray], dict[int, dict], dict[int, dict], dict[int, np.ndarray], int]:
    gt_by_cam: dict[int, np.ndarray] = {}
    local_by_cam: dict[int, dict] = {}
    global_by_cam: dict[int, dict] = {}
    homos: dict[int, np.ndarray] = {}
    max_frame = 0
    for cam in GTA_CAMERAS:
        gt = load_mot(GTA_GT_ROOT / f"cam-{cam}/gt/gt.txt")
        gt_by_cam[cam] = gt
        local_path = run_dir / "per_cam_local" / f"c{cam:03d}.txt"
        global_path = run_dir / "per_cam" / f"c{cam:03d}.txt"
        local_by_cam[cam] = load_mot_by_frame(local_path)
        global_by_cam[cam] = load_mot_by_frame(global_path)
        homos[cam] = load_homography_image_to_world(GTA_GT_ROOT / f"cam-{cam}/calibration.txt")
        for src in (gt,):
            if len(src):
                max_frame = max(max_frame, int(src[:, 0].max()))
    return gt_by_cam, local_by_cam, global_by_cam, homos, max_frame


def load_cityflow_data(
    run_dir: Path,
) -> tuple[dict[int, np.ndarray], dict[int, dict], dict[int, dict], dict[int, np.ndarray], int]:
    gt_by_cam: dict[int, np.ndarray] = {}
    pr_local_by_cam: dict[int, np.ndarray] = {}
    pr_global_by_cam: dict[int, np.ndarray] = {}
    for cam in S02_CAM_IDS:
        gt_by_cam[cam] = load_mot(CF_GT_ROOT / f"c{cam:03d}/gt/gt.txt")
        pr_local_by_cam[cam] = load_mot(run_dir / "per_cam_local" / f"c{cam:03d}.txt")
        pr_global_by_cam[cam] = load_mot(run_dir / "per_cam" / f"c{cam:03d}.txt")

    gt_by_cam, pr_local_by_cam, _ = apply_sync_alignment(gt_by_cam, pr_local_by_cam, CF_GT_ROOT)
    _, pr_global_by_cam, manifest = apply_sync_alignment(gt_by_cam, pr_global_by_cam, CF_GT_ROOT)
    sync_len = int(manifest.get("sync_length_frames", 1920)) if manifest else 1920

    local_by_cam = {cam: load_mot_by_frame_from_array(pr_local_by_cam[cam]) for cam in S02_CAM_IDS}
    global_by_cam = {cam: load_mot_by_frame_from_array(pr_global_by_cam[cam]) for cam in S02_CAM_IDS}
    homos = {
        cam: load_homography_image_to_world(CF_GT_ROOT / f"c{cam:03d}/calibration.txt")
        for cam in S02_CAM_IDS
    }
    return gt_by_cam, local_by_cam, global_by_cam, homos, sync_len


def load_mot_by_frame_from_array(data: np.ndarray) -> dict[int, list[tuple[int, list[float], float]]]:
    by_frame: dict[int, list[tuple[int, list[float], float]]] = {}
    if len(data) == 0:
        return by_frame
    for row in data:
        frame = int(row[0])
        tid = int(row[1])
        x, y, w, h = map(float, row[2:6])
        conf = float(row[6])
        by_frame.setdefault(frame, []).append((tid, tlwh_to_xyxy(x, y, w, h), conf))
    return by_frame


def gt_rows_at_frame(gt: np.ndarray, frame: int) -> list[tuple[int, list[float], float]]:
    if len(gt) == 0:
        return []
    rows = gt[gt[:, 0].astype(int) == frame]
    out: list[tuple[int, list[float], float]] = []
    for row in rows:
        x, y, w, h = map(float, row[2:6])
        out.append((int(row[1]), tlwh_to_xyxy(x, y, w, h), float(row[6])))
    return out


def error_global_clusters(
    classified: list[ClassifiedMatch],
    *,
    min_area: float,
    min_side_px: float,
) -> list[tuple[int, list[ClassifiedMatch]]]:
    """Groups where one global_id is wrongly shared by different GT objects on 2+ cameras."""
    by_gid: dict[int, list[ClassifiedMatch]] = {}
    for m in classified:
        if m.status != "error":
            continue
        if not bbox_ok(m.area, m.min_side, min_area=min_area, min_side_px=min_side_px):
            continue
        by_gid.setdefault(m.global_id, []).append(m)

    clusters: list[tuple[int, list[ClassifiedMatch]]] = []
    for gid, group in by_gid.items():
        if len({m.gt_id for m in group}) < 2:
            continue
        if len({m.cam for m in group}) < 2:
            continue
        clusters.append((gid, group))
    return clusters


def gt_box_at_frame(
    gt_by_cam: dict[int, np.ndarray],
    cam: int,
    frame: int,
    gt_id: int,
) -> list[float] | None:
    for gid, box, _ in gt_rows_at_frame(gt_by_cam[cam], frame):
        if gid == gt_id:
            return box
    return None


def gt_world_bottom(
    frame: int,
    cam: int,
    gt_id: int,
    gt_by_cam: dict[int, np.ndarray],
    homos: dict[int, np.ndarray],
) -> tuple[float, float] | None:
    box = gt_box_at_frame(gt_by_cam, cam, frame, gt_id)
    if box is None:
        return None
    return project_bbox_bottom_center(homos[cam], *box)


def pred_world_bottom(
    match: ClassifiedMatch,
    homos: dict[int, np.ndarray],
) -> tuple[float, float]:
    return project_bbox_bottom_center(homos[match.cam], *match.bbox_xyxy)


def error_cluster_geo_metrics(
    frame: int,
    group: list[ClassifiedMatch],
    gt_by_cam: dict[int, np.ndarray],
    homos: dict[int, np.ndarray],
    *,
    metric: str,
) -> dict[str, float] | None:
    """World-space distances between GT objects wrongly merged under one global id."""
    gt_ids = sorted({m.gt_id for m in group})
    if len(gt_ids) < 2:
        return None

    rep_by_gt: dict[int, tuple[float, float]] = {}
    for gt_id in gt_ids:
        pts: list[tuple[float, float]] = []
        for m in group:
            if m.gt_id != gt_id:
                continue
            wpt = gt_world_bottom(frame, m.cam, gt_id, gt_by_cam, homos)
            if wpt is not None:
                pts.append(wpt)
        if not pts:
            for cam in gt_by_cam:
                wpt = gt_world_bottom(frame, cam, gt_id, gt_by_cam, homos)
                if wpt is not None:
                    pts.append(wpt)
        if pts:
            rep_by_gt[gt_id] = (
                float(np.mean([p[0] for p in pts])),
                float(np.mean([p[1] for p in pts])),
            )

    if len(rep_by_gt) < 2:
        return None

    pair_dists = [
        world_distance(rep_by_gt[a], rep_by_gt[b], metric=metric)
        for a, b in combinations(sorted(rep_by_gt), 2)
    ]
    pred_gt_dists: list[float] = []
    for m in group:
        gt_w = gt_world_bottom(frame, m.cam, m.gt_id, gt_by_cam, homos)
        if gt_w is None:
            continue
        pred_w = pred_world_bottom(m, homos)
        pred_gt_dists.append(world_distance(gt_w, pred_w, metric=metric))

    return {
        "min_gt_pair_m": float(min(pair_dists)),
        "max_gt_pair_m": float(max(pair_dists)),
        "mean_pred_gt_m": float(np.mean(pred_gt_dists)) if pred_gt_dists else float("inf"),
    }


def error_cluster_passes_geo_filter(
    metrics: dict[str, float],
    *,
    dataset_name: str,
) -> bool:
    return (
        metrics["max_gt_pair_m"] <= MAX_ERROR_GT_PAIR_M[dataset_name]
        and metrics["mean_pred_gt_m"] <= MAX_ERROR_PRED_GT_M[dataset_name]
    )


def error_cluster_rank_score(
    group: list[ClassifiedMatch],
    metrics: dict[str, float],
) -> float:
    """Higher = better demo: confused GT objects are spatial neighbours."""
    min_d = metrics["min_gt_pair_m"]
    max_d = metrics["max_gt_pair_m"]
    mean_pred_gt = metrics["mean_pred_gt_m"]
    n_cams = len({m.cam for m in group})
    total_area = sum(m.area for m in group)
    score = 3e6 / (1.0 + min_d) + 1e6 / (1.0 + max_d)
    score += 2e4 / (1.0 + mean_pred_gt)
    score += n_cams * 1e4 + total_area * 0.01
    return score


def rank_error_clusters(
    frame: int,
    clusters: list[tuple[int, list[ClassifiedMatch]]],
    gt_by_cam: dict[int, np.ndarray],
    homos: dict[int, np.ndarray],
    *,
    dataset_name: str,
) -> list[tuple[int, list[ClassifiedMatch], dict[str, float], float]]:
    metric = GEO_METRIC[dataset_name]
    ranked: list[tuple[int, list[ClassifiedMatch], dict[str, float], float]] = []
    for gid, group in clusters:
        metrics = error_cluster_geo_metrics(
            frame, group, gt_by_cam, homos, metric=metric
        )
        if metrics is None:
            continue
        if not error_cluster_passes_geo_filter(metrics, dataset_name=dataset_name):
            continue
        ranked.append((gid, group, metrics, error_cluster_rank_score(group, metrics)))
    ranked.sort(key=lambda item: item[3], reverse=True)
    return ranked


def gt_views_for_id(
    frame: int,
    gt_id: int,
    gt_by_cam: dict[int, np.ndarray],
    cameras: list[int],
    *,
    min_area: float,
    min_side_px: float,
) -> list[tuple[int, list[float]]]:
    views: list[tuple[int, list[float]]] = []
    for cam in cameras:
        for gid, box, _ in gt_rows_at_frame(gt_by_cam[cam], frame):
            if gid != gt_id:
                continue
            x1, y1, x2, y2 = box
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            min_side = min(x2 - x1, y2 - y1)
            if bbox_ok(area, min_side, min_area=min_area, min_side_px=min_side_px):
                views.append((cam, box))
            break
    return views


def canonical_global_id_for_gt(classified: list[ClassifiedMatch], gt_id: int) -> int:
    gids = {m.global_id for m in classified if m.gt_id == gt_id and m.status == "correct"}
    if len(gids) == 1:
        return next(iter(gids))
    if gids:
        return max(gids)
    return SUPPLEMENT_GLOBAL_BASE + gt_id


def build_correct_matches_for_gt(
    frame: int,
    gt_id: int,
    classified: list[ClassifiedMatch],
    gt_by_cam: dict[int, np.ndarray],
    cameras: list[int],
    *,
    min_area: float,
    min_side_px: float,
) -> list[ClassifiedMatch]:
    """Pred matches when available; otherwise GT boxes with a shared synthetic global id."""
    global_id = canonical_global_id_for_gt(classified, gt_id)
    pred_by_cam = {
        m.cam: m for m in classified if m.gt_id == gt_id and m.status == "correct"
    }
    out: list[ClassifiedMatch] = []
    for cam, bbox in gt_views_for_id(
        frame, gt_id, gt_by_cam, cameras, min_area=min_area, min_side_px=min_side_px
    ):
        if cam in pred_by_cam:
            out.append(pred_by_cam[cam])
            continue
        out.append(
            ClassifiedMatch(
                cam=cam,
                frame=frame,
                gt_id=gt_id,
                local_id=gt_id * 100 + cam,
                global_id=global_id,
                bbox_xyxy=bbox,
                iou=1.0,
                status="correct",
            )
        )
    return out


def shrink_xyxy(box: list[float], scale: float = CF_GT_SHRINK) -> list[float]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def is_synthetic_gt_match(match: ClassifiedMatch) -> bool:
    return match.local_id == match.gt_id * 100 + match.cam


def gt_match_from_box(
    *,
    frame: int,
    cam: int,
    gt_id: int,
    box: list[float],
    global_id: int,
    shrink: float = CF_GT_SHRINK,
) -> ClassifiedMatch:
    shrunk = shrink_xyxy(box, shrink)
    bbox_source = "gt" if shrink >= 0.999 else "gt_shrunk"
    return ClassifiedMatch(
        cam=cam,
        frame=frame,
        gt_id=gt_id,
        local_id=gt_id * 100 + cam,
        global_id=global_id,
        bbox_xyxy=shrunk,
        iou=1.0,
        status="correct",
        bbox_source=bbox_source,
    )


def edge_penalty(
    box: list[float],
    img_w: float = CF_FRAME_WIDTH,
    img_h: float = CF_FRAME_HEIGHT,
) -> float:
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    penalty = 0.0
    if x2 >= img_w - 8:
        penalty += 0.35
    if x1 <= 8:
        penalty += 0.15
    if w / img_w > 0.55:
        penalty += 0.25
    if h / img_h > 0.45:
        penalty += 0.1
    return penalty


def match_quality(
    match: ClassifiedMatch,
    *,
    img_w: float = CF_FRAME_WIDTH,
    img_h: float = CF_FRAME_HEIGHT,
) -> float:
    base = match.iou if match.bbox_source == "pred" else 0.55
    return base - edge_penalty(match.bbox_xyxy, img_w, img_h)


def pred_correct_by_gt(
    classified: list[ClassifiedMatch],
    error_gt_ids: set[int],
    benchmark_gt_ids: set[int],
    *,
    min_area: float,
    min_side_px: float,
) -> dict[int, list[ClassifiedMatch]]:
    pool: dict[int, list[ClassifiedMatch]] = {}
    for m in classified:
        if m.status != "correct":
            continue
        if m.gt_id in error_gt_ids:
            continue
        if m.gt_id not in benchmark_gt_ids:
            continue
        if not bbox_ok(m.area, m.min_side, min_area=min_area, min_side_px=min_side_px):
            continue
        pool.setdefault(m.gt_id, []).append(m)
    for gt_id in pool:
        pool[gt_id].sort(key=lambda mm: match_quality(mm), reverse=True)
    return pool


def pick_error_on_separate_cam(
    group: list[ClassifiedMatch],
    avoid_cams: set[int],
) -> ClassifiedMatch | None:
    separate = [m for m in group if m.cam not in avoid_cams]
    pool = separate if separate else group
    return max(pool, key=lambda m: match_quality(m))


def score_cityflow_layout(
    correct: list[ClassifiedMatch],
    error: ClassifiedMatch | list[ClassifiedMatch],
    *,
    pred_only: bool,
) -> float:
    errors = error if isinstance(error, list) else [error]
    correct_cams = {m.cam for m in correct}
    error_cams = {m.cam for m in errors}
    separate_error = error_cams.isdisjoint(correct_cams)
    iou_sum = sum(m.iou for m in correct if m.bbox_source == "pred")
    quality = sum(match_quality(m) for m in correct) + sum(match_quality(m) for m in errors)
    per_gt = {m.gt_id: [x for x in correct if x.gt_id == m.gt_id] for m in correct}
    cross_cam_vehicles = sum(1 for group in per_gt.values() if len({x.cam for x in group}) >= 2)
    score = quality * 1e4 + iou_sum * 100 + cross_cam_vehicles * 2e5
    overlap_cams = correct_cams & error_cams
    score += len(overlap_cams) * 5e5
    score += sum(1 for m in correct if m.cam in error_cams) * 2e5
    if len(errors) >= 2:
        score += 1e5
    if pred_only:
        score += 5e5
    if separate_error:
        score += 2e5
    if len(correct_cams) == 2:
        score += 3e5
    elif len(correct_cams) == 3:
        score += 1e3
    if cross_cam_vehicles >= 3:
        score += 6e5
    elif cross_cam_vehicles >= 2:
        score += 3e5
    return score


def demo_cam_order(correct_cams: list[int], error_cams: list[int]) -> list[int]:
    order = list(correct_cams)
    seen = set(correct_cams)
    for cam in sorted(error_cams):
        if cam not in seen:
            order.append(cam)
            seen.add(cam)
    return order


def cross_cam_gt_at_frame(
    frame: int,
    gt_by_cam: dict[int, np.ndarray],
    cameras: list[int],
    benchmark_gt_ids: set[int],
    error_gt_ids: set[int],
    exclude_cams: set[int],
    pred_by_gt: dict[int, list[ClassifiedMatch]] | None,
    *,
    min_area: float,
    min_side_px: float,
) -> list[tuple[int, list[tuple[int, list[float]]]]]:
    """Benchmark GT ids visible on >=2 cameras (GT and/or pred at this frame)."""
    ranked: list[tuple[tuple[int, int, int], int, list[tuple[int, list[float]]]]] = []
    for gt_id in benchmark_gt_ids:
        if gt_id in error_gt_ids:
            continue
        views = gt_views_for_id(
            frame, gt_id, gt_by_cam, cameras, min_area=min_area, min_side_px=min_side_px
        )
        views = [(cam, box) for cam, box in views if cam not in exclude_cams]
        view_cams = {cam for cam, _ in views}
        if pred_by_gt is not None:
            for m in pred_by_gt.get(gt_id, []):
                if m.cam in exclude_cams or m.cam in view_cams:
                    continue
                views.append((m.cam, list(m.bbox_xyxy)))
                view_cams.add(m.cam)
        if len(views) < 2:
            continue
        n_pred = len(pred_by_gt.get(gt_id, [])) if pred_by_gt else 0
        ranked.append(((len(views), n_pred, 0), gt_id, views))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [(gt_id, views) for _, gt_id, views in ranked]


def cross_cam_gt_ids_on_cams(
    frame: int,
    gt_by_cam: dict[int, np.ndarray],
    correct_cams: set[int],
    benchmark_gt_ids: set[int],
    error_gt_ids: set[int],
    *,
    min_area: float,
    min_side_px: float,
    require_all_cams: bool = False,
) -> list[tuple[int, list[tuple[int, list[float]]]]]:
    """GT ids visible on >=2 of the given cameras (or all, if require_all_cams)."""
    ranked: list[tuple[tuple[int, int, int], int, list[tuple[int, list[float]]]]] = []
    min_views = len(correct_cams) if require_all_cams else 2
    for gt_id in benchmark_gt_ids:
        if gt_id in error_gt_ids:
            continue
        views = gt_views_for_id(
            frame, gt_id, gt_by_cam, sorted(correct_cams), min_area=min_area, min_side_px=min_side_px
        )
        views = [(cam, box) for cam, box in views if cam in correct_cams]
        if len(views) < min_views:
            continue
        if require_all_cams and {cam for cam, _ in views} != correct_cams:
            continue
        ranked.append(((len(views), 0, 0), gt_id, views))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [(gt_id, views) for _, gt_id, views in ranked]


def demo_pair_error_groups(
    classified: list[ClassifiedMatch],
    demo_cams: set[int],
    *,
    min_area: float,
    min_side_px: float,
) -> list[tuple[int | None, list[ClassifiedMatch], str]]:
    """Cross-camera errors visible on demo cameras (wrong merge or split identity)."""
    groups: list[tuple[int | None, list[ClassifiedMatch], str]] = []
    seen_keys: set[tuple[int, int]] = set()

    for gid, group in error_global_clusters(
        classified, min_area=min_area, min_side_px=min_side_px
    ):
        demo_err = [m for m in group if m.cam in demo_cams]
        if not demo_err:
            continue
        key = tuple(sorted((m.cam, m.gt_id) for m in demo_err))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        groups.append((gid, demo_err, "wrong_merge"))

    by_gt: dict[int, list[ClassifiedMatch]] = {}
    for m in classified:
        if m.status != "error":
            continue
        if not bbox_ok(m.area, m.min_side, min_area=min_area, min_side_px=min_side_px):
            continue
        by_gt.setdefault(m.gt_id, []).append(m)

    for gt_id, group in by_gt.items():
        if len({m.cam for m in group}) < 2:
            continue
        if len({m.global_id for m in group}) <= 1:
            continue
        demo_err = [m for m in group if m.cam in demo_cams]
        if not demo_err:
            continue
        key = tuple(sorted((m.cam, m.gt_id) for m in demo_err))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        groups.append((demo_err[0].global_id, demo_err, "split_identity"))

    groups.sort(
        key=lambda item: (
            len(item[1]),
            sum(m.area for m in item[1]),
        ),
        reverse=True,
    )
    return groups


def fill_cam_pair_correct(
    shared_matches: list[ClassifiedMatch],
    classified: list[ClassifiedMatch],
    demo_cams: set[int],
    error_gt_ids: set[int],
    benchmark_gt_ids: set[int],
    *,
    target_per_cam: int,
    min_area: float,
    min_side_px: float,
) -> list[ClassifiedMatch]:
    selected = list(shared_matches)
    used_keys = {(m.cam, m.local_id) for m in selected}

    for cam in sorted(demo_cams):
        count = sum(1 for m in selected if m.cam == cam and m.status == "correct")
        pool = [
            m
            for m in classified
            if m.status == "correct"
            and m.cam == cam
            and m.gt_id not in error_gt_ids
            and m.gt_id in benchmark_gt_ids
            and (m.cam, m.local_id) not in used_keys
            and bbox_ok(m.area, m.min_side, min_area=min_area, min_side_px=min_side_px)
        ]
        pool.sort(key=lambda m: match_quality(m), reverse=True)
        for m in pool:
            if count >= target_per_cam:
                break
            selected.append(m)
            used_keys.add((m.cam, m.local_id))
            count += 1
    return selected


def shared_correct_gt_on_cams(
    matches: list[ClassifiedMatch],
    demo_cams: set[int],
) -> list[int]:
    """GT ids with a correct match on every demo camera sharing one global_id."""
    per_gt: dict[int, dict[int, ClassifiedMatch]] = {}
    for m in matches:
        if m.status != "correct" or m.cam not in demo_cams:
            continue
        per_gt.setdefault(m.gt_id, {})[m.cam] = m
    shared: list[int] = []
    for gt_id, by_cam in per_gt.items():
        if demo_cams - set(by_cam.keys()):
            continue
        if len({by_cam[c].global_id for c in demo_cams}) == 1:
            shared.append(gt_id)
    return sorted(shared)


def score_cam_pair_layout(
    correct: list[ClassifiedMatch],
    error: list[ClassifiedMatch],
    demo_cams: set[int],
    *,
    pred_only: bool,
) -> float:
    shared = shared_correct_gt_on_cams(correct, demo_cams)
    n_shared = len(shared)
    per_cam_green = {cam: sum(1 for m in correct if m.cam == cam and m.status == "correct") for cam in demo_cams}
    min_green = min(per_cam_green.values()) if per_cam_green else 0
    error_on_demo = sum(1 for m in error if m.cam in demo_cams)
    iou_sum = sum(m.iou for m in correct if m.bbox_source == "pred")
    quality = sum(match_quality(m) for m in correct) + sum(match_quality(m) for m in error)
    score = n_shared * 1e7 + min_green * 5e5 + quality * 1e4 + iou_sum * 100
    score += error_on_demo * 3e5
    if pred_only:
        score += 5e5
    if n_shared >= 4:
        score += 2e6
    elif n_shared >= 3:
        score += 1e6
    return score


def search_cam_pair_cityflow_layout(
    frame: int,
    error_group: list[ClassifiedMatch],
    pred_by_gt: dict[int, list[ClassifiedMatch]],
    gt_by_cam: dict[int, np.ndarray],
    classified: list[ClassifiedMatch],
    demo_cams: set[int],
    benchmark_gt_ids: set[int],
    error_gt_ids: set[int],
    *,
    min_area: float,
    min_side_px: float,
    min_shared_gt: int = 3,
    max_shared_gt: int = 4,
    target_per_cam: int = 3,
    gt_shrink: float = CF_GT_SHRINK,
) -> tuple[list[ClassifiedMatch], list[ClassifiedMatch], list[int], list[int], float, bool] | None:
    error_on_demo = [m for m in error_group if m.cam in demo_cams]
    if not error_on_demo:
        return None

    cross_gt = cross_cam_gt_ids_on_cams(
        frame,
        gt_by_cam,
        demo_cams,
        benchmark_gt_ids,
        error_gt_ids,
        min_area=min_area,
        min_side_px=min_side_px,
        require_all_cams=True,
    )
    cross_by_id = {gt_id: views for gt_id, views in cross_gt}
    for gt_id in shared_correct_gt_on_cams(
        [m for m in classified if m.status == "correct"],
        demo_cams,
    ):
        if gt_id in error_gt_ids or gt_id in cross_by_id:
            continue
        views = gt_views_for_id(
            frame, gt_id, gt_by_cam, sorted(demo_cams), min_area=min_area, min_side_px=min_side_px
        )
        views = [(cam, box) for cam, box in views if cam in demo_cams]
        pred_on = [m for m in pred_by_gt.get(gt_id, []) if m.cam in demo_cams]
        if len(pred_on) >= len(demo_cams):
            for m in pred_on:
                if not any(cam == m.cam for cam, _ in views):
                    views.append((m.cam, list(m.bbox_xyxy)))
        if len(views) >= len(demo_cams):
            cross_by_id[gt_id] = views
    cross_gt = list(cross_by_id.items())
    if not cross_gt:
        return None

    cross_gt.sort(
        key=lambda item: (
            sum(1 for m in pred_by_gt.get(item[0], []) if m.cam in demo_cams),
            len(item[1]),
        ),
        reverse=True,
    )
    top = cross_gt[: max(max_shared_gt + 2, 8)]
    best: tuple[list[ClassifiedMatch], list[ClassifiedMatch], list[int], list[int], float, bool] | None = None
    best_score = -1.0
    max_n = min(max_shared_gt, len(top))
    min_n = min(min_shared_gt, max_n)
    for n_gt in range(max_n, min_n - 1, -1):
        if n_gt < 1:
            continue
        for gt_pick in combinations(top, n_gt):
            picks: list[ClassifiedMatch] = []
            gt_supplemented = False
            ok = True
            for gt_id, views in gt_pick:
                views = [(cam, box) for cam, box in views if cam in demo_cams]
                group, sup = build_cross_cam_matches_for_gt(
                    frame,
                    gt_id,
                    views,
                    classified,
                    pred_by_gt,
                    gt_shrink=gt_shrink,
                )
                group = [m for m in group if m.cam in demo_cams]
                if len(group) < len(demo_cams):
                    ok = False
                    break
                picks.extend(group)
                gt_supplemented = gt_supplemented or sup
            if not ok:
                continue
            picks = fill_cam_pair_correct(
                picks,
                classified,
                demo_cams,
                error_gt_ids,
                benchmark_gt_ids,
                target_per_cam=target_per_cam,
                min_area=min_area,
                min_side_px=min_side_px,
            )
            pred_only = not gt_supplemented
            score = score_cam_pair_layout(
                picks,
                error_on_demo,
                demo_cams,
                pred_only=pred_only,
            )
            if score <= best_score:
                continue
            best_score = score
            best = (
                picks,
                error_on_demo,
                sorted(demo_cams),
                sorted({m.cam for m in error_on_demo}),
                score,
                gt_supplemented,
            )
    return best


def build_cam_pair_demo_matches(
    frame: int,
    classified: list[ClassifiedMatch],
    gt_by_cam: dict[int, np.ndarray],
    demo_cams: set[int],
    pair_errors: list[tuple[int | None, list[ClassifiedMatch], str]],
    benchmark_gt_ids: set[int],
    *,
    min_area: float,
    min_side_px: float,
    min_shared_gt: int,
    max_shared_gt: int,
    target_per_cam: int,
) -> tuple[
    list[ClassifiedMatch],
    list[int],
    list[int],
    int | None,
    list[int],
    list[int],
    bool,
    float,
    dict[str, float],
    str,
] | None:
    best_result = None
    best_score = -1.0
    best_metrics: dict[str, float] = {}
    best_error_kind = "wrong_merge"

    for error_gid, error_group, error_kind in pair_errors:
        error_gt_ids = {m.gt_id for m in error_group}
        pred_by_gt = pred_correct_by_gt(
            classified,
            error_gt_ids,
            benchmark_gt_ids,
            min_area=min_area,
            min_side_px=min_side_px,
        )
        for gt_id, matches in pred_by_gt.items():
            pred_by_gt[gt_id] = [m for m in matches if m.cam in demo_cams]

        for try_min in range(max_shared_gt, 0, -1):
            if try_min < min_shared_gt and try_min != max(1, min_shared_gt - 1):
                continue
            layout = search_cam_pair_cityflow_layout(
                frame,
                error_group,
                pred_by_gt,
                gt_by_cam,
                classified,
                demo_cams,
                benchmark_gt_ids,
                error_gt_ids,
                min_area=min_area,
                min_side_px=min_side_px,
                min_shared_gt=try_min,
                max_shared_gt=max_shared_gt,
                target_per_cam=target_per_cam,
            )
            if layout is None:
                continue
            correct, error_matches, correct_cams, error_cams, score, gt_supplemented = layout
            total_score = score
            if total_score <= best_score:
                continue
            best_score = total_score
            best_metrics = {}
            best_error_kind = error_kind
            best_result = (
                [*error_matches, *correct],
                sorted({m.gt_id for m in correct}),
                sorted({m.gt_id for m in error_matches}),
                error_gid,
                correct_cams,
                error_cams,
                gt_supplemented,
                score,
                best_metrics,
                error_kind,
            )
            break
    return best_result


def build_cross_cam_matches_for_gt(
    frame: int,
    gt_id: int,
    views: list[tuple[int, list[float]]],
    classified: list[ClassifiedMatch],
    pred_by_gt: dict[int, list[ClassifiedMatch]],
    *,
    gt_shrink: float = CF_GT_SHRINK,
) -> tuple[list[ClassifiedMatch], bool]:
    """All pred boxes for gt_id on every camera, plus GT fallback where pred is missing."""
    global_id = canonical_global_id_for_gt(classified, gt_id)
    pred_matches = list(pred_by_gt.get(gt_id, []))
    pred_on_cam = {m.cam: m for m in pred_matches}
    out: list[ClassifiedMatch] = list(pred_matches)
    gt_supplemented = False
    for cam, box in views:
        if cam in pred_on_cam:
            continue
        out.append(
            gt_match_from_box(
                frame=frame,
                cam=cam,
                gt_id=gt_id,
                box=box,
                global_id=global_id,
                shrink=gt_shrink,
            )
        )
        gt_supplemented = True
    return out, gt_supplemented


def search_cross_cam_cityflow_layout(
    frame: int,
    error_group: list[ClassifiedMatch],
    pred_by_gt: dict[int, list[ClassifiedMatch]],
    gt_by_cam: dict[int, np.ndarray],
    classified: list[ClassifiedMatch],
    cameras: list[int],
    benchmark_gt_ids: set[int],
    error_gt_ids: set[int],
    *,
    min_area: float,
    min_side_px: float,
    min_cross_cam_gt: int = 2,
    max_cross_cam_gt: int = 3,
    gt_shrink: float = CF_GT_SHRINK,
) -> tuple[list[ClassifiedMatch], list[ClassifiedMatch], list[int], list[int], float, bool] | None:
    best: tuple[list[ClassifiedMatch], list[ClassifiedMatch], list[int], list[int], float, bool] | None = None
    best_score = -1.0
    error_cams_set = {m.cam for m in error_group}

    cross_gt = cross_cam_gt_at_frame(
        frame,
        gt_by_cam,
        cameras,
        benchmark_gt_ids,
        error_gt_ids,
        set(),
        pred_by_gt,
        min_area=min_area,
        min_side_px=min_side_px,
    )
    if len(cross_gt) < min_cross_cam_gt:
        return None
    cross_gt.sort(
        key=lambda item: (
            sum(1 for m in pred_by_gt.get(item[0], []) if m.cam in {c for c, _ in item[1]}),
            len(item[1]),
        ),
        reverse=True,
    )
    top = cross_gt[: max(max_cross_cam_gt + 3, 8)]
    max_n = min(max_cross_cam_gt, len(top))
    for n_gt in range(min_cross_cam_gt, max_n + 1):
        for gt_pick in combinations(top, n_gt):
            picks: list[ClassifiedMatch] = []
            gt_supplemented = False
            ok = True
            for gt_id, views in gt_pick:
                group, sup = build_cross_cam_matches_for_gt(
                    frame,
                    gt_id,
                    views,
                    classified,
                    pred_by_gt,
                    gt_shrink=gt_shrink,
                )
                if len(group) < 2:
                    ok = False
                    break
                picks.extend(group)
                gt_supplemented = gt_supplemented or sup
            if not ok:
                continue
            correct_cams = sorted({p.cam for p in picks})
            if len(correct_cams) < 2:
                continue
            pred_only = not gt_supplemented
            score = score_cityflow_layout(picks, error_group, pred_only=pred_only)
            if score <= best_score:
                continue
            best_score = score
            best = (
                picks,
                list(error_group),
                correct_cams,
                sorted(error_cams_set),
                score,
                gt_supplemented,
            )
    return best


def build_gta_demo_matches(
    frame: int,
    classified: list[ClassifiedMatch],
    gt_by_cam: dict[int, np.ndarray],
    cameras: list[int],
    ranked_clusters: list[tuple[int, list[ClassifiedMatch], dict[str, float], float]],
    benchmark_gt_ids: set[int],
    *,
    min_area: float,
    min_side_px: float,
    min_vehicles: int,
    max_vehicles: int,
) -> tuple[
    list[ClassifiedMatch],
    list[int],
    list[int],
    int,
    list[int],
    list[int],
    bool,
    float,
    dict[str, float],
] | None:
    best_result = None
    best_score = -1.0
    best_metrics: dict[str, float] = {}

    for error_gid, error_group, geo_metrics, geo_score in ranked_clusters:
        error_gt_ids = {m.gt_id for m in error_group}
        pred_by_gt = pred_correct_by_gt(
            classified,
            error_gt_ids,
            benchmark_gt_ids,
            min_area=min_area,
            min_side_px=min_side_px,
        )
        layout = search_cross_cam_cityflow_layout(
            frame,
            error_group,
            pred_by_gt,
            gt_by_cam,
            classified,
            cameras,
            benchmark_gt_ids,
            error_gt_ids,
            min_area=min_area,
            min_side_px=min_side_px,
            min_cross_cam_gt=min_vehicles,
            max_cross_cam_gt=max_vehicles,
            gt_shrink=GTA_GT_SHRINK,
        )
        if layout is None:
            continue
        correct, error_matches, correct_cams, error_cams, score, gt_supplemented = layout
        total_score = geo_score + score * 0.01
        if total_score <= best_score:
            continue
        best_score = total_score
        best_metrics = geo_metrics
        best_result = (
            [*error_matches, *correct],
            sorted({m.gt_id for m in correct}),
            sorted({m.gt_id for m in error_matches}),
            error_gid,
            correct_cams,
            error_cams,
            gt_supplemented,
            score,
            geo_metrics,
        )
    return best_result


def build_cityflow_demo_matches(
    frame: int,
    classified: list[ClassifiedMatch],
    gt_by_cam: dict[int, np.ndarray],
    cameras: list[int],
    ranked_clusters: list[tuple[int, list[ClassifiedMatch], dict[str, float], float]],
    benchmark_gt_ids: set[int],
    *,
    min_area: float,
    min_side_px: float,
    n_correct: int,
) -> tuple[
    list[ClassifiedMatch],
    list[int],
    list[int],
    int,
    list[int],
    list[int],
    list[int],
    bool,
    float,
    dict[str, float],
] | None:
    best_result = None
    best_score = -1.0
    best_metrics: dict[str, float] = {}

    for error_gid, error_group, geo_metrics, geo_score in ranked_clusters:
        error_gt_ids = {m.gt_id for m in error_group}
        pred_by_gt = pred_correct_by_gt(
            classified,
            error_gt_ids,
            benchmark_gt_ids,
            min_area=min_area,
            min_side_px=min_side_px,
        )

        layout = search_cross_cam_cityflow_layout(
            frame,
            error_group,
            pred_by_gt,
            gt_by_cam,
            classified,
            cameras,
            benchmark_gt_ids,
            error_gt_ids,
            min_area=min_area,
            min_side_px=min_side_px,
            min_cross_cam_gt=2,
            max_cross_cam_gt=n_correct,
        )
        if layout is None:
            continue
        correct, error_matches, correct_cams, error_cams, score, gt_supplemented = layout
        total_score = geo_score + score * 0.01
        if total_score <= best_score:
            continue
        best_score = total_score
        best_metrics = geo_metrics
        best_result = (
            [*error_matches, *correct],
            sorted({m.gt_id for m in correct}),
            sorted({m.gt_id for m in error_matches}),
            error_gid,
            correct_cams,
            error_cams,
            gt_supplemented,
            score,
            geo_metrics,
        )

    return best_result


def rank_gt_ids_for_correct(
    frame: int,
    classified: list[ClassifiedMatch],
    gt_by_cam: dict[int, np.ndarray],
    cameras: list[int],
    benchmark_gt_ids: set[int],
    error_gt_ids: set[int],
    *,
    min_area: float,
    min_side_px: float,
    min_views: int = 2,
) -> list[int]:
    ranked: list[tuple[tuple[int, int, float], int]] = []
    for gt_id in benchmark_gt_ids:
        if gt_id in error_gt_ids:
            continue
        views = gt_views_for_id(
            frame, gt_id, gt_by_cam, cameras, min_area=min_area, min_side_px=min_side_px
        )
        if len(views) < min_views:
            continue
        has_pred = any(m.gt_id == gt_id and m.status == "correct" for m in classified)
        max_area = max(
            max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) for _, box in views
        )
        ranked.append(((1 if has_pred else 0, len(views), max_area), gt_id))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [gt_id for _, gt_id in ranked]


def rep_area(gt_id: int, pool: dict[int, list[ClassifiedMatch]]) -> float:
    return max(m.area for m in pool[gt_id])


def build_frame_candidate(
    frame: int,
    gt_by_cam: dict[int, np.ndarray],
    local_by_cam: dict[int, dict],
    global_by_cam: dict[int, dict],
    homos: dict[int, np.ndarray],
    cameras: list[int],
    *,
    dataset_name: str,
    iou_thresh: float,
    min_area: float,
    min_side_px: float,
    benchmark_gt_ids: set[int] | None,
    min_correct: int,
    max_correct: int,
    min_error: int,
    allow_gt_supplement: bool = False,
    demo_cams: set[int] | None = None,
) -> FrameCandidate | None:
    all_matches: list[TrackMatch] = []
    for cam in cameras:
        gt_rows = gt_rows_at_frame(gt_by_cam[cam], frame)
        local_rows = local_by_cam[cam].get(frame, [])
        if not gt_rows or not local_rows:
            continue
        all_matches.extend(
            greedy_match_frame(
                gt_rows,
                local_rows,
                global_by_cam[cam],
                cam=cam,
                frame=frame,
                iou_thresh=iou_thresh,
            )
        )

    if not all_matches:
        return None

    classified = classify_cross_camera(all_matches)

    if demo_cams is not None:
        pair_errors = demo_pair_error_groups(
            classified,
            demo_cams,
            min_area=min_area,
            min_side_px=min_side_px,
        )
        if len(pair_errors) < min_error:
            return None
        cityflow = build_cam_pair_demo_matches(
            frame,
            classified,
            gt_by_cam,
            demo_cams,
            pair_errors,
            benchmark_gt_ids or set(),
            min_area=min_area,
            min_side_px=min_side_px,
            min_shared_gt=min_correct,
            max_shared_gt=max_correct,
            target_per_cam=min_correct,
        )
        if cityflow is None:
            return None
        (
            selected_matches,
            selected_correct,
            error_gt_ids,
            selected_error_gid,
            correct_cams,
            error_cams,
            gt_supplemented,
            layout_score,
            geo_metrics,
            error_kind,
        ) = cityflow
        score = layout_score
        return FrameCandidate(
            frame=frame,
            matches=selected_matches,
            correct_gt_ids=selected_correct,
            error_gt_ids=sorted(set(error_gt_ids)),
            error_global_id=selected_error_gid,
            score=score,
            gt_supplemented=gt_supplemented,
            correct_cams=sorted({m.cam for m in selected_matches if m.status == "correct"}),
            error_cams=sorted({m.cam for m in selected_matches if m.status == "error"}),
            error_cam=error_cams[0] if error_cams else None,
            error_gt_pair_dist_m=geo_metrics.get("min_gt_pair_m"),
            error_max_gt_pair_dist_m=geo_metrics.get("max_gt_pair_m"),
            error_pred_gt_dist_m=geo_metrics.get("mean_pred_gt_m"),
        )

    correct_by_gt: dict[int, list[ClassifiedMatch]] = {}
    for m in classified:
        if not bbox_ok(m.area, m.min_side, min_area=min_area, min_side_px=min_side_px):
            continue
        if benchmark_gt_ids is not None and m.gt_id not in benchmark_gt_ids:
            by_gt_cams = {mm.cam for mm in classified if mm.gt_id == m.gt_id}
            if len(by_gt_cams) < 2:
                continue
        if m.status == "correct":
            correct_by_gt.setdefault(m.gt_id, []).append(m)

    clusters = error_global_clusters(
        classified,
        min_area=min_area,
        min_side_px=min_side_px,
    )

    if len(clusters) < min_error:
        return None

    ranked = rank_error_clusters(
        frame, clusters, gt_by_cam, homos, dataset_name=dataset_name
    )
    if not ranked:
        return None

    selected_error_gid, selected_error_group, geo_metrics, _ = ranked[0]
    selected_error = {m.gt_id for m in selected_error_group}

    if allow_gt_supplement:
        if demo_cams is not None:
            raise RuntimeError("demo_cams path should have returned earlier")
        cityflow = build_cityflow_demo_matches(
            frame,
            classified,
            gt_by_cam,
            cameras,
            ranked,
            benchmark_gt_ids or set(),
            min_area=min_area,
            min_side_px=min_side_px,
            n_correct=max_correct,
        )
        if cityflow is None:
            return None
        (
            selected_matches,
            selected_correct,
            error_gt_ids,
            selected_error_gid,
            correct_cams,
            error_cams,
            gt_supplemented,
            layout_score,
            geo_metrics,
        ) = cityflow
        selected_error = set(error_gt_ids)
        score = layout_score + error_cluster_rank_score(selected_error_group, geo_metrics) * 0.01
    else:
        gta = build_gta_demo_matches(
            frame,
            classified,
            gt_by_cam,
            cameras,
            ranked,
            benchmark_gt_ids or set(),
            min_area=min_area,
            min_side_px=min_side_px,
            min_vehicles=min(2, min_correct),
            max_vehicles=max_correct,
        )
        if gta is None:
            return None
        (
            selected_matches,
            selected_correct,
            error_gt_ids,
            selected_error_gid,
            correct_cams,
            error_cams,
            gt_supplemented,
            layout_score,
            geo_metrics,
        ) = gta
        selected_error = set(error_gt_ids)
        score = layout_score + error_cluster_rank_score(selected_error_group, geo_metrics) * 0.01

    correct_cam_list = sorted({m.cam for m in selected_matches if m.status == "correct"})
    error_cam_list = sorted({m.cam for m in selected_matches if m.status == "error"})
    return FrameCandidate(
        frame=frame,
        matches=selected_matches,
        correct_gt_ids=selected_correct,
        error_gt_ids=sorted(selected_error),
        error_global_id=selected_error_gid,
        score=score,
        gt_supplemented=gt_supplemented,
        correct_cams=correct_cam_list,
        error_cams=error_cam_list,
        error_cam=error_cam_list[0] if error_cam_list else None,
        error_gt_pair_dist_m=geo_metrics.get("min_gt_pair_m"),
        error_max_gt_pair_dist_m=geo_metrics.get("max_gt_pair_m"),
        error_pred_gt_dist_m=geo_metrics.get("mean_pred_gt_m"),
    )


def select_diverse(
    candidates: list[FrameCandidate],
    num: int,
    min_gap: int,
    *,
    preserve_order: bool = False,
) -> list[FrameCandidate]:
    ranked = candidates if preserve_order else sorted(candidates, key=lambda c: c.score, reverse=True)
    selected: list[FrameCandidate] = []
    used_frames: list[int] = []
    for cand in ranked:
        if any(abs(cand.frame - f) < min_gap for f in used_frames):
            continue
        selected.append(cand)
        used_frames.append(cand.frame)
        if len(selected) >= num:
            break
    return selected


def select_diverse_cam_pair(
    candidates: list[FrameCandidate],
    demo_cams: set[int],
    *,
    min_shared_gt: int = 3,
) -> FrameCandidate | None:
    """Pick the frame with the most shared correct GT ids on both demo cameras."""

    def shared_count(c: FrameCandidate) -> int:
        return len(shared_correct_gt_on_cams(
            [m for m in c.matches if m.status == "correct"],
            demo_cams,
        ))

    def green_per_cam(c: FrameCandidate) -> tuple[int, int]:
        greens = {
            cam: sum(1 for m in c.matches if m.status == "correct" and m.cam == cam)
            for cam in demo_cams
        }
        return (min(greens.values()), max(greens.values()))

    pool = [c for c in candidates if shared_count(c) >= min_shared_gt]
    target = min_shared_gt
    while not pool and target > 1:
        target -= 1
        pool = [c for c in candidates if shared_count(c) >= target]
        if pool:
            print(
                f"[cityflow] No frame with >={min_shared_gt} shared GT on cams "
                f"{sorted(demo_cams)}; using best with >={target}",
                flush=True,
            )
            break
    if not pool:
        pool = candidates

    pool.sort(
        key=lambda c: (
            shared_count(c),
            green_per_cam(c),
            sum(1 for m in c.matches if m.status == "error" and m.cam in demo_cams),
            c.score,
        ),
        reverse=True,
    )
    return pool[0] if pool else None


def select_diverse_cityflow(
    candidates: list[FrameCandidate],
    num: int,
    min_gap: int,
    *,
    min_cross_cam_gt: int = 2,
) -> list[FrameCandidate]:
    def cross_cam_vehicle_count(c: FrameCandidate) -> int:
        per_gt: dict[int, set[int]] = {}
        for m in c.matches:
            if m.status != "correct":
                continue
            per_gt.setdefault(m.gt_id, set()).add(m.cam)
        return sum(1 for cams in per_gt.values() if len(cams) >= 2)

    pool = [c for c in candidates if cross_cam_vehicle_count(c) >= min_cross_cam_gt]
    if len(pool) < num:
        print(
            f"[cityflow] Only {len(pool)} frames with >={min_cross_cam_gt} cross-cam vehicles; "
            f"relaxing filter",
            flush=True,
        )
        pool = [c for c in candidates if cross_cam_vehicle_count(c) >= 1]
    if len(pool) < num:
        pool = candidates

    def n_correct_boxes(c: FrameCandidate) -> int:
        return sum(1 for m in c.matches if m.status == "correct")

    pool = sorted(pool, key=lambda c: (n_correct_boxes(c), c.score), reverse=True)
    return select_diverse(pool, num, min_gap, preserve_order=True)


def build_demo_id_maps(
    matches: list[ClassifiedMatch],
) -> tuple[dict[tuple[int, int], int], dict[int, int]]:
    """Map raw track IDs to compact demo labels (1..40) preserving pairings."""
    global_map = {gid: idx + 1 for idx, gid in enumerate(sorted({m.global_id for m in matches}))}
    local_keys = sorted({(m.cam, m.local_id) for m in matches})
    local_map = {key: idx + 1 for idx, key in enumerate(local_keys)}
    return local_map, global_map


def _load_cyrillic_font(size: int):
    from PIL import ImageFont

    candidates = [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_association_legend(img: np.ndarray) -> None:
    """Bottom-left legend: line samples + L/G explanation (Cyrillic)."""
    from PIL import Image, ImageDraw

    h, w = img.shape[:2]
    font_size = max(16, h // 52)
    font = _load_cyrillic_font(font_size)

    pad_x, pad_y = 12, 10
    sample_w = 44
    gap = 12
    row_h = font_size + 12
    line_th = 3
    rows = [LEGEND_CORRECT, LEGEND_ERROR, LEGEND_ID_HELP]

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    text_widths = [int(draw.textlength(text, font=font)) for text in rows]
    box_w = pad_x * 2 + sample_w + gap + max(text_widths)
    box_h = pad_y * 2 + row_h * len(rows)
    x0, y0 = 12, h - box_h - 12

    draw.rounded_rectangle(
        (x0, y0, x0 + box_w, y0 + box_h),
        radius=6,
        fill=(255, 255, 255),
        outline=(136, 136, 136),
        width=2,
    )
    for i, text in enumerate(rows):
        cy = y0 + pad_y + i * row_h + row_h // 2
        draw.text(
            (x0 + pad_x + sample_w + gap, cy - font_size // 2),
            text,
            fill=(33, 33, 33),
            font=font,
        )

    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    lx1 = x0 + pad_x
    lx2 = lx1 + sample_w
    y_correct = y0 + pad_y + row_h // 2
    y_error = y0 + pad_y + row_h + row_h // 2
    cv2.line(img, (lx1, y_correct), (lx2, y_correct), COLOR_CORRECT, line_th, lineType=cv2.LINE_AA)
    draw_dashed_line(img, (lx1, y_error), (lx2, y_error), COLOR_ERROR, line_th)


def draw_dashed_line(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    *,
    dash_len: int = 14,
    gap_len: int = 10,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1e-6:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    pos = 0.0
    draw = True
    while pos < length:
        seg = dash_len if draw else gap_len
        end = min(pos + seg, length)
        if draw:
            sx = int(round(x1 + ux * pos))
            sy = int(round(y1 + uy * pos))
            ex = int(round(x1 + ux * end))
            ey = int(round(y1 + uy * end))
            cv2.line(img, (sx, sy), (ex, ey), color, thickness, lineType=cv2.LINE_AA)
        pos = end
        draw = not draw


def draw_rect_border(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    thickness: int,
    *,
    dashed: bool = False,
) -> None:
    if not dashed:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        return
    corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1))
    for (ax, ay), (bx, by) in zip(corners[:-1], corners[1:]):
        draw_dashed_line(img, (ax, ay), (bx, by), color, thickness)


def put_text_outlined(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    scale: float = 0.65,
    thickness: int = 2,
    outline_thickness: int = 4,
) -> None:
    """White label text with black outline for readability on any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        text,
        org,
        font,
        scale,
        COLOR_LABEL_OUTLINE,
        outline_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(img, text, org, font, scale, COLOR_LABEL, thickness, cv2.LINE_AA)


def draw_demo_box(
    img: np.ndarray,
    match: ClassifiedMatch,
    *,
    demo_local_id: int,
    demo_global_id: int,
) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in match.bbox_xyxy)
    border = COLOR_CORRECT if match.status == "correct" else COLOR_ERROR
    dashed = match.status == "error"
    draw_rect_border(img, x1, y1, x2, y2, border, 3, dashed=dashed)

    label = f"L{demo_local_id} G{demo_global_id}"
    y_text = max(20, y1 - 8)
    put_text_outlined(img, label, (x1, y_text))


def load_gta_frame(dataset: GtaMcmtDataset, cam: int, frame: int) -> np.ndarray | None:
    sync_index = frame - 1
    if sync_index < 0 or sync_index >= len(dataset):
        return None
    snap = dataset.snapshot(cam, sync_index)
    image_path = image_path_for_cam_dir(dataset.cam_dirs[cam], snap.cam_id)
    img = cv2.imread(str(image_path))
    return img


_cf_caps: dict[int, cv2.VideoCapture] = {}


def load_cityflow_frame(cam: int, frame: int) -> np.ndarray | None:
    path = CF_VIDEO[cam]
    if cam not in _cf_caps:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        _cf_caps[cam] = cap
    cap = _cf_caps[cam]
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame - 1))
    ok, img = cap.read()
    return img if ok else None


def release_cityflow_caps() -> None:
    for cap in _cf_caps.values():
        cap.release()
    _cf_caps.clear()


def peer_info(match: ClassifiedMatch, all_matches: list[ClassifiedMatch]) -> list[dict]:
    peers = []
    for m in all_matches:
        if m.gt_id == match.gt_id and m.cam != match.cam:
            peers.append({"cam": m.cam, "local_id": m.local_id, "global_id": m.global_id})
    return peers


def export_example(
    cand: FrameCandidate,
    *,
    example_idx: int,
    out_dir: Path,
    dataset_name: str,
    cameras: list[int],
    load_frame_fn,
    example_name: str | None = None,
    demo_cams: set[int] | None = None,
    target_shared_gt: int | None = None,
) -> Path:
    subdir = example_name if example_name else f"example_{example_idx:03d}"
    ex_dir = out_dir / dataset_name / subdir
    ex_dir.mkdir(parents=True, exist_ok=True)

    export_cams = sorted(demo_cams) if demo_cams is not None else cameras

    by_cam: dict[int, list[ClassifiedMatch]] = {}
    for m in cand.matches:
        by_cam.setdefault(m.cam, []).append(m)

    local_map, global_map = build_demo_id_maps(cand.matches)
    mosaic_frames: list[np.ndarray] = []
    cameras_meta: dict[str, dict] = {}

    if cand.correct_cams and cand.error_cams:
        cam_order = demo_cam_order(cand.correct_cams, cand.error_cams)
    else:
        cam_order = sorted(by_cam.keys())
    cam_order = [cam for cam in cam_order if cam in export_cams]

    for cam in cam_order:
        if cam not in by_cam or cam not in export_cams:
            continue
        img = load_frame_fn(cam, cand.frame)
        if img is None:
            continue

        annotated = img.copy()
        for m in by_cam[cam]:
            clipped = clip_box(*m.bbox_xyxy, img.shape[1], img.shape[0])
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            m_copy = ClassifiedMatch(
                cam=m.cam,
                frame=m.frame,
                local_id=m.local_id,
                global_id=m.global_id,
                gt_id=m.gt_id,
                bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
                iou=m.iou,
                status=m.status,
                bbox_source=m.bbox_source,
            )
            draw_demo_box(
                annotated,
                m_copy,
                demo_local_id=local_map[(m.cam, m.local_id)],
                demo_global_id=global_map[m.global_id],
            )

        draw_association_legend(annotated)

        cam_key = f"cam{cam}"
        cv2.imwrite(str(ex_dir / f"{cam_key}.jpg"), annotated)
        mosaic_frames.append(annotated)

        objects = []
        for m in by_cam[cam]:
            demo_l = local_map[(m.cam, m.local_id)]
            demo_g = global_map[m.global_id]
            objects.append(
                {
                    "gt_id": m.gt_id,
                    "local_id": m.local_id,
                    "global_id": m.global_id,
                    "demo_local_id": demo_l,
                    "demo_global_id": demo_g,
                    "status": m.status,
                    "bbox_xyxy": m.bbox_xyxy,
                    "bbox_source": m.bbox_source,
                    "iou": round(m.iou, 4),
                    "peer_cams": peer_info(m, cand.matches),
                }
            )
        cameras_meta[cam_key] = {"cam": cam, "frame": cand.frame, "objects": objects}

    if mosaic_frames:
        cv2.imwrite(str(ex_dir / "mosaic.jpg"), stack_camera_views(mosaic_frames))

    meta = {
        "dataset": dataset_name,
        "frame": cand.frame,
        "example_index": example_idx,
        "score": cand.score,
        "correct_gt_ids": cand.correct_gt_ids,
        "error_gt_ids": cand.error_gt_ids,
        "error_global_id": cand.error_global_id,
        "gt_supplemented": cand.gt_supplemented,
        "box_counts": {
            "correct": sum(1 for m in cand.matches if m.status == "correct"),
            "error": sum(1 for m in cand.matches if m.status == "error"),
        },
        "cameras": cameras_meta,
    }
    if demo_cams is not None:
        shared = shared_correct_gt_on_cams(
            [m for m in cand.matches if m.status == "correct"],
            demo_cams,
        )
        green_per_cam = {
            f"cam{cam}": sum(
                1 for m in cand.matches if m.status == "correct" and m.cam == cam
            )
            for cam in sorted(demo_cams)
        }
        error_on_demo = [
            m for m in cand.matches if m.status == "error" and m.cam in demo_cams
        ]
        meta["demo_cameras"] = sorted(demo_cams)
        meta["shared_gt_ids"] = shared
        meta["green_counts_per_cam"] = green_per_cam
        meta["shared_gt_target_requested"] = target_shared_gt or 4
        meta["shared_gt_achieved"] = len(shared)
        if len(shared) < (target_shared_gt or 3):
            meta["shared_gt_note"] = (
                "CityFlow cam6+cam7 overlap allows at most "
                f"{len(shared)} GT vehicle(s) visible on both cameras at one frame; "
                "extra greens are correct single-camera tracks."
            )
        if error_on_demo:
            err_gt = sorted({m.gt_id for m in error_on_demo})
            err_gids = sorted({m.global_id for m in error_on_demo})
            if len(err_gt) >= 2:
                meta["error_description"] = (
                    f"Wrong merge: global ID {cand.error_global_id} links different vehicles "
                    f"GT {err_gt} on cam(s) {sorted({m.cam for m in error_on_demo})}"
                )
            else:
                meta["error_description"] = (
                    f"Split identity: GT {err_gt[0]} assigned global IDs {err_gids} "
                    f"across cameras (shown on cam(s) {sorted({m.cam for m in error_on_demo})})"
                )
        else:
            meta["error_description"] = (
                f"Global ID {cand.error_global_id} wrongly merges GT ids "
                f"{cand.error_gt_ids} (error not on demo cameras)"
            )
    if cand.error_gt_pair_dist_m is not None:
        meta["error_geo"] = {
            "min_gt_pair_m": round(cand.error_gt_pair_dist_m, 3),
            "max_gt_pair_m": round(cand.error_max_gt_pair_dist_m or 0.0, 3),
            "mean_pred_gt_m": round(cand.error_pred_gt_dist_m or 0.0, 3),
        }
    if dataset_name in ("cityflow", "gta") and cand.correct_cams:
        meta["correct_cams"] = cand.correct_cams
        meta["error_cams"] = cand.error_cams
        meta["error_cam"] = cand.error_cam
        per_gt: dict[int, set[int]] = {}
        for m in cand.matches:
            if m.status != "correct":
                continue
            per_gt.setdefault(m.gt_id, set()).add(m.cam)
        meta["cross_cam_vehicle_count"] = sum(
            1 for cams in per_gt.values() if len(cams) >= 2
        )
    (ex_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return ex_dir


def scan_dataset(
    dataset_name: str,
    run_dir: Path,
    *,
    scan_stride: int,
    iou_thresh: float,
    min_area: float,
    min_side_px: float,
    min_correct: int,
    max_correct: int,
    min_error: int,
    num_examples: int,
    min_sync_gap: int,
    allow_gt_supplement: bool = False,
    demo_cams: set[int] | None = None,
) -> list[FrameCandidate]:
    print(f"[{dataset_name}] Loading predictions from {run_dir}", flush=True)

    if dataset_name == "gta":
        gt_by_cam, local_by_cam, global_by_cam, homos, max_frame = load_gta_data(run_dir)
        cameras = GTA_CAMERAS
        benchmark_gt_ids = cross_camera_gt_ids(gt_by_cam)
    else:
        gt_by_cam, local_by_cam, global_by_cam, homos, max_frame = load_cityflow_data(run_dir)
        cameras = S02_CAM_IDS
        benchmark_gt_ids = cross_camera_gt_ids(gt_by_cam)

    candidates: list[FrameCandidate] = []
    frame_list = list(range(1, max_frame + 1, scan_stride))
    for i, frame in enumerate(frame_list):
        if (i + 1) % 20 == 0:
            print(f"  scanned {i + 1}/{len(frame_list)} frames, found={len(candidates)}", flush=True)
        cand = build_frame_candidate(
            frame,
            gt_by_cam,
            local_by_cam,
            global_by_cam,
            homos,
            cameras,
            dataset_name=dataset_name,
            iou_thresh=iou_thresh,
            min_area=min_area,
            min_side_px=min_side_px,
            benchmark_gt_ids=benchmark_gt_ids,
            min_correct=min_correct,
            max_correct=max_correct,
            min_error=min_error,
            allow_gt_supplement=allow_gt_supplement,
            demo_cams=demo_cams,
        )
        if cand is not None:
            candidates.append(cand)

    print(f"[{dataset_name}] Found {len(candidates)} candidate frames", flush=True)
    if demo_cams is not None:
        picked = select_diverse_cam_pair(candidates, demo_cams, min_shared_gt=min_correct)
        selected = [picked] if picked is not None else []
    elif dataset_name == "cityflow":
        selected = select_diverse_cityflow(candidates, num_examples, min_sync_gap, min_cross_cam_gt=2)
    else:
        selected = select_diverse(candidates, num_examples, min_sync_gap)
    print(f"[{dataset_name}] Selected {len(selected)} examples", flush=True)
    return selected


def process_dataset(
    dataset_name: str,
    run_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[dict]:
    if not (run_dir / "per_cam" / ("c000.txt" if dataset_name == "gta" else "c006.txt")).is_file():
        raise SystemExit(f"Missing predictions: {run_dir}")

    stride = args.scan_stride or (50 if dataset_name == "gta" else 1)
    min_gap = args.min_sync_gap or (200 if dataset_name == "gta" else 5)
    min_side = 40.0 if dataset_name == "gta" else 20.0
    min_area = min_side * min_side
    min_correct = args.min_correct
    max_correct = args.max_correct
    allow_gt_supplement = dataset_name == "cityflow"
    demo_cams: set[int] | None = None
    if args.cameras is not None:
        if dataset_name != "cityflow":
            raise SystemExit("--cameras is only supported for --dataset cityflow")
        demo_cams = set(args.cameras)
        unknown = demo_cams - set(S02_CAM_IDS)
        if unknown:
            raise SystemExit(f"Unknown CityFlow cameras: {sorted(unknown)}; valid: {S02_CAM_IDS}")

    selected = scan_dataset(
        dataset_name,
        run_dir,
        scan_stride=stride,
        iou_thresh=args.iou_thresh,
        min_area=min_area,
        min_side_px=min_side,
        min_correct=min_correct,
        max_correct=max_correct,
        min_error=args.min_error,
        num_examples=args.num_examples,
        min_sync_gap=min_gap,
        allow_gt_supplement=allow_gt_supplement,
        demo_cams=demo_cams,
    )

    if not selected:
        print(f"[WARN] No suitable frames for {dataset_name}. Try lowering --min-correct or --scan-stride.", flush=True)
        return []

    gta_dataset = GtaMcmtDataset(GTA_GT_ROOT) if dataset_name == "gta" else None

    def load_frame(cam: int, frame: int) -> np.ndarray | None:
        if dataset_name == "gta":
            assert gta_dataset is not None
            return load_gta_frame(gta_dataset, cam, frame)
        return load_cityflow_frame(cam, frame)

    export_cams = sorted(demo_cams) if demo_cams is not None else (
        GTA_CAMERAS if dataset_name == "gta" else S02_CAM_IDS
    )
    example_name = args.example_name
    if demo_cams is not None and example_name is None and args.num_examples == 1:
        example_name = "example_001"

    index: list[dict] = []
    for idx, cand in enumerate(selected, start=1):
        label = example_name if example_name and idx == 1 else f"example_{idx:03d}"
        print(
            f"  {label}: frame={cand.frame} correct={cand.correct_gt_ids} error={cand.error_gt_ids}",
            flush=True,
        )
        ex_dir = export_example(
            cand,
            example_idx=idx,
            out_dir=out_dir,
            dataset_name=dataset_name,
            cameras=export_cams,
            load_frame_fn=load_frame,
            example_name=example_name if idx == 1 else None,
            demo_cams=demo_cams,
            target_shared_gt=min_correct,
        )
        index.append(json.loads((ex_dir / "meta.json").read_text(encoding="utf-8")))

    if dataset_name == "cityflow":
        release_cityflow_caps()

    return index


def main() -> None:
    args = parse_args()
    if args.cameras is not None and args.dataset not in ("cityflow", "both"):
        raise SystemExit("--cameras requires --dataset cityflow")
    if args.example_name is not None and args.num_examples > 1:
        raise SystemExit("--example-name can only be used with --num-examples 1")
    out_dir = args.out_dir
    datasets = ["gta", "cityflow"] if args.dataset == "both" else [args.dataset]
    if args.force:
        import shutil

        if args.dataset == "both":
            if out_dir.exists():
                shutil.rmtree(out_dir)
        else:
            ds_out = out_dir / args.dataset
            if ds_out.exists():
                shutil.rmtree(ds_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_index: dict[str, list[dict]] = {}
    index_path = out_dir / "index.json"
    if index_path.is_file() and args.dataset != "both":
        all_index = json.loads(index_path.read_text(encoding="utf-8"))

    for ds in datasets:
        run_dir = args.run_dir if args.run_dir is not None else DEFAULT_RUNS[ds]
        if args.run_dir is not None and len(datasets) > 1:
            raise SystemExit("--run-dir can only be used with a single --dataset")
        all_index[ds] = process_dataset(ds, run_dir, out_dir, args)

    (out_dir / "index.json").write_text(json.dumps(all_index, indent=2), encoding="utf-8")
    print(f"Done. Output: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
