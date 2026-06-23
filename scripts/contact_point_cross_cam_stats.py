"""Cross-camera bottom-center vs contact-point distance statistics (pred bbox)."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.geometry.contact_point.inference import ContactPointInference
from scripts.build_contact_point_geo_demo import (
    ExampleCandidate,
    PredMatch,
    build_view,
    compute_pair_distances,
    greedy_match_frame,
    gt_rows_at_frame,
    load_cityflow_data,
    load_gta_data,
)


@dataclass
class CrossCamScanResult:
    example_pool: list[ExampleCandidate] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _distance_stats(arr: list[float]) -> dict:
    if not arr:
        return {}
    a = np.asarray(arr, dtype=np.float64)
    return {
        "count": int(a.size),
        "mean_m": float(a.mean()),
        "median_m": float(np.median(a)),
        "p95_m": float(np.percentile(a, 95)),
    }


def summarize_cross_cam_stats(
    *,
    dataset_name: str,
    metric: str,
    all_bottom: list[float],
    all_contact: list[float],
    all_improvements: list[float],
    frames_scanned: int,
    vehicles_total: int,
) -> dict:
    imp_arr = np.asarray(all_improvements, dtype=np.float64) if all_improvements else np.array([])
    bottom_stats = _distance_stats(all_bottom)
    contact_stats = _distance_stats(all_contact)
    improvement_stats = _distance_stats(all_improvements)

    mean_bottom = bottom_stats.get("mean_m")
    mean_contact = contact_stats.get("mean_m")
    relative_pct = None
    if mean_bottom is not None and mean_bottom > 1e-9 and mean_contact is not None:
        relative_pct = float((mean_bottom - mean_contact) / mean_bottom * 100.0)

    return {
        "dataset": dataset_name,
        "metric": metric,
        "pairs_total": len(all_improvements),
        "vehicles_total": vehicles_total,
        "frames_scanned": frames_scanned,
        "bottom": bottom_stats,
        "contact": contact_stats,
        "improvement": improvement_stats,
        "improvement_relative_pct": relative_pct,
        "pairs_improved_pct": float(100.0 * np.mean(imp_arr > 0)) if imp_arr.size else 0.0,
        "contact_model_note": (
            "trained on GTA; CityFlow is zero-shot"
            if dataset_name == "cityflow"
            else "trained on GTA"
        ),
    }


def collect_cross_cam_distance_stats(
    *,
    dataset_name: str,
    run_dir: Path,
    contact: ContactPointInference,
    scan_stride: int,
    min_side_px: float,
    min_area: float,
    metric: str,
    cameras: list[int],
    load_frame_fn: Callable[[int, int], np.ndarray | None],
    max_frames: int | None = None,
    collect_examples: bool = True,
    progress_every: int = 0,
) -> CrossCamScanResult:
    if dataset_name == "gta":
        gt_by_cam, local_by_cam, homos, _, max_frame = load_gta_data(run_dir)
    else:
        gt_by_cam, local_by_cam, homos, _, max_frame = load_cityflow_data(run_dir)

    if max_frames is not None:
        max_frame = min(max_frame, max_frames)

    all_improvements: list[float] = []
    all_bottom: list[float] = []
    all_contact: list[float] = []
    example_pool: list[ExampleCandidate] = []
    vehicles_total = 0
    frames_scanned = 0

    frame_list = list(range(1, max_frame + 1, scan_stride))
    for i, frame in enumerate(frame_list):
        if progress_every and (i + 1) % progress_every == 0:
            print(
                f"  scanned {i + 1}/{len(frame_list)} frames, pairs={len(all_improvements)}",
                flush=True,
            )
        frames_scanned += 1
        by_gt: dict[int, list[tuple[int, int, list[float], float]]] = {}
        for cam in cameras:
            gt_rows = gt_rows_at_frame(gt_by_cam[cam], frame)
            local_rows = local_by_cam[cam].get(frame, [])
            if not gt_rows or not local_rows:
                continue
            for local_id, gt_id, bbox, iou in greedy_match_frame(
                gt_rows, local_rows, cam=cam, frame=frame, iou_thresh=0.5
            ):
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                if area < min_area or min(x2 - x1, y2 - y1) < min_side_px:
                    continue
                by_gt.setdefault(gt_id, []).append((cam, local_id, bbox, iou))

        for gt_id, entries in by_gt.items():
            cams = {e[0] for e in entries}
            if len(cams) < 2:
                continue

            views: list[PredMatch] = []
            for cam, local_id, bbox, iou in entries:
                frame_bgr = load_frame_fn(cam, frame)
                view = build_view(
                    contact,
                    frame_bgr,
                    cam=cam,
                    frame=frame,
                    gt_id=gt_id,
                    local_id=local_id,
                    bbox=bbox,
                    H_i2w=homos[cam],
                    iou=iou,
                )
                if view is not None and view.contact_px is not None:
                    views.append(view)

            if len(views) < 2:
                continue

            d_b, d_c, imps = compute_pair_distances(views, metric=metric, dataset=dataset_name)
            if len(d_c) < 1:
                continue

            vehicles_total += 1
            all_bottom.extend(d_b.values())
            all_contact.extend(d_c.values())
            all_improvements.extend(imps)

            if not collect_examples:
                continue

            mean_imp = float(np.mean(imps)) if imps else 0.0
            min_area_v = min(v.area for v in views)
            score = mean_imp * 1e3 + min_area_v + len(views) * 100.0
            example_pool.append(
                ExampleCandidate(
                    frame=frame,
                    gt_id=gt_id,
                    views=views,
                    pair_distances_bottom=d_b,
                    pair_distances_contact=d_c,
                    mean_improvement=mean_imp,
                    score=score,
                )
            )

    stats = summarize_cross_cam_stats(
        dataset_name=dataset_name,
        metric=metric,
        all_bottom=all_bottom,
        all_contact=all_contact,
        all_improvements=all_improvements,
        frames_scanned=frames_scanned,
        vehicles_total=vehicles_total,
    )
    return CrossCamScanResult(example_pool=example_pool, stats=stats)
