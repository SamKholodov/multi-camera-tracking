"""GT-derived kinematic thresholds for GTA MCMT homography world points."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.io.calibration import (  # noqa: E402
    load_homography_image_to_world,
    project_bbox_bottom_center,
    world_distance,
)
from core.io.gta_mcmt import GtaMcmtDataset, center_bbox_to_xyxy  # noqa: E402
from core.mot.association.trajectory import linear_prediction  # noqa: E402


def _world_point(H_i2w, ann) -> tuple[float, float]:
    x1, y1, x2, y2 = center_bbox_to_xyxy(ann.cx, ann.cy, ann.w, ann.h)
    return project_bbox_bottom_center(H_i2w, x1, y1, x2, y2)


def _stats(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def collect_kinematic_stats(
    dataset_root: Path,
    *,
    max_frames: int | None,
    fps: float,
    metric: str,
    history_ks: tuple[int, ...],
) -> dict:
    ds = GtaMcmtDataset(dataset_root)
    homos = [
        load_homography_image_to_world(dataset_root / f"cam-{c}" / "calibration.txt")
        for c in range(ds.num_cameras)
    ]
    n_frames = ds.length if max_frames is None else min(ds.length, max_frames)

    per_obj_cam: dict[tuple[int, int], list[tuple[int, tuple[float, float]]]] = defaultdict(list)
    by_frame_obj: dict[tuple[int, int], list[tuple[int, tuple[float, float]]]] = defaultdict(list)

    for frame_idx in range(n_frames):
        for cam in range(ds.num_cameras):
            snap = ds.snapshot(cam, frame_idx)
            for ann in snap.annotations:
                wpt = _world_point(homos[cam], ann)
                per_obj_cam[(ann.obj_id, cam)].append((frame_idx, wpt))
                by_frame_obj[(frame_idx, ann.obj_id)].append((cam, wpt))

    same_cam_speeds: list[float] = []
    for entries in per_obj_cam.values():
        entries.sort(key=lambda x: x[0])
        for (f0, p0), (f1, p1) in zip(entries, entries[1:]):
            gap = int(f1) - int(f0)
            if gap <= 0:
                continue
            same_cam_speeds.append(
                world_distance(p0, p1, metric=metric) / (float(gap) / float(fps))
            )

    cross_cam_same_frame_speeds: list[float] = []
    for entries in by_frame_obj.values():
        if len(entries) < 2:
            continue
        for (cam_a, p_a), (cam_b, p_b) in combinations(entries, 2):
            if int(cam_a) == int(cam_b):
                continue
            # One-frame denominator keeps the value finite for simultaneous views;
            # this diagnoses homography disagreement rather than physical speed.
            cross_cam_same_frame_speeds.append(
                world_distance(p_a, p_b, metric=metric) * float(fps)
            )

    trajectory_errors: dict[int, list[float]] = {k: [] for k in history_ks}
    for entries in per_obj_cam.values():
        entries.sort(key=lambda x: x[0])
        for idx in range(1, len(entries)):
            frame_idx, query_wpt = entries[idx]
            for k in history_ks:
                hist = entries[max(0, idx - k) : idx]
                pred = linear_prediction(hist, frame_idx)
                if pred is None:
                    continue
                trajectory_errors[k].append(world_distance(query_wpt, pred, metric=metric))

    result = {
        "dataset_root": str(dataset_root),
        "frames": int(n_frames),
        "fps": float(fps),
        "metric": metric,
        "same_cam_speed_mps": _stats(same_cam_speeds),
        "cross_cam_same_frame_disagreement_mps_equiv": _stats(cross_cam_same_frame_speeds),
        "trajectory_error_m": {
            str(k): _stats(vals) for k, vals in trajectory_errors.items()
        },
    }
    speed_p99 = result["same_cam_speed_mps"].get("p99", 25.0)
    result["suggested"] = {
        "speed_v_max_mps": float(min(max(float(speed_p99) * 1.2, 15.0), 35.0)),
        "trajectory_threshold_m": {
            str(k): vals.get("p95", 10.0)
            for k, vals in result["trajectory_error_m"].items()
        },
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=_ROOT / "datasets" / "gta_mcmt")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Sync-frame rate for GTA MCMT (30). Use 10 for S02/CityFlow.",
    )
    ap.add_argument("--metric", choices=("plane", "gps"), default="plane")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    result = collect_kinematic_stats(
        args.dataset_root,
        max_frames=args.max_frames,
        fps=args.fps,
        metric=args.metric,
        history_ks=(1, 3, 5),
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"GTA kinematic GT stats ({result['frames']} frames, {args.metric}, fps={args.fps})")
    for section in ("same_cam_speed_mps", "cross_cam_same_frame_disagreement_mps_equiv"):
        print(f"\n{section}:")
        for key, val in result[section].items():
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    print("\ntrajectory_error_m:")
    for k, stats in result["trajectory_error_m"].items():
        print(f"  K={k}:")
        for key, val in stats.items():
            print(f"    {key}: {val:.4f}" if isinstance(val, float) else f"    {key}: {val}")
    print("\nSuggested:")
    print(json.dumps(result["suggested"], indent=2))


if __name__ == "__main__":
    main()
