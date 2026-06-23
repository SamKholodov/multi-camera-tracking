"""Cross-camera same-GT-object world distances on GTA MCMT (plane metric).

Projects bbox bottom-center via homography (same as pipeline world_anchor=bottom_center).
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.io.calibration import (
    load_homography_image_to_world,
    project_bbox_bottom_center,
    world_distance,
)
from core.io.gta_mcmt import (
    GtaMcmtDataset,
    center_bbox_to_xyxy,
)


def _world_point(H_i2w, ann) -> tuple[float, float]:
    x1, y1, x2, y2 = center_bbox_to_xyxy(ann.cx, ann.cy, ann.w, ann.h)
    return project_bbox_bottom_center(H_i2w, x1, y1, x2, y2)


def collect_distances(
    dataset_root: Path,
    *,
    max_frames: int | None = None,
    metric: str = "plane",
) -> list[float]:
    ds = GtaMcmtDataset(dataset_root)
    homos = [
        load_homography_image_to_world(dataset_root / f"cam-{c}" / "calibration.txt")
        for c in range(ds.num_cameras)
    ]
    n_frames = ds.length if max_frames is None else min(ds.length, max_frames)
    dists: list[float] = []

    for k in range(n_frames):
        by_obj: dict[int, list[tuple[int, tuple[float, float]]]] = {}
        for cam in range(ds.num_cameras):
            snap = ds.snapshot(cam, k)
            for ann in snap.annotations:
                wpt = _world_point(homos[cam], ann)
                by_obj.setdefault(ann.obj_id, []).append((cam, wpt))

        for obj_id, entries in by_obj.items():
            if len(entries) < 2:
                continue
            for (cam_a, wpt_a), (cam_b, wpt_b) in combinations(entries, 2):
                if cam_a == cam_b:
                    continue
                d = world_distance(wpt_a, wpt_b, metric=metric)
                dists.append(float(d))

    return dists


def _pct(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p)) if arr.size else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=_ROOT / "datasets" / "gta_mcmt",
    )
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--metric", choices=("plane", "gps"), default="plane")
    ap.add_argument("--json", action="store_true", help="Print stats as JSON")
    args = ap.parse_args()

    dists = collect_distances(
        args.dataset_root,
        max_frames=args.max_frames,
        metric=args.metric,
    )
    if not dists:
        print("No cross-camera same-object pairs found.")
        return

    arr = np.asarray(dists, dtype=np.float64)
    stats = {
        "pairs": int(arr.size),
        "min_m": float(arr.min()),
        "median_m": float(np.median(arr)),
        "mean_m": float(arr.mean()),
        "p90_m": _pct(arr, 90),
        "p95_m": _pct(arr, 95),
        "p99_m": _pct(arr, 99),
        "max_m": float(arr.max()),
    }
    t_min = stats["p95_m"]
    t_distant = max(stats["max_m"] * 1.5, t_min + 0.5)
    suggested = {
        "geometry_t_min_m": round(t_min, 3),
        "geometry_t_distant_m": round(t_distant, 3),
        "geometry_distance_metric": args.metric,
    }
    if args.json:
        print(json.dumps({"stats": stats, "suggested_thresholds": suggested}, indent=2))
        return

    print(f"GTA MCMT cross-cam same-object distances ({args.metric})")
    for key, val in stats.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    print(f"\nSuggested thresholds (homography / association plane metric):")
    print(f"  geometry_t_min_m: {suggested['geometry_t_min_m']:.3f}")
    print(f"  geometry_t_distant_m: {suggested['geometry_t_distant_m']:.3f}")

    # Native GT world coords (CSV bottom_world) for reference.
    from core.io.gta_mcmt import GtaMcmtDataset

    ds = GtaMcmtDataset(args.dataset_root)
    n_frames = ds.length if args.max_frames is None else min(ds.length, args.max_frames)
    native: list[float] = []
    for k in range(n_frames):
        by_obj: dict[int, list[tuple[int, tuple[float, float]]]] = {}
        for cam in range(ds.num_cameras):
            for ann in ds.snapshot(cam, k).annotations:
                wpt = (
                    (ann.bottom_world_x, ann.bottom_world_y)
                    if ann.bottom_world_x is not None
                    else (ann.world_x, ann.world_y)
                )
                by_obj.setdefault(ann.obj_id, []).append((cam, wpt))
        for entries in by_obj.values():
            if len(entries) < 2:
                continue
            for (cam_a, wpt_a), (cam_b, wpt_b) in combinations(entries, 2):
                if cam_a != cam_b:
                    native.append(
                        float(world_distance(wpt_a, wpt_b, metric=args.metric))
                    )
    if native:
        na = np.asarray(native, dtype=np.float64)
        print("\nGT CSV bottom_world (reference only, not used by homography pipeline):")
        print(f"  median_m: {float(np.median(na)):.4f}")
        print(f"  p95_m: {_pct(na, 95):.4f}")
        print(f"  max_m: {float(na.max()):.4f}")


if __name__ == "__main__":
    main()
