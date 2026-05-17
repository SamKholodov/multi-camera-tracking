"""Evaluate MCMT / SCT baseline on the S02 validation set.

Reports IDF1 / IDP / IDR / MOTA / IDsw / MT / ML per camera and a
multi-camera summary, using py-motmetrics.

Install once:
    pip install motmetrics

Usage:
    python scripts/eval_s02.py \\
        --gt-root  datasets/validation/S02 \\
        --pred-dir outputs/s02_baseline/per_cam \\
        --cameras  6 7 8 9

Pass --pred-dir outputs/s02_baseline/per_cam_local to evaluate the
single-camera baseline (local ids only).

For cross-camera ID-aware aggregation we concatenate all per-camera
results into one stream where each detection's frame index is offset
by camera_index * 1e6; predicted ids stay global, GT ids stay global
across cameras (CityFlow GT uses scene-global ids), so IDF1 across the
concatenated stream is a fair MCMT proxy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

# motmetrics 1.4.0 still calls np.asfarray (removed in NumPy 2.0).
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

try:
    import motmetrics as mm
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "motmetrics is required for evaluation. Install with: pip install motmetrics"
    ) from e

from core.io.roi import ROIFilter


def _load_mot(path: Path) -> np.ndarray:
    """MOT16: frame,id,x,y,w,h,conf,-1,-1,-1 -> ndarray."""
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 10))
    data = np.loadtxt(str(path), delimiter=",")
    if data.size == 0:
        return np.empty((0, 10))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _to_motchallenge_frames(data: np.ndarray):
    """Yield (frame_id, ids, boxes_tlwh) groups sorted by frame."""
    if len(data) == 0:
        return
    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]
    frames = data[:, 0].astype(int)
    unique_frames = np.unique(frames)
    for f in unique_frames:
        mask = frames == f
        ids = data[mask, 1].astype(int)
        boxes = data[mask, 2:6]
        yield int(f), ids, boxes


def _accumulate(gt: np.ndarray, pred: np.ndarray, max_iou_dist: float = 0.5):
    acc = mm.MOTAccumulator(auto_id=True)
    gt_by_frame = {f: (ids, boxes) for f, ids, boxes in _to_motchallenge_frames(gt)}
    pr_by_frame = {f: (ids, boxes) for f, ids, boxes in _to_motchallenge_frames(pred)}
    all_frames = sorted(set(gt_by_frame) | set(pr_by_frame))
    for f in all_frames:
        g_ids, g_boxes = gt_by_frame.get(f, (np.empty(0, int), np.empty((0, 4))))
        p_ids, p_boxes = pr_by_frame.get(f, (np.empty(0, int), np.empty((0, 4))))
        dist = mm.distances.iou_matrix(g_boxes, p_boxes, max_iou=max_iou_dist)
        acc.update(g_ids.tolist(), p_ids.tolist(), dist)
    return acc


def _print_summary(summaries: dict[str, "mm.MOTAccumulator"]):
    mh = mm.metrics.create()
    metrics = [
        "idf1", "idp", "idr",
        "mota", "motp",
        "num_switches", "mostly_tracked", "mostly_lost",
        "num_false_positives", "num_misses", "num_unique_objects",
    ]
    summary = mh.compute_many(
        list(summaries.values()),
        names=list(summaries.keys()),
        metrics=metrics,
        generate_overall=True,
    )
    print(mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    ))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-root", required=True, type=Path)
    ap.add_argument("--pred-dir", required=True, type=Path,
                    help="Folder with c{cam:03d}.txt prediction files")
    ap.add_argument("--cameras", nargs="+", type=int, default=[6, 7, 8, 9])
    ap.add_argument("--max-iou-dist", type=float, default=0.5)
    ap.add_argument(
        "--apply-roi",
        action="store_true",
        help="Filter GT and predictions with datasets/.../cXXX/roi.jpg (CityFlow rule)",
    )
    args = ap.parse_args()

    accs: dict[str, "mm.MOTAccumulator"] = {}
    # Concatenated MCMT stream uses per-cam frame offset to keep frames disjoint.
    mcmt_gt_rows = []
    mcmt_pr_rows = []
    frame_offset = 0
    FRAME_GAP = 10_000_000

    for cam in args.cameras:
        gt_path = args.gt_root / f"c{cam:03d}" / "gt" / "gt.txt"
        pr_path = args.pred_dir / f"c{cam:03d}.txt"
        if not gt_path.exists():
            print(f"[WARN] GT missing: {gt_path}")
            continue
        gt = _load_mot(gt_path)
        pr = _load_mot(pr_path)
        if args.apply_roi:
            roi_path = args.gt_root / f"c{cam:03d}" / "roi.jpg"
            if roi_path.exists():
                roi = ROIFilter.from_path(roi_path)
                gt = roi.filter_mot(gt)
                pr = roi.filter_mot(pr)
            else:
                print(f"[WARN] ROI missing for c{cam:03d}: {roi_path}")
        accs[f"c{cam:03d}"] = _accumulate(gt, pr, args.max_iou_dist)

        if len(gt):
            gt_off = gt.copy(); gt_off[:, 0] += frame_offset
            mcmt_gt_rows.append(gt_off)
        if len(pr):
            pr_off = pr.copy(); pr_off[:, 0] += frame_offset
            mcmt_pr_rows.append(pr_off)
        frame_offset += FRAME_GAP

    if not accs:
        raise SystemExit("No GT files found, nothing to evaluate.")

    print("\n=== Per-camera (cross-cam ID-aware if pred uses global ids) ===")
    _print_summary(accs)

    if mcmt_gt_rows and mcmt_pr_rows:
        mcmt_gt = np.concatenate(mcmt_gt_rows, axis=0)
        mcmt_pr = np.concatenate(mcmt_pr_rows, axis=0)
        print("\n=== Concatenated multi-camera stream ===")
        _print_summary({"MCMT": _accumulate(mcmt_gt, mcmt_pr, args.max_iou_dist)})


if __name__ == "__main__":
    main()
