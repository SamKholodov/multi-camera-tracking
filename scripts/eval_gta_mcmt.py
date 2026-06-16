"""Evaluate MCMT / SCT on the GTA MCMT synthetic dataset.

GT layout: {gt_root}/cam-{N}/gt/gt.txt (from scripts/convert_gta_mcmt_gt.py)
Predictions: {pred_dir}/c{cam:03d}.txt (from MCMTResultWriter)

Usage:
    python scripts/convert_gta_mcmt_gt.py
    python run.py --config config/gta_mcmt_baseline.yaml
    python scripts/eval_gta_mcmt.py \\
        --gt-root datasets/gta_mcmt \\
        --pred-dir outputs/gta_mcmt_baseline/per_cam \\
        --cameras 0 1 2 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

try:
    import motmetrics as mm
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "motmetrics is required for evaluation. Install with: pip install motmetrics"
    ) from e

from core.io.gta_mcmt import NUM_CAMERAS
from core.io.roi import ROIFilter
from scripts.eval_s02 import (
    BATCH_METRICS,
    FRAME_GAP,
    SUMMARY_METRICS,
    accumulate,
    load_mot,
)


def gta_roi_path(gt_root: Path, cam: int) -> Path:
    return gt_root / f"cam-{cam}" / "roi.jpg"


def _apply_roi(gt_root: Path, cam: int, gt: np.ndarray, pr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    roi_path = gta_roi_path(gt_root, cam)
    if not roi_path.exists():
        print(f"[WARN] ROI missing for cam-{cam}: {roi_path}")
        return gt, pr
    roi = ROIFilter.from_path(roi_path)
    return roi.filter_mot(gt), roi.filter_mot(pr)


def gta_gt_path(gt_root: Path, cam: int) -> Path:
    return gt_root / f"cam-{cam}" / "gt" / "gt.txt"


def _load_gt_pred(
    gt_root: Path,
    pred_dir: Path,
    cam: int,
    *,
    max_frame: int | None,
    apply_roi: bool,
) -> tuple[np.ndarray, np.ndarray]:
    gt_path = gta_gt_path(gt_root, cam)
    pr_path = pred_dir / f"c{cam:03d}.txt"
    gt = _filter_mot_max_frame(load_mot(gt_path), max_frame)
    pr = _filter_mot_max_frame(load_mot(pr_path), max_frame)
    if apply_roi:
        gt, pr = _apply_roi(gt_root, cam, gt, pr)
    return gt, pr


def _filter_mot_max_frame(data: np.ndarray, max_frame: int | None) -> np.ndarray:
    if max_frame is None or len(data) == 0:
        return data
    mask = data[:, 0].astype(int) <= max_frame
    return data[mask]


def evaluate_gta_mcmt(
    gt_root: Path,
    pred_dir: Path,
    cameras: list[int] | None = None,
    max_iou_dist: float = 0.5,
    max_frame: int | None = None,
    apply_roi: bool = False,
) -> dict:
    if cameras is None:
        cameras = list(range(NUM_CAMERAS))

    gt_by_cam: dict[int, np.ndarray] = {}
    pr_by_cam: dict[int, np.ndarray] = {}
    for cam in cameras:
        gt_path = gta_gt_path(gt_root, cam)
        pr_path = pred_dir / f"c{cam:03d}.txt"
        if not gt_path.exists():
            print(f"[WARN] GT missing: {gt_path}")
            continue
        gt, pr = _load_gt_pred(
            gt_root,
            pred_dir,
            cam,
            max_frame=max_frame,
            apply_roi=apply_roi,
        )
        gt_by_cam[cam] = gt
        pr_by_cam[cam] = pr

    if not gt_by_cam:
        raise SystemExit("No GT files found, nothing to evaluate.")

    accs: dict[str, mm.MOTAccumulator] = {}
    mcmt_gt_rows: list[np.ndarray] = []
    mcmt_pr_rows: list[np.ndarray] = []
    frame_offset = 0

    for cam in cameras:
        if cam not in gt_by_cam:
            continue
        gt = gt_by_cam[cam]
        pr = pr_by_cam.get(cam, np.empty((0, 10)))
        accs[f"c{cam:03d}"] = accumulate(gt, pr, max_iou_dist)
        if len(gt):
            gt_off = gt.copy()
            gt_off[:, 0] += frame_offset
            mcmt_gt_rows.append(gt_off)
        if len(pr):
            pr_off = pr.copy()
            pr_off[:, 0] += frame_offset
            mcmt_pr_rows.append(pr_off)
        frame_offset += FRAME_GAP

    mh = mm.metrics.create()
    per_cam = mh.compute_many(
        list(accs.values()),
        names=list(accs.keys()),
        metrics=BATCH_METRICS,
        generate_overall=True,
    )

    mcmt_row = None
    if mcmt_gt_rows and mcmt_pr_rows:
        mcmt_acc = accumulate(
            np.concatenate(mcmt_gt_rows, axis=0),
            np.concatenate(mcmt_pr_rows, axis=0),
            max_iou_dist,
        )
        mcmt_row = mh.compute(mcmt_acc, metrics=BATCH_METRICS)

    return {"per_cam": per_cam, "mcmt": mcmt_row}


def _print_summary(summaries: dict[str, mm.MOTAccumulator]) -> None:
    mh = mm.metrics.create()
    summary = mh.compute_many(
        list(summaries.values()),
        names=list(summaries.keys()),
        metrics=SUMMARY_METRICS,
        generate_overall=True,
    )
    print(
        mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-root", type=Path, default=Path("datasets/gta_mcmt"))
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--cameras", nargs="+", type=int, default=list(range(NUM_CAMERAS)))
    ap.add_argument("--max-iou-dist", type=float, default=0.5)
    ap.add_argument(
        "--max-frame",
        type=int,
        default=None,
        help="Evaluate only frames <= this id (useful after smoke runs with max_frames=30)",
    )
    ap.add_argument(
        "--apply-roi",
        action="store_true",
        help="Filter GT and predictions with cam-N/roi.jpg (CityFlow bottom-center rule)",
    )
    args = ap.parse_args()

    gt_by_cam: dict[int, np.ndarray] = {}
    pr_by_cam: dict[int, np.ndarray] = {}
    accs: dict[str, mm.MOTAccumulator] = {}
    mcmt_gt_rows: list[np.ndarray] = []
    mcmt_pr_rows: list[np.ndarray] = []
    frame_offset = 0

    for cam in args.cameras:
        gt_path = gta_gt_path(args.gt_root, cam)
        pr_path = args.pred_dir / f"c{cam:03d}.txt"
        if not gt_path.exists():
            print(f"[WARN] GT missing: {gt_path}")
            continue
        gt, pr = _load_gt_pred(
            args.gt_root,
            args.pred_dir,
            cam,
            max_frame=args.max_frame,
            apply_roi=args.apply_roi,
        )
        gt_by_cam[cam] = gt
        pr_by_cam[cam] = pr
        accs[f"c{cam:03d}"] = accumulate(gt, pr, args.max_iou_dist)
        if len(gt):
            gt_off = gt.copy()
            gt_off[:, 0] += frame_offset
            mcmt_gt_rows.append(gt_off)
        if len(pr):
            pr_off = pr.copy()
            pr_off[:, 0] += frame_offset
            mcmt_pr_rows.append(pr_off)
        frame_offset += FRAME_GAP

    if not accs:
        raise SystemExit("No GT files found, nothing to evaluate.")

    print("\n=== Per-camera (GTA MCMT) ===")
    _print_summary(accs)

    if mcmt_gt_rows and mcmt_pr_rows:
        print("\n=== Concatenated multi-camera stream ===")
        _print_summary(
            {
                "MCMT": accumulate(
                    np.concatenate(mcmt_gt_rows, axis=0),
                    np.concatenate(mcmt_pr_rows, axis=0),
                    args.max_iou_dist,
                )
            }
        )


if __name__ == "__main__":
    main()
