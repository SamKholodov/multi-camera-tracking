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

Use --cityflow-protocol for CityFlow-aligned evaluation (ROI + cross-camera
objects only). Add --full-mot to also print standard MOT metrics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

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

from core.eval.cityflow_protocol import (
    PredIdMode,
    apply_cityflow_filters,
    infer_pred_id_mode,
)
from core.io.roi import ROIFilter

FRAME_GAP = 10_000_000
SUMMARY_METRICS = [
    "idf1", "idp", "idr",
    "mota", "motp",
    "num_switches", "mostly_tracked", "mostly_lost",
    "num_false_positives", "num_misses", "num_unique_objects",
]
BATCH_METRICS = ["idf1", "idp", "idr", "mota", "motp", "num_false_positives", "num_misses"]


def load_mot(path: Path) -> np.ndarray:
    """MOT16: frame,id,x,y,w,h,conf,-1,-1,-1 -> ndarray."""
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 10))
    data = np.loadtxt(str(path), delimiter=",")
    if data.size == 0:
        return np.empty((0, 10))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def s02_gt_max_frame(gt_root: Path, cameras: list[int] | None = None) -> int:
    """Return the maximum GT frame index across S02 cameras."""
    if cameras is None:
        cameras = [6, 7, 8, 9]
    mx = 0
    for cam in cameras:
        gt_path = gt_root / f"c{cam:03d}" / "gt" / "gt.txt"
        if not gt_path.is_file():
            continue
        gt = load_mot(gt_path)
        if len(gt):
            mx = max(mx, int(gt[:, 0].max()))
    return mx


_load_mot = load_mot


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


def accumulate(gt: np.ndarray, pred: np.ndarray, max_iou_dist: float = 0.5):
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


_accumulate = accumulate


def _apply_roi(gt_root: Path, cam: int, gt: np.ndarray, pr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    roi_path = gt_root / f"c{cam:03d}" / "roi.jpg"
    if not roi_path.exists():
        print(f"[WARN] ROI missing for c{cam:03d}: {roi_path}")
        return gt, pr
    roi = ROIFilter.from_path(roi_path)
    return roi.filter_mot(gt), roi.filter_mot(pr)


def _resolve_id_mode(pred_dir: Path, pred_id_mode: str) -> PredIdMode:
    if pred_id_mode == "auto":
        return infer_pred_id_mode(pred_dir)
    return pred_id_mode  # type: ignore[return-value]


def evaluate_s02(
    gt_root: Path,
    pred_dir: Path,
    cameras: list[int] | None = None,
    max_iou_dist: float = 0.5,
    apply_roi: bool = False,
    cityflow_protocol: bool = False,
    pred_id_mode: Literal["auto", "global", "local"] = "auto",
    align_sync: bool = True,
) -> dict:
    """Load GT/predictions and return per-camera + MCMT metric dicts.

    When ``align_sync`` is True and ``gt_root/sync_manifest.json`` exists,
    GT frame indices are shifted per camera to match ``vdo_synch.avi`` and
    predictions are capped to ``sync_length_frames``.
    """
    if cameras is None:
        cameras = [6, 7, 8, 9]

    use_roi = apply_roi or cityflow_protocol
    id_mode = _resolve_id_mode(pred_dir, pred_id_mode)

    gt_by_cam: dict[int, np.ndarray] = {}
    pr_by_cam: dict[int, np.ndarray] = {}
    for cam in cameras:
        gt_path = gt_root / f"c{cam:03d}" / "gt" / "gt.txt"
        pr_path = pred_dir / f"c{cam:03d}.txt"
        if not gt_path.exists():
            print(f"[WARN] GT missing: {gt_path}")
            continue
        gt = load_mot(gt_path)
        pr = load_mot(pr_path)
        if use_roi:
            gt, pr = _apply_roi(gt_root, cam, gt, pr)
        gt_by_cam[cam] = gt
        pr_by_cam[cam] = pr

    if not gt_by_cam:
        raise SystemExit("No GT files found, nothing to evaluate.")

    if align_sync:
        from scripts.cityflow_sync_eval import apply_sync_alignment

        gt_by_cam, pr_by_cam, manifest = apply_sync_alignment(gt_by_cam, pr_by_cam, gt_root)
        if manifest is not None:
            print(
                f"[INFO] Sync-aligned eval: length={manifest.get('sync_length_frames')} "
                f"(skips from sync_manifest.json)"
            )

    if cityflow_protocol:
        pr_by_cam = apply_cityflow_filters(
            gt_by_cam, pr_by_cam, mode=id_mode, iou_thresh=max_iou_dist
        )

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

    return {
        "per_cam": per_cam,
        "mcmt": mcmt_row,
        "cityflow_protocol": cityflow_protocol,
        "pred_id_mode": id_mode,
    }


def _print_summary(summaries: dict[str, "mm.MOTAccumulator"]):
    mh = mm.metrics.create()
    summary = mh.compute_many(
        list(summaries.values()),
        names=list(summaries.keys()),
        metrics=SUMMARY_METRICS,
        generate_overall=True,
    )
    print(mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    ))


def _run_eval_pass(
    gt_root: Path,
    pred_dir: Path,
    cameras: list[int],
    max_iou_dist: float,
    apply_roi: bool,
    cityflow_protocol: bool,
    pred_id_mode: str,
    title: str,
):
    id_mode = _resolve_id_mode(pred_dir, pred_id_mode)
    use_roi = apply_roi or cityflow_protocol

    gt_by_cam: dict[int, np.ndarray] = {}
    pr_by_cam: dict[int, np.ndarray] = {}
    for cam in cameras:
        gt_path = gt_root / f"c{cam:03d}" / "gt" / "gt.txt"
        pr_path = pred_dir / f"c{cam:03d}.txt"
        if not gt_path.exists():
            print(f"[WARN] GT missing: {gt_path}")
            continue
        gt = load_mot(gt_path)
        pr = load_mot(pr_path)
        if use_roi:
            gt, pr = _apply_roi(gt_root, cam, gt, pr)
        gt_by_cam[cam] = gt
        pr_by_cam[cam] = pr

    if not gt_by_cam:
        raise SystemExit("No GT files found, nothing to evaluate.")

    from scripts.cityflow_sync_eval import apply_sync_alignment

    gt_by_cam, pr_by_cam, manifest = apply_sync_alignment(gt_by_cam, pr_by_cam, gt_root)
    if manifest is not None:
        print(
            f"[INFO] Sync-aligned eval: length={manifest.get('sync_length_frames')}"
        )

    if cityflow_protocol:
        pr_by_cam = apply_cityflow_filters(
            gt_by_cam, pr_by_cam, mode=id_mode, iou_thresh=max_iou_dist
        )

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

    print(f"\n=== {title} ===")
    if cityflow_protocol:
        print(f"(CityFlow protocol: ROI + pred filter, id_mode={id_mode})")
    _print_summary(accs)

    if mcmt_gt_rows and mcmt_pr_rows:
        mcmt_gt = np.concatenate(mcmt_gt_rows, axis=0)
        mcmt_pr = np.concatenate(mcmt_pr_rows, axis=0)
        print("\n=== Concatenated multi-camera stream ===")
        _print_summary({"MCMT": accumulate(mcmt_gt, mcmt_pr, max_iou_dist)})


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
    ap.add_argument(
        "--cityflow-protocol",
        action="store_true",
        help="CityFlow eval: ROI + cross-camera pred filter (implies --apply-roi)",
    )
    ap.add_argument(
        "--pred-id-mode",
        choices=["auto", "global", "local"],
        default="auto",
        help="auto: infer from per_cam vs per_cam_local in --pred-dir",
    )
    ap.add_argument(
        "--full-mot",
        action="store_true",
        help="With --cityflow-protocol, also print standard MOT metrics (no CityFlow filter)",
    )
    args = ap.parse_args()

    if args.cityflow_protocol:
        _run_eval_pass(
            args.gt_root,
            args.pred_dir,
            args.cameras,
            args.max_iou_dist,
            apply_roi=args.apply_roi,
            cityflow_protocol=True,
            pred_id_mode=args.pred_id_mode,
            title="Per-camera (CityFlow protocol)",
        )
        if args.full_mot:
            _run_eval_pass(
                args.gt_root,
                args.pred_dir,
                args.cameras,
                args.max_iou_dist,
                apply_roi=args.apply_roi,
                cityflow_protocol=False,
                pred_id_mode=args.pred_id_mode,
                title="Per-camera (full MOT)",
            )
    else:
        _run_eval_pass(
            args.gt_root,
            args.pred_dir,
            args.cameras,
            args.max_iou_dist,
            apply_roi=args.apply_roi,
            cityflow_protocol=False,
            pred_id_mode=args.pred_id_mode,
            title="Per-camera (cross-cam ID-aware if pred uses global ids)",
        )


if __name__ == "__main__":
    main()
