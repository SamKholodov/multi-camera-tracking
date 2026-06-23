"""Aggregate CityFlow S02 ablation metrics (full configs_cityflow mirror)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cityflow_ablation_common import GT_ROOT  # noqa: E402
from scripts.eval_s02 import BATCH_METRICS, evaluate_s02, s02_gt_max_frame  # noqa: E402

PRED_ROOT = _ROOT / "outputs" / "configs_cityflow"
CAMERAS = [6, 7, 8, 9]

GROUP_DIRS = (
    "assoc_ablation",
    "reid_ablation",
    "temporal_ablation",
    "kinematic_ablation",
    "trajectory_ablation",
    "geo_ablation",
    "baseline_trackers",
    "byte_ablation",
    "conf_ablation",
    "ema_vs_aaf",
    "latency_ablation",
    "sort",
)

SINGLE_RUNS: tuple[tuple[str, str, Path], ...] = (
    ("baseline", "baseline", PRED_ROOT / "baseline" / "per_cam"),
    ("zone_tracklet", "zone_tracklet", PRED_ROOT / "zone_tracklet" / "per_cam"),
)


def _has_predictions(pred_dir: Path) -> bool:
    return any((pred_dir / f"c{c:03d}.txt").is_file() for c in CAMERAS)


def _pred_max_frame(pred_dir: Path) -> int:
    mx = 0
    for cam in CAMERAS:
        path = pred_dir / f"c{cam:03d}.txt"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            mx = max(mx, int(line.split(",")[0]))
    return mx


def _scalar(value) -> float:
    if hasattr(value, "iloc"):
        return float(value.iloc[0])
    return float(value)


def _metrics_row(result: dict, stream: str) -> dict[str, float]:
    if stream == "mcmt":
        row = result["mcmt"]
        if row is None:
            return {m: float("nan") for m in BATCH_METRICS}
        return {m: _scalar(row[m]) for m in BATCH_METRICS}
    return {m: _scalar(result["per_cam"].loc["OVERALL", m]) for m in BATCH_METRICS}


def _append_run(
    rows: list[dict],
    *,
    group: str,
    name: str,
    pred_dir: Path,
    gt_max: int,
    max_iou_dist: float,
    cityflow_protocol: bool,
    max_frame: int | None,
) -> None:
    if not _has_predictions(pred_dir):
        print(f"[SKIP] {group}/{name}: missing predictions")
        return
    pred_mx = _pred_max_frame(pred_dir)
    eval_max = max_frame if max_frame is not None else min(pred_mx, gt_max) if pred_mx else gt_max
    if pred_mx < gt_max:
        print(f"[WARN] {group}/{name}: incomplete preds (max_frame={pred_mx}, GT={gt_max})")
    result = evaluate_s02(
        GT_ROOT,
        pred_dir,
        cameras=CAMERAS,
        max_iou_dist=max_iou_dist,
        apply_roi=cityflow_protocol,
        cityflow_protocol=cityflow_protocol,
    )
    for stream in ("per_cam", "mcmt"):
        metrics = _metrics_row(result, stream)
        rows.append(
            {
                "group": group,
                "name": name,
                "stream": "sct" if stream == "per_cam" else "mcmt",
                "pred_dir": str(pred_dir),
                "max_iou_dist": max_iou_dist,
                "cityflow_protocol": cityflow_protocol,
                "pred_max_frame": pred_mx,
                "gt_max_frame": gt_max,
                "eval_max_frame": eval_max or "",
                **{m: metrics[m] for m in BATCH_METRICS},
            }
        )
        print(
            f"[OK] {group}/{name}/{stream}: "
            f"IDF1={metrics['idf1']:.4f} MOTA={metrics['mota']:.4f} "
            f"(pred_max={pred_mx})"
        )


def collect_rows(
    max_iou_dist: float,
    cityflow_protocol: bool,
    max_frame: int | None,
) -> list[dict]:
    rows: list[dict] = []
    gt_max = s02_gt_max_frame(GT_ROOT, CAMERAS)

    for group, name, pred_dir in SINGLE_RUNS:
        _append_run(
            rows,
            group=group,
            name=name,
            pred_dir=pred_dir,
            gt_max=gt_max,
            max_iou_dist=max_iou_dist,
            cityflow_protocol=cityflow_protocol,
            max_frame=max_frame,
        )

    for group in GROUP_DIRS:
        group_dir = PRED_ROOT / group
        if not group_dir.is_dir():
            continue
        for run_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            _append_run(
                rows,
                group=group,
                name=run_dir.name,
                pred_dir=run_dir / "per_cam",
                gt_max=gt_max,
                max_iou_dist=max_iou_dist,
                cityflow_protocol=cityflow_protocol,
                max_frame=max_frame,
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-iou-dist", type=float, default=0.5)
    ap.add_argument("--cityflow-protocol", action="store_true")
    ap.add_argument("--max-frame", type=int, default=None)
    ap.add_argument("--output", type=Path, default=PRED_ROOT / "ablation_summary.csv")
    args = ap.parse_args()

    rows = collect_rows(args.max_iou_dist, args.cityflow_protocol, args.max_frame)
    if not rows:
        raise SystemExit("No prediction folders found to evaluate.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "name",
        "stream",
        "pred_dir",
        "max_iou_dist",
        "cityflow_protocol",
        "pred_max_frame",
        "gt_max_frame",
        "eval_max_frame",
        *BATCH_METRICS,
    ]
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
