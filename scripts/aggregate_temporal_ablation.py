"""Aggregate GTA temporal / kinematic / trajectory ablation metrics."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eval_gta_mcmt import (  # noqa: E402
    evaluate_gta_mcmt,
    resolve_eval_max_frame,
)
from scripts.eval_s02 import BATCH_METRICS  # noqa: E402

GT_ROOT = _ROOT / "datasets" / "gta_mcmt"
PRED_ROOT = _ROOT / "outputs" / "configs_gta"
GROUPS = (
    "temporal_ablation",
    "kinematic_ablation",
    "trajectory_ablation",
    "geo_ablation",
)


def _has_predictions(pred_dir: Path) -> bool:
    return (pred_dir / "c000.txt").is_file()


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


def collect_rows(max_iou_dist: float, apply_roi: bool, max_frame: int | None) -> list[dict]:
    rows: list[dict] = []
    for group in GROUPS:
        group_dir = PRED_ROOT / group
        if not group_dir.is_dir():
            continue
        for run_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            pred_dir = run_dir / "per_cam"
            if not _has_predictions(pred_dir):
                print(f"[SKIP] {group}/{run_dir.name}: missing predictions")
                continue
            eval_cap, pred_mx, gt_mx = resolve_eval_max_frame(GT_ROOT, pred_dir)
            eval_max = max_frame if max_frame is not None else eval_cap
            if pred_mx < gt_mx:
                print(
                    f"[WARN] {group}/{run_dir.name}: incomplete preds "
                    f"(max_frame={pred_mx}, GT={gt_mx}); eval capped to {eval_max}"
                )
            elif pred_mx > gt_mx:
                print(
                    f"[WARN] {group}/{run_dir.name}: preds past GT "
                    f"(max_frame={pred_mx}, GT={gt_mx})"
                )
            result = evaluate_gta_mcmt(
                GT_ROOT,
                pred_dir,
                max_iou_dist=max_iou_dist,
                max_frame=eval_max,
                apply_roi=apply_roi,
            )
            for stream in ("per_cam", "mcmt"):
                metrics = _metrics_row(result, stream)
                rows.append(
                    {
                        "group": group,
                        "name": run_dir.name,
                        "stream": "sct" if stream == "per_cam" else "mcmt",
                        "pred_dir": str(pred_dir),
                        "max_iou_dist": max_iou_dist,
                        "apply_roi": apply_roi,
                        "pred_max_frame": pred_mx,
                        "gt_max_frame": gt_mx,
                        "eval_max_frame": eval_max or "",
                        **{m: metrics[m] for m in BATCH_METRICS},
                    }
                )
                print(
                    f"[OK] {group}/{run_dir.name}/{stream}: "
                    f"IDF1={metrics['idf1']:.4f} MOTA={metrics['mota']:.4f} "
                    f"(pred_max={pred_mx}, eval_max={eval_max or 'full'})"
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-iou-dist", type=float, default=0.7)
    ap.add_argument("--apply-roi", action="store_true")
    ap.add_argument("--max-frame", type=int, default=None, help="Limit eval frames (default: full pred)")
    ap.add_argument(
        "--output",
        type=Path,
        default=PRED_ROOT / "temporal_ablation_summary.csv",
    )
    args = ap.parse_args()

    rows = collect_rows(args.max_iou_dist, args.apply_roi, args.max_frame)
    if not rows:
        raise SystemExit("No prediction folders found to evaluate.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "name",
        "stream",
        "pred_dir",
        "max_iou_dist",
        "apply_roi",
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
