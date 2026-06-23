"""Aggregate GTA MCMT ablation metrics into a single CSV summary."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eval_gta_mcmt import evaluate_gta_mcmt
from scripts.eval_s02 import BATCH_METRICS

GT_ROOT = Path("datasets/gta_mcmt")
PRED_ROOT = Path("outputs/configs_gta")

RUNS: list[tuple[str, str, str]] = [
    ("baseline_trackers", "sort", "baseline_trackers/sort/per_cam"),
    ("baseline_trackers", "ocsort", "baseline_trackers/ocsort/per_cam"),
    ("baseline_trackers", "deepocsort", "baseline_trackers/deepocsort/per_cam"),
    ("baseline_trackers", "botsort", "baseline_trackers/botsort/per_cam"),
    ("reid_ablation", "osnet_ibn_msmt17", "reid_ablation/osnet_ibn_msmt17/per_cam"),
    ("reid_ablation", "vehicle_osnet_veri_vric", "reid_ablation/vehicle_osnet_veri_vric/per_cam"),
    (
        "reid_ablation",
        "vehicle_osnet_view_finetune",
        "reid_ablation/vehicle_osnet_view_finetune/per_cam",
    ),
    (
        "reid_ablation",
        "vehicle_osnet_veri_vric_wild_epoch120",
        "reid_ablation/vehicle_osnet_veri_vric_wild_epoch120/per_cam",
    ),
    ("assoc_ablation", "reid_only", "assoc_ablation/reid_only/per_cam"),
    ("assoc_ablation", "+zone_tracklet", "assoc_ablation/+zone_tracklet/per_cam"),
    ("assoc_ablation", "no_different_cam_geo_tiers", "assoc_ablation/no_different_cam_geo_tiers/per_cam"),
]


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


def collect_rows(max_iou_dist: float, apply_roi: bool) -> list[dict]:
    rows: list[dict] = []
    for group, name, rel_pred in RUNS:
        pred_dir = PRED_ROOT / rel_pred
        if not _has_predictions(pred_dir):
            print(f"[SKIP] {group}/{name}: missing predictions in {pred_dir}")
            continue
        try:
            result = evaluate_gta_mcmt(
                GT_ROOT,
                pred_dir,
                max_iou_dist=max_iou_dist,
                max_frame=None,
                apply_roi=apply_roi,
            )
        except SystemExit as exc:
            print(f"[SKIP] {group}/{name}: {exc}")
            continue

        for stream in ("per_cam", "mcmt"):
            metrics = _metrics_row(result, stream)
            rows.append(
                {
                    "group": group,
                    "name": name,
                    "stream": "sct" if stream == "per_cam" else "mcmt",
                    "pred_dir": str(pred_dir),
                    "max_iou_dist": max_iou_dist,
                    "apply_roi": apply_roi,
                    **{m: metrics[m] for m in BATCH_METRICS},
                }
            )
            print(
                f"[OK] {group}/{name}/{stream}: "
                f"IDF1={metrics['idf1']:.4f} MOTA={metrics['mota']:.4f}"
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-iou-dist", type=float, default=0.7)
    ap.add_argument(
        "--apply-roi",
        action="store_true",
        help="Filter GT and predictions with cam-N/roi.jpg",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=PRED_ROOT / "ablation_summary.csv",
    )
    args = ap.parse_args()

    rows = collect_rows(args.max_iou_dist, args.apply_roi)
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
        *BATCH_METRICS,
    ]
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
