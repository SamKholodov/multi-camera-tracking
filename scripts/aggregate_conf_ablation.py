"""Aggregate GTA conf_thres ablation results and suggest the best threshold."""
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
PRED_ROOT = Path("outputs/configs_gta/conf_ablation")
SUMMARY_PATH = PRED_ROOT / "summary.csv"
PRIMARY_METRIC = "idf1"
STREAM = "mcmt"


def _conf_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "_")


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


def collect_rows(conf_values: list[float], max_iou_dist: float, apply_roi: bool, max_frame: int | None) -> list[dict]:
    rows: list[dict] = []
    for conf in conf_values:
        tag = _conf_tag(conf)
        pred_dir = PRED_ROOT / f"conf_{tag}" / "per_cam"
        if not (pred_dir / "c000.txt").is_file():
            print(f"[SKIP] conf={conf}: missing predictions in {pred_dir}")
            continue
        result = evaluate_gta_mcmt(
            GT_ROOT,
            pred_dir,
            max_iou_dist=max_iou_dist,
            max_frame=max_frame,
            apply_roi=apply_roi,
        )
        for stream in ("per_cam", "mcmt"):
            metrics = _metrics_row(result, stream)
            rows.append(
                {
                    "conf_thres": conf,
                    "stream": "sct" if stream == "per_cam" else "mcmt",
                    "pred_dir": str(pred_dir),
                    "max_iou_dist": max_iou_dist,
                    "apply_roi": apply_roi,
                    **{m: metrics[m] for m in BATCH_METRICS},
                }
            )
            if stream == STREAM:
                print(
                    f"[OK] conf={conf:.2f} mcmt: "
                    f"IDF1={metrics['idf1']:.4f} MOTA={metrics['mota']:.4f} "
                    f"IDR={metrics['idr']:.4f}"
                )
    return rows


def _pick_best_mcmt(rows: list[dict]) -> dict | None:
    mcmt_rows = [r for r in rows if r["stream"] == STREAM]
    if not mcmt_rows:
        return None
    return max(
        mcmt_rows,
        key=lambda r: (
            r.get(PRIMARY_METRIC, float("-inf")),
            r.get("mota", float("-inf")),
            r.get("idr", float("-inf")),
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--values",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6",
        help="Confidence thresholds to evaluate",
    )
    ap.add_argument("--max-iou-dist", type=float, default=0.7)
    ap.add_argument(
        "--max-frame",
        type=int,
        default=2000,
        help="Evaluate only frames <= this id (must match multi_camera.max_frames in conf ablation configs)",
    )
    ap.add_argument("--apply-roi", action="store_true")
    ap.add_argument("--output", type=Path, default=SUMMARY_PATH)
    args = ap.parse_args()

    conf_values = [float(v.strip()) for v in args.values.split(",") if v.strip()]
    rows = collect_rows(conf_values, args.max_iou_dist, args.apply_roi, args.max_frame)
    if not rows:
        raise SystemExit("No conf ablation predictions found to evaluate.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "conf_thres",
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

    best = _pick_best_mcmt(rows)
    print(f"\nWrote {len(rows)} rows to {args.output}")
    if best is not None:
        conf = best["conf_thres"]
        print(
            f"\nBest by MCMT {PRIMARY_METRIC.upper()}: conf_thres={conf:.2f} "
            f"({PRIMARY_METRIC}={best[PRIMARY_METRIC]:.4f}, "
            f"mota={best['mota']:.4f}, idr={best['idr']:.4f})"
        )
        print(f"\nApply with:")
        print(f"  python scripts/apply_gta_conf_thres.py --conf {conf:.2f}")
        print(f"  .\\scripts\\run_gta_ablations.ps1")


if __name__ == "__main__":
    main()
