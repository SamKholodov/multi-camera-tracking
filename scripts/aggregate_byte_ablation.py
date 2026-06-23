"""Aggregate BYTE ablation metrics into a single CSV summary."""
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
PRED_ROOT = Path("outputs/configs_gta/byte_ablation")

RUNS = [
    "byte_off",
    "byte_on_det03",
    "byte_on_det02",
    "byte_on_narrow",
]


def _scalar(value) -> float:
    if hasattr(value, "iloc"):
        return float(value.iloc[0])
    return float(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-iou-dist", type=float, default=0.7)
    ap.add_argument("--apply-roi", action="store_true", default=True)
    ap.add_argument(
        "--output",
        type=Path,
        default=PRED_ROOT / "summary.csv",
    )
    args = ap.parse_args()

    rows: list[dict] = []
    for name in RUNS:
        pred_dir = PRED_ROOT / name / "per_cam"
        if not (pred_dir / "c000.txt").is_file():
            print(f"[SKIP] {name}: missing {pred_dir}")
            continue
        result = evaluate_gta_mcmt(
            GT_ROOT,
            pred_dir,
            max_iou_dist=args.max_iou_dist,
            apply_roi=args.apply_roi,
        )
        for stream_key, stream_name in (("per_cam", "sct"), ("mcmt", "mcmt")):
            if stream_key == "mcmt":
                row = result["mcmt"]
                if row is None:
                    metrics = {m: float("nan") for m in BATCH_METRICS}
                else:
                    metrics = {m: _scalar(row[m]) for m in BATCH_METRICS}
            else:
                metrics = {
                    m: _scalar(result["per_cam"].loc["OVERALL", m])
                    for m in BATCH_METRICS
                }
            rows.append(
                {
                    "name": name,
                    "stream": stream_name,
                    "pred_dir": str(pred_dir),
                    **metrics,
                }
            )
            print(
                f"[OK] {name}/{stream_name}: "
                f"IDF1={metrics['idf1']:.4f} MOTA={metrics['mota']:.4f}"
            )

    if not rows:
        raise SystemExit("No predictions found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "stream", "pred_dir", *BATCH_METRICS]
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
