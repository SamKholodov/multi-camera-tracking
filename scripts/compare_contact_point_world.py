"""Compare GTA MCMT eval metrics for world_anchor ablations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eval_gta_mcmt import evaluate_gta_mcmt
from scripts.eval_s02 import BATCH_METRICS


def _scalar(value) -> float:
    if hasattr(value, "iloc"):
        return float(value.iloc[0])
    return float(value)


def _mcmt_summary(mcmt) -> dict[str, float]:
    if mcmt is None:
        return {m: float("nan") for m in BATCH_METRICS}
    return {m: _scalar(mcmt[m]) for m in BATCH_METRICS if m in mcmt.columns}


def _eval_run(gt_root: Path, pred_dir: Path, max_iou_dist: float, max_frame: int | None) -> dict:
    result = evaluate_gta_mcmt(
        gt_root=gt_root,
        pred_dir=pred_dir,
        apply_roi=True,
        max_iou_dist=max_iou_dist,
        max_frame=max_frame,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare contact point world anchor ablations.")
    parser.add_argument("--gt-root", type=Path, default=Path("datasets/gta_mcmt"))
    parser.add_argument("--max-iou-dist", type=float, default=0.7)
    parser.add_argument("--max-frame", type=int, default=None)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[
            "bottom_center:outputs/configs_gta/geo_ablation/contact_point_world_bottom_center/per_cam",
            "contact_point:outputs/configs_gta/geo_ablation/contact_point_world_contact/per_cam",
        ],
        help="name:pred_dir pairs",
    )
    args = parser.parse_args()

    rows = []
    for item in args.runs:
        name, pred = item.split(":", 1)
        pred_dir = Path(pred)
        if not pred_dir.is_dir():
            print(f"[SKIP] {name}: missing {pred_dir}")
            continue
        metrics = _eval_run(args.gt_root, pred_dir, args.max_iou_dist, args.max_frame)
        summary = _mcmt_summary(metrics.get("mcmt"))
        row = {"name": name, "pred_dir": str(pred_dir), "mcmt": summary}
        rows.append(row)
        print(f"\n=== {name} ===")
        print(json.dumps(summary, indent=2))

    out_path = Path("outputs/configs_gta/geo_ablation/contact_point_world_compare.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
