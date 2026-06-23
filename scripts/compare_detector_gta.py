"""Compare GTA detector runs (same pipeline, different weights)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eval_gta_mcmt import evaluate_gta_mcmt


def _scalar(value) -> float:
    if hasattr(value, "iloc"):
        return float(value.iloc[0])
    return float(value)


def _eval_run(gt_root: Path, pred_dir: Path, max_iou_dist: float, max_frame: int) -> dict:
    result = evaluate_gta_mcmt(
        gt_root,
        pred_dir,
        max_iou_dist=max_iou_dist,
        max_frame=max_frame,
        apply_roi=True,
    )
    per_cam = result["per_cam"].loc["OVERALL"]
    mcmt = result["mcmt"]
    out = {
        "idf1": _scalar(per_cam.idf1),
        "idr": _scalar(per_cam.idr),
        "mota": _scalar(per_cam.mota),
        "fp": int(_scalar(per_cam.num_false_positives)),
        "fn": int(_scalar(per_cam.num_misses)),
    }
    if mcmt is not None:
        out["mcmt_idf1"] = _scalar(mcmt.idf1)
        out["mcmt_idr"] = _scalar(mcmt.idr)
        out["mcmt_mota"] = _scalar(mcmt.mota)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-iou-dist", type=float, default=0.7)
    ap.add_argument("--max-frame", type=int, default=2000)
    args = ap.parse_args()

    gt_root = _ROOT / "datasets" / "gta_mcmt"
    runs = [
        ("yolo26l_pretrained", "outputs/configs_gta/conf_ablation/conf_0_20/per_cam"),
        ("yolo26l_pretrained_sct", "outputs/configs_gta/conf_ablation/conf_0_20/per_cam_local"),
        (
            "yolo26l_fine_tune_gta",
            "outputs/configs_gta/detector_ablation/yolo26l_fine_tune_gta_conf_0_20/per_cam",
        ),
        (
            "yolo26l_fine_tune_gta_sct",
            "outputs/configs_gta/detector_ablation/yolo26l_fine_tune_gta_conf_0_20/per_cam_local",
        ),
    ]

    print(f"conf=0.2, max_frame={args.max_frame}, max_iou_dist={args.max_iou_dist}, apply_roi=True\n")
    header = f"{'run':<28} {'IDF1':>7} {'IDR':>7} {'MOTA':>7} {'MCMT_IDF1':>10} {'FP':>8} {'FN':>8}"
    print(header)
    print("-" * len(header))

    for name, rel in runs:
        pred_dir = _ROOT / rel
        if not (pred_dir / "c000.txt").is_file():
            print(f"{name:<28} SKIP (no predictions)")
            continue
        m = _eval_run(gt_root, pred_dir, args.max_iou_dist, args.max_frame)
        mcmt_idf1 = f"{m.get('mcmt_idf1', float('nan')):.1%}" if "mcmt_idf1" in m else "n/a"
        print(
            f"{name:<28} {m['idf1']:>6.1%} {m['idr']:>6.1%} {m['mota']:>6.1%} "
            f"{mcmt_idf1:>10} {m['fp']:>8} {m['fn']:>8}"
        )


if __name__ == "__main__":
    main()
