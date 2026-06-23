"""Filter duplicate overlapping tracks from MCMT predictions and compare eval metrics."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.duplicate_track_utils import (
    find_duplicate_pairs,
    global_ids_to_remove,
)
from scripts.eval_gta_mcmt import evaluate_gta_mcmt


def filter_per_cam(
    src_dir: Path,
    dst_dir: Path,
    remove_by_cam: dict[int, set[int]],
) -> dict[int, int]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    removed_rows: dict[int, int] = {}
    for src_file in sorted(src_dir.glob("c*.txt")):
        cam = int(src_file.stem[1:])
        drop = remove_by_cam.get(cam, set())
        out_path = dst_dir / src_file.name
        kept = 0
        removed = 0
        with src_file.open(encoding="utf-8") as fin, out_path.open(
            "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    fout.write(line)
                    continue
                gid = int(parts[1])
                if gid in drop:
                    removed += 1
                    continue
                fout.write(line)
                kept += 1
        removed_rows[cam] = removed
        print(f"  c{cam:03d}: kept {kept}, removed {removed} rows")
    return removed_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/configs_gta/geo_ablation/geo_tight"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <run-dir>/duplicate_analysis/filtered",
    )
    ap.add_argument("--gt-root", type=Path, default=Path("datasets/gta_mcmt"))
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--min-overlap-frames", type=int, default=3)
    ap.add_argument("--max-iou-dist", type=float, default=0.7)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    out_root = args.out_dir or (args.run_dir / "duplicate_analysis" / "filtered")
    out_root.mkdir(parents=True, exist_ok=True)
    filtered_per_cam = out_root / "per_cam"

    pairs = find_duplicate_pairs(
        args.run_dir,
        iou_thresh=args.iou_thresh,
        min_overlap_frames=args.min_overlap_frames,
    )
    remove_by_cam = global_ids_to_remove(pairs)

    report = {
        "source_run_dir": str(args.run_dir),
        "filtered_dir": str(out_root),
        "iou_thresh": args.iou_thresh,
        "min_overlap_frames": args.min_overlap_frames,
        "duplicate_pairs": [
            {
                "cam": p.cam,
                "local_tid1": p.local_tid1,
                "local_tid2": p.local_tid2,
                "global_tid1": p.global_tid1,
                "global_tid2": p.global_tid2,
                "overlap_frames": p.overlap_frames,
                "mean_iou": round(p.mean_iou, 4),
                "removed_global": p.loser_global,
                "kept_global": p.winner_global,
            }
            for p in pairs
        ],
        "removed_global_ids": {
            f"c{cam:03d}": sorted(ids) for cam, ids in sorted(remove_by_cam.items())
        },
    }

    print(f"Found {len(pairs)} duplicate pairs")
    print("Filtering per_cam predictions...")
    removed_rows = filter_per_cam(
        args.run_dir / "per_cam",
        filtered_per_cam,
        remove_by_cam,
    )
    report["removed_rows_by_cam"] = removed_rows

    for extra in ("fps.json",):
        src = args.run_dir / extra
        if src.is_file():
            shutil.copy2(src, out_root / extra)

    report_path = out_root / "filter_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")

    if args.skip_eval:
        return

    print("\n=== Baseline metrics (original) ===")
    baseline = evaluate_gta_mcmt(
        args.gt_root,
        args.run_dir / "per_cam",
        max_iou_dist=args.max_iou_dist,
        apply_roi=True,
    )
    print(baseline["mcmt"].to_string())

    print("\n=== Filtered metrics (duplicate tracks removed) ===")
    filtered = evaluate_gta_mcmt(
        args.gt_root,
        filtered_per_cam,
        max_iou_dist=args.max_iou_dist,
        apply_roi=True,
    )
    print(filtered["mcmt"].to_string())

    compare_path = out_root / "metrics_compare.json"
    metrics = ["idf1", "idp", "idr", "recall", "precision", "mota", "motp"]
    compare = {}
    for key in metrics:
        if key in baseline["mcmt"].columns and key in filtered["mcmt"].columns:
            b = float(baseline["mcmt"][key].iloc[0])
            f = float(filtered["mcmt"][key].iloc[0])
            compare[key] = {"baseline": b, "filtered": f, "delta": f - b}
    compare_path.write_text(json.dumps(compare, indent=2), encoding="utf-8")
    print(f"\nMetric comparison saved to {compare_path}")


if __name__ == "__main__":
    main()
