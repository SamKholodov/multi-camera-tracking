"""Evaluate CityFlow MTSC baselines with official ``datasets/eval/eval.py``.

Converts per-camera ``mtsc/mtsc_*.txt`` (MOTChallenge SCT tracks) to
AICity ``track_results`` format and runs the same protocol as the challenge
(ROI filter, single-cam pred filter, dedupe, IDF1/IDP/IDR).

Note: MTSC files use **local** track ids per camera. ``eval.py`` treats the
same numeric ``Id`` on different cameras as one global object. Numbers are
therefore **not directly comparable** to true MCMT submissions (Luna, your
pipeline with global ids), but show how challenge SCT baselines score under
the official script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.io.calibration import load_homography_image_to_world, project_point

# eval.py imports
sys.path.insert(0, str(_ROOT / "datasets" / "eval"))
from eval import eval as cityflow_eval, readData  # noqa: E402


def mtsc_to_track_results(
    gt_root: Path,
    cameras: list[int],
    mtsc_basename: str,
    out_path: Path,
) -> int:
    rows: list[str] = []
    for cam in cameras:
        mtsc_path = gt_root / f"c{cam:03d}" / "mtsc" / mtsc_basename
        cal_path = gt_root / f"c{cam:03d}" / "calibration.txt"
        if not mtsc_path.is_file():
            raise FileNotFoundError(mtsc_path)
        H_i2w = load_homography_image_to_world(cal_path)
        data = np.loadtxt(str(mtsc_path), delimiter=",", ndmin=2)
        for r in data:
            frame_id = int(r[0])
            obj_id = int(r[1])
            x, y, w, h = float(r[2]), float(r[3]), float(r[4]), float(r[5])
            bcx, bcy = x + w / 2.0, y + h
            xw, yw = project_point(H_i2w, bcx, bcy)
            rows.append(
                f"{cam} {obj_id} {frame_id} {x:.2f} {y:.2f} {w:.2f} {h:.2f} {xw:.6f} {yw:.6f}"
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-root", type=Path, default=Path("datasets/validation/S02"))
    ap.add_argument("--gt-eval", type=Path, default=Path("datasets/validation/S02/gt_for_eval.txt"))
    ap.add_argument("--cameras", nargs="+", type=int, default=[6, 7, 8, 9])
    ap.add_argument("--dstype", default="S02")
    ap.add_argument("--roidir", default="datasets/validation")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/mtsc_cityflow_eval"),
        help="Converted track_results + summary table",
    )
    args = ap.parse_args()

    if not args.gt_eval.is_file():
        raise SystemExit(f"GT eval file missing: {args.gt_eval}")

    mtsc_dir = args.gt_root / "c006" / "mtsc"
    baselines = sorted(p.name for p in mtsc_dir.glob("mtsc_*.txt"))
    test_df = readData(str(args.gt_eval))

    results: list[tuple[str, float, float, float, int]] = []
    for bn in baselines:
        label = bn.replace("mtsc_", "").replace(".txt", "")
        tr_path = args.out_dir / label / "track_results.txt"
        n = mtsc_to_track_results(args.gt_root, args.cameras, bn, tr_path)
        pred_df = readData(str(tr_path))
        summary = cityflow_eval(
            test_df,
            pred_df,
            dstype=args.dstype,
            roidir=args.roidir,
        )
        row = summary.iloc[-1]
        results.append(
            (label, float(row["idf1"]) * 100, float(row["idp"]) * 100, float(row["idr"]) * 100, n)
        )
        print(f"{label}: IDF1={row['idf1']*100:.2f}% rows={n}")

    print("\n=== CityFlow eval.py on MTSC baselines (local ids) ===")
    print(f"{'baseline':24} {'IDF1':>7} {'IDP':>7} {'IDR':>7} {'rows':>8}")
    print("-" * 58)
    for label, idf1, idp, idr, n in sorted(results, key=lambda x: -x[1]):
        print(f"{label:24} {idf1:7.2f} {idp:7.2f} {idr:7.2f} {n:8d}")

    # Reference runs if present
    refs = [
        ("yolo_deepocsort", Path("outputs/s02_baseline/track_results.txt")),
        ("maskrcnn_det", Path("outputs/s02_maskrcnn_det/track_results.txt")),
    ]
    print("\n=== Your MCMT runs (global ids) ===")
    for name, p in refs:
        if not p.is_file():
            continue
        pred_df = readData(str(p))
        summary = cityflow_eval(test_df, pred_df, dstype=args.dstype, roidir=args.roidir)
        row = summary.iloc[-1]
        print(
            f"{name:24} {row['idf1']*100:7.2f} {row['idp']*100:7.2f} {row['idr']*100:7.2f}"
        )


if __name__ == "__main__":
    main()
