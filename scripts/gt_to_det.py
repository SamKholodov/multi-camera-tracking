"""Convert CityFlow GT (MOT) files to CityFlow detection format.

GT:   frame,id,x,y,w,h,conf,-1,-1,-1
Det:  frame,-1,x,y,w,h,conf,-1,-1,-1

Usage:
    python scripts/gt_to_det.py --gt-root datasets/validation/S02 --cameras 6 7 8 9
"""
from __future__ import annotations

import argparse
from pathlib import Path


def convert_gt_file(gt_path: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gt_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            frame, _obj_id, x, y, w, h, conf = parts[:7]
            fout.write(f"{frame},-1,{x},{y},{w},{h},{conf},-1,-1,-1\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--cameras", nargs="+", type=int, default=[6, 7, 8, 9])
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <gt-root>/cXXX/det/det_gt.txt per camera",
    )
    args = ap.parse_args()

    for cam in args.cameras:
        gt_path = args.gt_root / f"c{cam:03d}" / "gt" / "gt.txt"
        if not gt_path.is_file():
            print(f"[SKIP] missing GT: {gt_path}")
            continue
        if args.out_dir is None:
            out_path = gt_path.parent.parent / "det" / "det_gt.txt"
        else:
            out_path = args.out_dir / f"c{cam:03d}.txt"
        n = convert_gt_file(gt_path, out_path)
        print(f"c{cam:03d}: {n} dets -> {out_path}")


if __name__ == "__main__":
    main()
