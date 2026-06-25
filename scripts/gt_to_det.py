"""Convert CityFlow GT (MOT) files to CityFlow detection format.

GT:   frame,id,x,y,w,h,conf,-1,-1,-1
Det:  frame,-1,x,y,w,h,conf,-1,-1,-1

Usage:
    python scripts/gt_to_det.py --gt-root datasets/validation/S02 --cameras 6 7 8 9
    python scripts/gt_to_det.py --gt-root datasets/.../S02 --sync   # for vdo_synch.avi
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cityflow_sync_eval import (
    align_gt_to_sync,
    load_sync_manifest,
    sync_length_frames,
    sync_skip_by_cam,
)


def load_gt_mot(gt_path: Path) -> np.ndarray:
    if not gt_path.is_file() or gt_path.stat().st_size == 0:
        return np.empty((0, 10))
    data = np.loadtxt(str(gt_path), delimiter=",")
    if data.size == 0:
        return np.empty((0, 10))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def write_det_file(rows: np.ndarray, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            frame = int(row[0])
            x, y, w, h, conf = row[2:7]
            fout.write(f"{frame},-1,{x:.3f},{y:.3f},{w:.3f},{h:.3f},{conf:.3f},-1,-1,-1\n")
    return len(rows)


def convert_gt_file(
    gt_path: Path,
    out_path: Path,
    *,
    skip_frames: int = 0,
    sync_length: int | None = None,
) -> int:
    gt = load_gt_mot(gt_path)
    if skip_frames or sync_length is not None:
        gt = align_gt_to_sync(gt, skip_frames, sync_length)
    return write_det_file(gt, out_path)


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
    ap.add_argument(
        "--sync",
        action="store_true",
        help="Align GT frame ids to vdo_synch.avi using sync_manifest.json",
    )
    args = ap.parse_args()

    manifest = load_sync_manifest(args.gt_root) if args.sync else None
    skips = sync_skip_by_cam(manifest)
    sync_len = sync_length_frames(manifest)
    if args.sync and manifest is None:
        raise SystemExit(f"--sync requires {args.gt_root / 'sync_manifest.json'}")

    for cam in args.cameras:
        gt_path = args.gt_root / f"c{cam:03d}" / "gt" / "gt.txt"
        if not gt_path.is_file():
            print(f"[SKIP] missing GT: {gt_path}")
            continue
        if args.out_dir is None:
            out_path = gt_path.parent.parent / "det" / "det_gt.txt"
        else:
            out_path = args.out_dir / f"c{cam:03d}.txt"
        n = convert_gt_file(
            gt_path,
            out_path,
            skip_frames=skips.get(cam, 0) if args.sync else 0,
            sync_length=sync_len if args.sync else None,
        )
        suffix = f" (sync skip={skips.get(cam, 0)})" if args.sync else ""
        print(f"c{cam:03d}: {n} dets -> {out_path}{suffix}")


if __name__ == "__main__":
    main()
