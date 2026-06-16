"""Convert GTA MCMT CSV annotations to per-camera MOT16 GT files.

Output layout:
  {dataset_root}/cam-{N}/gt/gt.txt

MOT row: frame,id,x,y,w,h,conf,-1,-1,-1
  frame = sync_index + 1 (matches pipeline frame_idx + 1)
  id    = global obj_id from CSV
  x,y   = top-left from center bbox (cx - w/2, cy - h/2)

Usage:
    python scripts/convert_gta_mcmt_gt.py
    python scripts/convert_gta_mcmt_gt.py --dataset-root datasets/gta_mcmt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.io.gta_mcmt import GtaMcmtDataset, center_bbox_to_tlwh, NUM_CAMERAS


def convert_camera(dataset: GtaMcmtDataset, cam: int, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for sync_index in range(len(dataset)):
            snap = dataset.snapshot(cam, sync_index)
            frame_id = sync_index + 1
            for ann in snap.annotations:
                x, y, w, h = center_bbox_to_tlwh(ann.cx, ann.cy, ann.w, ann.h)
                fh.write(
                    f"{frame_id},{ann.obj_id},{x:.6f},{y:.6f},{w:.6f},{h:.6f},"
                    f"{ann.confidence:.6f},-1,-1,-1\n"
                )
                rows_written += 1
    return rows_written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=Path("datasets/gta_mcmt"))
    ap.add_argument("--cameras", type=int, nargs="+", default=list(range(NUM_CAMERAS)))
    args = ap.parse_args()

    dataset = GtaMcmtDataset(args.dataset_root)
    print(f"Dataset: {args.dataset_root.resolve()}  synced_frames={len(dataset)}")

    for cam in args.cameras:
        out_path = args.dataset_root / f"cam-{cam}" / "gt" / "gt.txt"
        n = convert_camera(dataset, cam, out_path)
        print(f"cam-{cam}: {n} rows -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
