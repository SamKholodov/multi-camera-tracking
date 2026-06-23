#!/usr/bin/env python3
"""Export one GTA MCMT frame per camera (snapshot with fewest GT detections).

Usage:
    python scripts/export_gta_calib_frames.py
    python scripts/export_gta_calib_frames.py --out-dir outputs/gta_calib_frames
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.io.gta_mcmt import GtaMcmtDataset, NUM_CAMERAS, image_path_for_cam_dir

DEFAULT_DATASET = Path("datasets/gta_mcmt")
DEFAULT_OUT = Path("outputs/gta_calib_frames")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    dataset = GtaMcmtDataset(args.dataset_root)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, object] = {"dataset": str(args.dataset_root.resolve()), "cameras": {}}

    for cam in range(NUM_CAMERAS):
        snaps = dataset.snapshots_by_cam[cam]
        best = min(snaps, key=lambda s: len(s.annotations))
        src = image_path_for_cam_dir(dataset.cam_dirs[cam], best.cam_id)
        dst = out_dir / f"cam{cam}_min_det.jpg"
        shutil.copy2(src, dst)
        meta["cameras"][str(cam)] = {
            "sync_index": best.sync_index,
            "frame_idx": best.frame_idx,
            "image_cam_id": best.cam_id,
            "num_detections": len(best.annotations),
            "source": str(src.resolve()),
            "saved": str(dst.resolve()),
        }
        print(
            f"cam-{cam}: sync={best.sync_index} dets={len(best.annotations)} "
            f"-> {dst.name}"
        )

    (out_dir / "frames.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSaved {NUM_CAMERAS} frames to {out_dir}")
    print(f"Metadata: {out_dir / 'frames.json'}")
    print("\nCalibrate homography (one camera at a time):")
    for cam in range(NUM_CAMERAS):
        frame = out_dir / f"cam{cam}_min_det.jpg"
        calib_out = args.dataset_root / f"cam-{cam}" / "calibration.txt"
        print(
            f"  python scripts/calibrate_homography.py {frame} "
            f"--out {calib_out}"
        )


if __name__ == "__main__":
    main()
