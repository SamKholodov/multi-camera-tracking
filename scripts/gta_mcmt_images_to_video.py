#!/usr/bin/env python3
"""Assemble GTA MCMT camera JPEG sequences into MP4 videos.

Frame order follows ``coords_cam_N.csv`` (same as the tracker pipeline).
Images are ``cam-N/image_{cam_id}.jpg``.

Usage:
    python scripts/gta_mcmt_images_to_video.py
    python scripts/gta_mcmt_images_to_video.py --out-dir outputs/gta_mcmt_videos
    python scripts/gta_mcmt_images_to_video.py --max-frames 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.io.gta_mcmt import GtaMcmtFrameSource, NUM_CAMERAS

DEFAULT_DATASET = Path("datasets/gta_mcmt")
DEFAULT_OUT = Path("outputs/gta_mcmt_videos")
DEFAULT_FPS = 30
DEFAULT_SIZE = (1920, 1080)  # width, height


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ap.add_argument("--width", type=int, default=DEFAULT_SIZE[0])
    ap.add_argument("--height", type=int, default=DEFAULT_SIZE[1])
    ap.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit frames per camera (default: all snapshots in CSV)",
    )
    ap.add_argument(
        "--codec",
        default="mp4v",
        help="OpenCV fourcc tag (default: mp4v)",
    )
    return ap.parse_args()


def _open_writer(path: Path, fps: float, size: tuple[int, int], codec: str) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec[:4])
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {path} (codec={codec!r})")
    return writer


def _prepare_frame(frame, width: int, height: int):
    h, w = frame.shape[:2]
    if (w, h) == (width, height):
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def export_camera_video(
    cam_dir: Path,
    out_path: Path,
    *,
    fps: float,
    width: int,
    height: int,
    max_frames: int | None,
    codec: str,
) -> int:
    source = GtaMcmtFrameSource(cam_dir)
    total = len(source.snapshots)
    if max_frames is not None:
        total = min(total, max_frames)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _open_writer(out_path, fps, (width, height), codec)

    written = 0
    try:
        while written < total:
            ok, frame = source.read()
            if not ok or frame is None:
                break
            writer.write(_prepare_frame(frame, width, height))
            written += 1
            if written % 500 == 0 or written == total:
                print(f"  {cam_dir.name}: {written}/{total}")
    finally:
        writer.release()
        source.release()

    return written


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Dataset: {dataset_root}\n"
        f"Output:  {out_dir}\n"
        f"FPS: {args.fps}, size: {args.width}x{args.height}"
    )

    for cam in range(NUM_CAMERAS):
        cam_dir = dataset_root / f"cam-{cam}"
        if not cam_dir.is_dir():
            raise FileNotFoundError(f"Missing camera directory: {cam_dir}")

        out_path = out_dir / f"cam-{cam}.mp4"
        print(f"Writing {out_path.name} ...")
        n = export_camera_video(
            cam_dir,
            out_path,
            fps=args.fps,
            width=args.width,
            height=args.height,
            max_frames=args.max_frames,
            codec=args.codec,
        )
        duration_s = n / args.fps if args.fps > 0 else 0.0
        print(f"Done cam-{cam}: {n} frames, ~{duration_s:.1f}s -> {out_path}")

    print(f"All {NUM_CAMERAS} videos saved to {out_dir}")


if __name__ == "__main__":
    main()
