"""Synchronize AICity S02 camera videos using cam_timestamp + black-frame trim.

Writes vdo_synch.avi per camera and sync_manifest.json under validation/S02/.

Usage:
    python scripts/sync_aicity_s02_videos.py
    python scripts/sync_aicity_s02_videos.py --force
    python scripts/sync_aicity_s02_videos.py --black-threshold 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cityflow_ablation_common import (  # noqa: E402
    AICITY_ROOT_REL,
    S02_CAM_IDS,
    S02_ROOT_REL,
    VIDEO_FPS,
)

TIMESTAMP_FILE = _ROOT / AICITY_ROOT_REL / "cam_timestamp" / "S02.txt"
S02_DIR = _ROOT / S02_ROOT_REL
MANIFEST_PATH = S02_DIR / "sync_manifest.json"


def _load_timestamps() -> dict[int, float]:
    if not TIMESTAMP_FILE.is_file():
        raise FileNotFoundError(f"Missing timestamp file: {TIMESTAMP_FILE}")
    ts: dict[int, float] = {}
    for line in TIMESTAMP_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cam_token, sec = line.split()
        cam_id = int(cam_token.lstrip("c"))
        ts[cam_id] = float(sec)
    return ts


def _detect_black_skip(cap: cv2.VideoCapture, max_scan: int, threshold: float) -> int:
    skip = 0
    for _ in range(max_scan):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if float(frame.mean()) >= threshold:
            break
        skip += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return skip


def _write_synch(
    src: Path,
    dst: Path,
    skip: int,
    max_frames: int | None,
    fps: float,
) -> int:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src}")

    if skip > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        raise RuntimeError(f"Invalid frame size for {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create writer: {dst}")

    written = 0
    while True:
        if max_frames is not None and written >= max_frames:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        written += 1

    writer.release()
    cap.release()
    return written


def sync_scene(
    *,
    force: bool,
    black_threshold: float,
    max_black_scan: int,
    fps: float,
) -> dict:
    timestamps = _load_timestamps()
    ref_cam = S02_CAM_IDS[0]
    ref_ts = timestamps.get(ref_cam, 0.0)

    skips: dict[int, int] = {}
    available_after_skip: dict[int, int] = {}

    for cam in S02_CAM_IDS:
        src = S02_DIR / f"c{cam:03d}" / "vdo.avi"
        if not src.is_file():
            raise FileNotFoundError(f"Missing source video: {src}")
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {src}")
        black_skip = _detect_black_skip(cap, max_black_scan, black_threshold)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        ts_skip = round((timestamps.get(cam, 0.0) - ref_ts) * fps)
        skip = max(black_skip, ts_skip)
        skips[cam] = skip
        available_after_skip[cam] = max(0, total - skip)

    sync_len = min(available_after_skip.values())
    if sync_len <= 0:
        raise RuntimeError("No frames left after sync skips")

    manifest = {
        "scene": "S02",
        "fps": fps,
        "reference_camera": ref_cam,
        "sync_length_frames": sync_len,
        "cameras": [],
    }

    for cam in S02_CAM_IDS:
        src = S02_DIR / f"c{cam:03d}" / "vdo.avi"
        dst = S02_DIR / f"c{cam:03d}" / "vdo_synch.avi"
        skip = skips[cam]
        if dst.is_file() and not force:
            print(f"[SKIP] c{cam:03d}: {dst.name} exists (use --force to rebuild)")
        else:
            n = _write_synch(src, dst, skip, sync_len, fps)
            print(f"[OK] c{cam:03d}: skip={skip} wrote {n} frames -> {dst.relative_to(_ROOT)}")
        manifest["cameras"].append(
            {
                "cam_id": cam,
                "source": src.relative_to(_ROOT).as_posix(),
                "output": dst.relative_to(_ROOT).as_posix(),
                "black_skip": skips[cam],
                "timestamp_skip": round((timestamps.get(cam, 0.0) - ref_ts) * fps),
                "skip_frames": skip,
                "sync_length_frames": sync_len,
            }
        )

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {MANIFEST_PATH.relative_to(_ROOT)} (sync_len={sync_len})")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="Rebuild vdo_synch.avi even if present")
    ap.add_argument("--black-threshold", type=float, default=10.0)
    ap.add_argument("--max-black-scan", type=int, default=120)
    ap.add_argument("--fps", type=float, default=VIDEO_FPS)
    args = ap.parse_args()
    sync_scene(
        force=args.force,
        black_threshold=args.black_threshold,
        max_black_scan=args.max_black_scan,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
