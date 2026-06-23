"""Draw GTA MCMT GT boxes on a synced multi-camera sample.

Usage:
    python scripts/visualize_gta_mcmt.py
    python scripts/visualize_gta_mcmt.py --sync-indices 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.io.gta_mcmt import (
    GtaMcmtDataset,
    center_bbox_to_xyxy,
    image_path_for_cam_dir,
)
from core.visualization.visualizer import Visualizer

DATASET_ROOT = Path("datasets/gta_mcmt")
NUM_CAMERAS = 4


def clip_box(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    xi1, yi1 = int(round(x1)), int(round(y1))
    xi2, yi2 = int(round(x2)), int(round(y2))
    if xi2 <= 0 or yi2 <= 0 or xi1 >= img_w or yi1 >= img_h:
        return None
    xi1 = max(0, min(xi1, img_w - 1))
    yi1 = max(0, min(yi1, img_h - 1))
    xi2 = max(0, min(xi2, img_w))
    yi2 = max(0, min(yi2, img_h))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    return xi1, yi1, xi2, yi2


def draw_gt_frame(image_path: Path, snapshot, cam: int, sync_index: int) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_h, img_w = img.shape[:2]
    drawn = 0
    for ann in snapshot.annotations:
        x1, y1, x2, y2 = center_bbox_to_xyxy(ann.cx, ann.cy, ann.w, ann.h)
        box = clip_box(x1, y1, x2, y2, img_w, img_h)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        color = Visualizer.color_from_id(ann.obj_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"ID:{ann.obj_id} c{ann.obj_class}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        drawn += 1

    header = (
        f"sync_k={sync_index}  cam-{cam}  frame_idx={snapshot.frame_idx.split('.')[0]}  "
        f"image={image_path.name}  objs={drawn}/{len(snapshot.annotations)}"
    )
    cv2.putText(
        img,
        header,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def stack_camera_views(frames: list[np.ndarray], target_width: int = 960) -> np.ndarray:
    resized = []
    for frame in frames:
        scale = target_width / frame.shape[1]
        h = int(frame.shape[0] * scale)
        resized.append(cv2.resize(frame, (target_width, h)))
    return cv2.vconcat(resized)


def default_sync_indices(max_k: int, count: int = 5) -> list[int]:
    if max_k <= 0:
        return [0]
    if count == 1:
        return [0]
    step = max(max_k // (count - 1), 1)
    indices = [min(i * step, max_k) for i in range(count)]
    seen: set[int] = set()
    unique: list[int] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gta_mcmt_preview"))
    ap.add_argument(
        "--sync-indices",
        type=int,
        nargs="+",
        default=None,
        help="Sequence index k (same k on all 4 cameras = synced moment). Default: 5 evenly spaced.",
    )
    ap.add_argument("--count", type=int, default=5, help="Number of samples if --sync-indices omitted")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = GtaMcmtDataset(root)
    max_k = len(dataset) - 1
    sync_indices = args.sync_indices if args.sync_indices else default_sync_indices(max_k, args.count)
    print(f"Dataset: {root.resolve()}")
    print(f"Sync indices k (same moment on all cameras): {sync_indices}")
    print(f"Output: {out_dir.resolve()}")

    for k in sync_indices:
        cam_frames: list[np.ndarray] = []
        obj_sets: list[set[int]] = []
        frame_ids: list[str] = []

        for cam in range(NUM_CAMERAS):
            snap = dataset.snapshot(cam, k)
            image_path = image_path_for_cam_dir(dataset.cam_dirs[cam], snap.cam_id)
            if not image_path.exists():
                raise FileNotFoundError(f"Missing synced image for cam-{cam}, k={k}: {image_path}")
            frame_ids.append(snap.frame_idx.split(".")[0])
            obj_sets.append({a.obj_id for a in snap.annotations})
            frame = draw_gt_frame(image_path, snap, cam, k)
            cam_frames.append(frame)
            cv2.imwrite(str(out_dir / f"sync{k:05d}_cam{cam}.jpg"), frame)

        mosaic = stack_camera_views(cam_frames)
        cv2.imwrite(str(out_dir / f"sync{k:05d}_all_cams.jpg"), mosaic)

        overlap = set.intersection(*obj_sets) if obj_sets else set()
        print(
            f"k={k:5d}  frame_idx=[{', '.join(frame_ids)}]  "
            f"objs/cam={[len(s) for s in obj_sets]}  shared_obj_ids={len(overlap)}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
