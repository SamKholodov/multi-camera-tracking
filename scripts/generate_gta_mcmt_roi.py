"""Generate CityFlow-style roi.jpg masks for GTA MCMT cameras.

Exclusion polygons (black zones) define areas that are NOT part of the ROI.
The rest of the frame is white (valid tracking / eval region).

Usage:
    python scripts/generate_gta_mcmt_roi.py
    python scripts/generate_gta_mcmt_roi.py --preview
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

from core.io.gta_mcmt import GtaMcmtDataset, NUM_CAMERAS, image_path_for_cam_dir
from core.io.roi import build_roi_mask_from_exclusions, write_roi_mask

DATASET_ROOT = Path("datasets/gta_mcmt")

# Zones that must NOT be included in ROI (image pixel coordinates, x right / y down).
GTA_MCMT_EXCLUSION_POLYGONS: dict[int, list[np.ndarray]] = {
    0: [
        np.array(
            [
                [-1, 269],
                [120, 268],
                [216, 259],
                [813, 392],
                [1221, 385],
                [1548, 360],
                [1669, 398],
                [1919, 432],
                [1918, 6],
                [4, 3],
            ],
            dtype=np.int32,
        )
    ],
    1: [
        np.array(
            [
                [4, 410],
                [221, 399],
                [595, 350],
                [898, 310],
                [998, 305],
                [1068, 326],
                [1318, 335],
                [1387, 311],
                [1919, 358],
                [1916, 1],
                [4, 1],
            ],
            dtype=np.int32,
        )
    ],
    2: [
        np.array(
            [
                [586, 385],
                [932, 365],
                [1066, 361],
                [1207, 0],
                [7, 1],
                [4, 369],
            ],
            dtype=np.int32,
        )
    ],
    3: [
        np.array(
            [
                [774, 235],
                [1085, 238],
                [1287, 4],
                [527, 4],
                [711, 238],
            ],
            dtype=np.int32,
        )
    ],
}


def _sample_image_size(dataset: GtaMcmtDataset, cam: int) -> tuple[int, int]:
    snap = dataset.snapshot(cam, 0)
    image_path = image_path_for_cam_dir(dataset.cam_dirs[cam], snap.cam_id)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read sample image: {image_path}")
    h, w = img.shape[:2]
    return w, h


def generate_roi_for_cam(
    dataset: GtaMcmtDataset,
    cam: int,
    *,
    width: int | None = None,
    height: int | None = None,
) -> np.ndarray:
    if cam not in GTA_MCMT_EXCLUSION_POLYGONS:
        raise KeyError(f"No exclusion polygons defined for cam-{cam}")
    if width is None or height is None:
        width, height = _sample_image_size(dataset, cam)
    return build_roi_mask_from_exclusions(
        width, height, GTA_MCMT_EXCLUSION_POLYGONS[cam]
    )


def save_preview(
    dataset: GtaMcmtDataset,
    cam: int,
    mask: np.ndarray,
    out_path: Path,
) -> None:
    snap = dataset.snapshot(cam, 0)
    image_path = image_path_for_cam_dir(dataset.cam_dirs[cam], snap.cam_id)
    img = cv2.imread(str(image_path))
    if img is None:
        return
    overlay = img.copy()
    excluded = mask == 0
    overlay[excluded] = (overlay[excluded] * 0.35 + np.array([0, 0, 255]) * 0.65).astype(
        np.uint8
    )
    contours, _ = cv2.findContours(
        (255 - mask).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    ap.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("outputs/gta_mcmt_roi_preview"),
        help="If set, write cam preview overlays (use --preview to enable)",
    )
    ap.add_argument("--preview", action="store_true", help="Save preview overlays")
    ap.add_argument("--cameras", type=int, nargs="+", default=list(range(NUM_CAMERAS)))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    dataset = GtaMcmtDataset(args.dataset_root)
    width, height = _sample_image_size(dataset, 0)
    print(f"Dataset: {args.dataset_root.resolve()}  frame_size={width}x{height}")

    for cam in args.cameras:
        mask = generate_roi_for_cam(dataset, cam, width=width, height=height)
        roi_path = args.dataset_root / f"cam-{cam}" / "roi.jpg"
        write_roi_mask(roi_path, mask)
        inside_pct = 100.0 * np.count_nonzero(mask) / mask.size
        print(f"cam-{cam}: {roi_path}  inside_ROI={inside_pct:.1f}%")
        if args.preview:
            preview_path = args.preview_dir / f"cam{cam}_roi_preview.jpg"
            save_preview(dataset, cam, mask, preview_path)
            print(f"  preview -> {preview_path}")

    print("Done. Enable in config: multi_camera.roi: auto")


if __name__ == "__main__":
    main()
