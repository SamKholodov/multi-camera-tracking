"""Overlay GTA MCMT zone polygons on sample camera frames.

Usage:
    python scripts/generate_gta_mcmt_zone_preview.py
    python scripts/generate_gta_mcmt_zone_preview.py --config config/gta_mcmt_zone_polygons.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.io.gta_mcmt import GtaMcmtDataset, NUM_CAMERAS, image_path_for_cam_dir

DEFAULT_CONFIG = Path("config/gta_mcmt_zone_polygons.yaml")
DEFAULT_DATASET = Path("datasets/gta_mcmt")
DEFAULT_OUT = Path("outputs/gta_mcmt_zone_preview")

# BGR colors for global zones z1–z8.
ZONE_COLORS: dict[int, tuple[int, int, int]] = {
    1: (0, 0, 255),
    2: (255, 128, 0),
    3: (0, 200, 0),
    4: (0, 220, 255),
    5: (255, 0, 255),
    6: (255, 255, 0),
    7: (0, 140, 255),
    8: (200, 0, 200),
}


def _load_config(path: Path) -> tuple[dict, dict[int, list[int]]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    zones_raw = data.get("zones") or {}
    zones: dict[int, dict[int, np.ndarray]] = {}
    for cam_s, zone_dict in zones_raw.items():
        cam = int(cam_s)
        zones[cam] = {}
        for zone_s, pts in (zone_dict or {}).items():
            zone_id = int(zone_s)
            arr = np.asarray(pts, dtype=np.int32)
            zones[cam][zone_id] = arr

    merged: dict[int, dict[int, list[int]]] = {}
    merged_raw = data.get("merged_zones") or {}
    for cam_s, mapping in merged_raw.items():
        cam = int(cam_s)
        merged[cam] = {int(z): [int(x) for x in ids] for z, ids in (mapping or {}).items()}
    return zones, merged


def _zone_color(
    zone_id: int,
    cam: int,
    merged: dict[int, dict[int, list[int]]],
) -> tuple[int, int, int]:
    group = merged.get(cam, {}).get(zone_id)
    if not group or len(group) < 2:
        return ZONE_COLORS.get(zone_id, (200, 200, 200))
    colors = [np.array(ZONE_COLORS[z], dtype=np.float32) for z in group if z in ZONE_COLORS]
    if not colors:
        return (200, 200, 200)
    blend = np.mean(colors, axis=0).astype(np.uint8)
    return int(blend[0]), int(blend[1]), int(blend[2])


def _zone_label(
    zone_id: int,
    cam: int,
    merged: dict[int, dict[int, list[int]]],
) -> str:
    group = merged.get(cam, {}).get(zone_id)
    if group and len(group) > 1:
        return "Z" + "+".join(str(z) for z in group)
    return f"Z{zone_id}"


def _polygon_centroid(poly: np.ndarray) -> tuple[int, int]:
    m = cv2.moments(poly.astype(np.float32))
    if abs(m["m00"]) < 1e-6:
        return int(poly[:, 0].mean()), int(poly[:, 1].mean())
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return cx, cy


def render_zone_preview(
    img: np.ndarray,
    cam_zones: dict[int, np.ndarray],
    cam: int,
    merged: dict[int, dict[int, list[int]]],
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = img.copy()
    h, w = img.shape[:2]
    legend_y = 28

    for zone_id in sorted(cam_zones):
        poly = cam_zones[zone_id].reshape(-1, 1, 2)
        color = _zone_color(zone_id, cam, merged)
        label = _zone_label(zone_id, cam, merged)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        colored = np.zeros_like(img)
        colored[:] = color
        region = mask.astype(bool)
        overlay[region] = (
            overlay[region].astype(np.float32) * (1.0 - alpha)
            + colored[region].astype(np.float32) * alpha
        ).astype(np.uint8)

        cv2.polylines(overlay, [poly], True, color, 2, cv2.LINE_AA)
        cx, cy = _polygon_centroid(cam_zones[zone_id])
        cv2.putText(
            overlay,
            label,
            (cx - 20, cy + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            (cx - 20, cy + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            1,
            cv2.LINE_AA,
        )

        cv2.rectangle(overlay, (8, legend_y - 18), (28, legend_y + 2), color, -1)
        cv2.putText(
            overlay,
            label,
            (34, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            (34, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
        legend_y += 24

    cv2.putText(
        overlay,
        f"cam-{cam}",
        (w - 120, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"cam-{cam}",
        (w - 120, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return overlay


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cameras", type=int, nargs="+", default=list(range(NUM_CAMERAS)))
    ap.add_argument("--alpha", type=float, default=0.45)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    zones, merged = _load_config(args.config)
    dataset = GtaMcmtDataset(args.dataset_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: {args.config.resolve()}")
    print(f"Output: {args.out_dir.resolve()}")

    for cam in args.cameras:
        if cam not in zones:
            print(f"cam-{cam}: skip (no zones in config)")
            continue
        snap = dataset.snapshot(cam, 0)
        image_path = image_path_for_cam_dir(dataset.cam_dirs[cam], snap.cam_id)
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Missing image: {image_path}")

        preview = render_zone_preview(
            img,
            zones[cam],
            cam,
            merged,
            alpha=args.alpha,
        )
        out_path = args.out_dir / f"cam{cam}_zones_preview.jpg"
        cv2.imwrite(str(out_path), preview)
        zone_list = ", ".join(_zone_label(z, cam, merged) for z in sorted(zones[cam]))
        print(f"cam-{cam}: {out_path}  zones=[{zone_list}]")

    print("Done.")


if __name__ == "__main__":
    main()
