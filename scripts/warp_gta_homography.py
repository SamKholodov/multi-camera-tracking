#!/usr/bin/env python3
"""Warp GTA camera frames to a shared BEV canvas (homography check).

World coordinates come from calibration (arbitrary global map units).
BEV pixels: ``(world - origin) * world_scale`` (default scale = 10).

Draws calibration control points when ``calibration_points.json`` exists:
  green = target world (as entered during calibration)
  red   = H(image point)

Usage:
    python scripts/warp_gta_homography.py --cam 0
    python scripts/warp_gta_homography.py --cam 0 --use-calib-image
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

from core.io.calibration import (
    CalibPointPair,
    load_calibration_points,
    load_homography,
    load_homography_image_to_world,
    project_bbox_bottom_center,
    project_point,
    project_world_to_image,
)
from core.io.gta_mcmt import (
    GtaMcmtDataset,
    center_bbox_to_xyxy,
    image_path_for_cam_dir,
)

DEFAULT_OUT = Path("outputs/gta_calib_frames/warp")
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _image_world_bounds(H_i2w: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, float(w), 40)
    ys = np.linspace(float(h) * 0.55, float(h), 30)
    grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float32).reshape(-1, 1, 2)
    world = cv2.perspectiveTransform(grid, H_i2w).reshape(-1, 2)
    return world.min(axis=0), world.max(axis=0)


def _pairs_world_bounds(pairs: list[CalibPointPair], H_i2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts: list[tuple[float, float]] = []
    for p in pairs:
        pts.append((p.world_x, p.world_y))
        pts.append(project_point(H_i2w, p.image_x, p.image_y))
    arr = np.asarray(pts, dtype=np.float64)
    return arr.min(axis=0), arr.max(axis=0)


def _world_to_bev_matrix(min_xy: np.ndarray, world_scale: float) -> np.ndarray:
    return np.array(
        [
            [world_scale, 0.0, -float(min_xy[0]) * world_scale],
            [0.0, world_scale, -float(min_xy[1]) * world_scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _world_to_bev_xy(
    wx: float,
    wy: float,
    min_xy: np.ndarray,
    world_scale: float,
    pad: int,
) -> tuple[int, int]:
    u = (wx - float(min_xy[0])) * world_scale + pad
    v = (wy - float(min_xy[1])) * world_scale + pad
    return int(round(u)), int(round(v))


def warp_to_bev(
    img: np.ndarray,
    H_i2w: np.ndarray,
    min_xy: np.ndarray,
    max_xy: np.ndarray,
    world_scale: float,
    pad: int,
) -> np.ndarray:
    span = max_xy - min_xy
    out_w = max(int(np.ceil(float(span[0]) * world_scale)) + 2 * pad, 1)
    out_h = max(int(np.ceil(float(span[1]) * world_scale)) + 2 * pad, 1)
    T = _world_to_bev_matrix(min_xy, world_scale)
    if pad:
        T = np.array([[1.0, 0.0, float(pad)], [0.0, 1.0, float(pad)], [0.0, 0.0, 1.0]]) @ T
    M = T @ H_i2w
    return cv2.warpPerspective(img, M, (out_w, out_h))


def _draw_calib_pairs_bev(
    bev: np.ndarray,
    pairs: list[CalibPointPair],
    H_i2w: np.ndarray,
    min_xy: np.ndarray,
    world_scale: float,
    pad: int,
) -> list[float]:
    errors: list[float] = []
    for i, p in enumerate(pairs, start=1):
        px_w, py_w = project_point(H_i2w, p.image_x, p.image_y)
        err = float(np.hypot(px_w - p.world_x, py_w - p.world_y))
        errors.append(err)

        gx, gy = _world_to_bev_xy(p.world_x, p.world_y, min_xy, world_scale, pad)
        rx, ry = _world_to_bev_xy(px_w, py_w, min_xy, world_scale, pad)

        if 0 <= gx < bev.shape[1] and 0 <= gy < bev.shape[0]:
            cv2.circle(bev, (gx, gy), 7, (0, 255, 0), -1)
            cv2.putText(bev, str(i), (gx + 8, gy - 8), FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(bev, str(i), (gx + 8, gy - 8), FONT, 0.55, (0, 180, 0), 1, cv2.LINE_AA)
        if 0 <= rx < bev.shape[1] and 0 <= ry < bev.shape[0]:
            cv2.circle(bev, (rx, ry), 7, (0, 0, 255), 2)
        if (
            0 <= gx < bev.shape[1]
            and 0 <= gy < bev.shape[0]
            and 0 <= rx < bev.shape[1]
            and 0 <= ry < bev.shape[0]
        ):
            cv2.line(bev, (gx, gy), (rx, ry), (255, 0, 255), 1, cv2.LINE_AA)
    return errors


def _draw_calib_pairs_image(
    img: np.ndarray,
    pairs: list[CalibPointPair],
    H_w2i: np.ndarray,
    H_i2w: np.ndarray,
) -> np.ndarray:
    check = img.copy()
    for i, p in enumerate(pairs, start=1):
        ix, iy = int(round(p.image_x)), int(round(p.image_y))
        px_w, py_w = project_point(H_i2w, p.image_x, p.image_y)
        rx, ry = project_world_to_image(H_w2i, px_w, py_w)
        gx, gy = project_world_to_image(H_w2i, p.world_x, p.world_y)

        if 0 <= ix < check.shape[1] and 0 <= iy < check.shape[0]:
            cv2.circle(check, (ix, iy), 7, (255, 255, 0), -1)
            cv2.putText(check, str(i), (ix + 8, iy - 8), FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(check, str(i), (ix + 8, iy - 8), FONT, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        irx, iry = int(round(rx)), int(round(ry))
        if 0 <= irx < check.shape[1] and 0 <= iry < check.shape[0]:
            cv2.circle(check, (irx, iry), 7, (0, 0, 255), 2)
        igx, igy = int(round(gx)), int(round(gy))
        if 0 <= igx < check.shape[1] and 0 <= igy < check.shape[0]:
            cv2.circle(check, (igx, igy), 5, (0, 255, 0), 2)
    return check


def _draw_detection_feet_bev(
    bev: np.ndarray,
    H_i2w: np.ndarray,
    min_xy: np.ndarray,
    world_scale: float,
    pad: int,
    annotations,
) -> None:
    for ann in annotations:
        x1, y1, x2, y2 = center_bbox_to_xyxy(ann.cx, ann.cy, ann.w, ann.h)
        wx, wy = project_bbox_bottom_center(H_i2w, x1, y1, x2, y2)
        bx, by = _world_to_bev_xy(wx, wy, min_xy, world_scale, pad)
        if 0 <= bx < bev.shape[1] and 0 <= by < bev.shape[0]:
            cv2.circle(bev, (bx, by), 4, (0, 165, 255), -1)


def _resolve_calib_image(
    cam_dir: Path,
    calib_image_name: str | None,
    dataset_root: Path,
) -> Path | None:
    if not calib_image_name:
        return None
    candidates = [
        cam_dir / calib_image_name,
        dataset_root.parent / "outputs" / "gta_calib_frames" / calib_image_name,
        Path("outputs/gta_calib_frames") / calib_image_name,
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--dataset-root", type=Path, default=Path("datasets/gta_mcmt"))
    ap.add_argument("--images", type=Path, nargs="*", default=None)
    ap.add_argument("--sync-indices", type=int, nargs="*", default=None)
    ap.add_argument(
        "--use-calib-image",
        action="store_true",
        help="Warp the calibration frame (recommended for point check)",
    )
    ap.add_argument("--world-scale", type=float, default=10.0, help="BEV px per world unit")
    ap.add_argument("--pad", type=int, default=20, help="Border padding in BEV px")
    ap.add_argument(
        "--show-detections",
        action="store_true",
        help="Also draw GT bbox feet (orange, not calibration points)",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cam = int(args.cam)
    world_scale = float(args.world_scale)
    pad = int(args.pad)

    cam_dir = args.dataset_root / f"cam-{cam}"
    calib = cam_dir / "calibration.txt"
    if not calib.is_file():
        raise FileNotFoundError(f"Missing {calib}")

    H_w2i = load_homography(calib)
    H_i2w = load_homography_image_to_world(calib, write_cache=False)
    calib_image_name, calib_pairs, calib_reproj = load_calibration_points(calib)
    dataset = GtaMcmtDataset(args.dataset_root)

    if not calib_pairs:
        print(
            "WARNING: calibration_points.json not found — control pairs were not saved.\n"
            "  Re-run calibration to save pairs:\n"
            f"    python scripts/calibrate_homography.py <frame> --out {calib}"
        )
    else:
        print(f"Loaded {len(calib_pairs)} calibration pairs from calibration_points.json")
        if calib_image_name:
            print(f"  calibration image: {calib_image_name}")
        if calib_reproj is not None:
            print(f"  stored reproj error: {calib_reproj:.3f}")

    image_paths: list[Path] = []
    sync_indices: list[int | None] = []

    if args.use_calib_image or (calib_pairs and not args.images and not args.sync_indices):
        calib_path = _resolve_calib_image(cam_dir, calib_image_name, args.dataset_root)
        if calib_path is not None:
            image_paths = [calib_path]
            sync_indices = [None]
        elif args.use_calib_image:
            print(f"WARNING: calibration image missing: {calib_image_name}")
    elif args.images:
        image_paths = [p.resolve() for p in args.images]
        sync_indices = [None] * len(image_paths)
    else:
        syncs = args.sync_indices or [4089, 1000]
        for si in syncs:
            snap = dataset.snapshot(cam, si)
            image_paths.append(image_path_for_cam_dir(dataset.cam_dirs[cam], snap.cam_id))
            sync_indices.append(si)

    imgs: list[np.ndarray] = []
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        h, w = img.shape[:2]
        lo, hi = _image_world_bounds(H_i2w, w, h)
        imgs.append(img)
        mins.append(lo)
        maxs.append(hi)

    if calib_pairs:
        p_lo, p_hi = _pairs_world_bounds(calib_pairs, H_i2w)
        mins.append(p_lo)
        maxs.append(p_hi)

    min_xy = np.min(np.stack(mins), axis=0)
    max_xy = np.max(np.stack(maxs), axis=0)
    margin = 0.05 * (max_xy - min_xy)
    min_xy -= margin
    max_xy += margin
    span = max_xy - min_xy
    print(
        f"world scale={world_scale:g} px/unit  "
        f"world span={span[0]:.2f}x{span[1]:.2f}  "
        f"bev ~{int(span[0] * world_scale)}x{int(span[1] * world_scale)} px"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlays: list[np.ndarray] = []

    for path, img, si in zip(image_paths, imgs, sync_indices):
        tag = f"sync{si}" if si is not None else path.stem

        bev = warp_to_bev(img, H_i2w, min_xy, max_xy, world_scale, pad)
        check = img.copy()

        if calib_pairs:
            bev_errs = _draw_calib_pairs_bev(bev, calib_pairs, H_i2w, min_xy, world_scale, pad)
            check = _draw_calib_pairs_image(check, calib_pairs, H_w2i, H_i2w)
            if bev_errs:
                print(
                    f"  calib point world error: mean={np.mean(bev_errs):.3f}  "
                    f"max={max(bev_errs):.3f}  n={len(bev_errs)}"
                )

        if args.show_detections and si is not None:
            snap = dataset.snapshot(cam, si)
            _draw_detection_feet_bev(bev, H_i2w, min_xy, world_scale, pad, snap.annotations)

        bev_path = args.out_dir / f"cam{cam}_{tag}_bev.jpg"
        check_path = args.out_dir / f"cam{cam}_{tag}_reproj.jpg"
        cv2.imwrite(str(bev_path), bev)
        cv2.imwrite(str(check_path), check)
        overlays.append(bev)
        print(f"{tag}: {path.name}")
        print(f"  BEV   -> {bev_path}")
        if calib_pairs:
            print("  check -> green=target world  red=H(image)  yellow=image click")
        print(f"  reproj -> {check_path}")

    if len(overlays) >= 2:
        h_max = max(im.shape[0] for im in overlays)
        padded = []
        for im in overlays:
            if im.shape[0] < h_max:
                im = cv2.copyMakeBorder(
                    im, 0, h_max - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
            padded.append(im)
        blend = cv2.addWeighted(padded[0], 0.5, padded[1], 0.5, 0)
        blend_path = args.out_dir / f"cam{cam}_bev_blend.jpg"
        cv2.imwrite(str(blend_path), blend)
        print(f"  blend -> {blend_path}")


if __name__ == "__main__":
    main()
