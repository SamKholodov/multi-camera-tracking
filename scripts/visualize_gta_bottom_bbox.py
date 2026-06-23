"""Draw 3D-OBB projection envelope and bottom-center GT on GTA MCMT frames.

Usage:
    python scripts/visualize_gta_bottom_bbox.py \\
        --dataset-root "C:/.../MTMCT" \\
        --num-frames 20 \\
        --out-dir test_bbox5
"""
from __future__ import annotations

import argparse
import math
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
    coords_csv_path,
    image_path_for_cam_dir,
    load_snapshots,
)
from core.visualization.visualizer import Visualizer


def clip_point(x: float, y: float, img_w: int, img_h: int) -> tuple[int, int] | None:
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or yi < 0 or xi >= img_w or yi >= img_h:
        return None
    return xi, yi


def draw_bottom_marker(img: np.ndarray, x: float, y: float, color: tuple[int, int, int]) -> None:
    pt = clip_point(x, y, img.shape[1], img.shape[0])
    if pt is None:
        return
    cv2.drawMarker(img, pt, color, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
    cv2.circle(img, pt, 6, color, -1, lineType=cv2.LINE_AA)


def draw_yaw_arrow(
    img: np.ndarray,
    x: float,
    y: float,
    yaw_deg: float,
    length: float,
    color: tuple[int, int, int],
) -> None:
    """World yaw hint at bottom point (approximate direction on image)."""
    rad = math.radians(yaw_deg)
    x2 = x + length * math.sin(rad)
    y2 = y - length * math.cos(rad)
    p1 = clip_point(x, y, img.shape[1], img.shape[0])
    p2 = clip_point(x2, y2, img.shape[1], img.shape[0])
    if p1 and p2:
        cv2.arrowedLine(img, p1, p2, color, 2, tipLength=0.25)


def bbox_drawable(
    ann, img_w: int, img_h: int
) -> tuple[bool, tuple[float, float, float, float] | None]:
    x1, y1, x2, y2 = center_bbox_to_xyxy(ann.cx, ann.cy, ann.w, ann.h)
    xi1, yi1 = int(round(x1)), int(round(y1))
    xi2, yi2 = int(round(x2)), int(round(y2))
    if xi2 <= 0 or yi2 <= 0 or xi1 >= img_w or yi1 >= img_h:
        return False, None
    xi1 = max(0, min(xi1, img_w - 1))
    yi1 = max(0, min(yi1, img_h - 1))
    xi2 = max(0, min(xi2, img_w))
    yi2 = max(0, min(yi2, img_h))
    if xi2 <= xi1 or yi2 <= yi1:
        return False, None
    return True, (x1, y1, x2, y2)


def bottom_valid(ann) -> bool:
    return (
        ann.bottom_px is not None
        and ann.bottom_py is not None
        and ann.bottom_px >= 0
        and ann.bottom_py >= 0
    )


def draw_frame(image_path: Path, snapshot, sync_index: int, cam: int) -> tuple[np.ndarray, int]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_h, img_w = img.shape[:2]
    drawn = 0
    for ann in snapshot.annotations:
        ok, xyxy = bbox_drawable(ann, img_w, img_h)
        if not ok or xyxy is None:
            continue

        x1, y1, x2, y2 = xyxy
        xi1, yi1 = int(round(x1)), int(round(y1))
        xi2, yi2 = int(round(x2)), int(round(y2))

        color = Visualizer.color_from_id(ann.obj_id)
        cv2.rectangle(img, (xi1, yi1), (xi2, yi2), color, 2, lineType=cv2.LINE_AA)

        aabb_bcx = (x1 + x2) / 2.0
        aabb_bcy = y2
        draw_bottom_marker(img, aabb_bcx, aabb_bcy, (0, 220, 255))
        if bottom_valid(ann):
            draw_bottom_marker(img, ann.bottom_px, ann.bottom_py, (0, 0, 255))
            if ann.yaw is not None:
                draw_yaw_arrow(
                    img,
                    ann.bottom_px,
                    ann.bottom_py,
                    ann.yaw,
                    length=ann.w * 0.35,
                    color=(255, 180, 0),
                )

        cv2.putText(
            img,
            f"id={ann.obj_id}",
            (xi1, max(0, yi1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        drawn += 1

    header = (
        f"sync_k={sync_index}  cam-{cam}  image={image_path.name}"
        f"  csv={len(snapshot.annotations)}  drawn={drawn}"
    )
    cv2.putText(img, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        img,
        "box=3D-OBB proj  red=bottom_gt  cyan=aabb_bottom  orange=yaw",
        (10, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return img, drawn


def detect_num_cameras(dataset_root: Path) -> int:
    n = 0
    while (dataset_root / f"cam-{n}").is_dir():
        n += 1
    return n


def stack_camera_grid(frames: list[np.ndarray], target_width: int = 640) -> np.ndarray:
    if not frames:
        raise ValueError("stack_camera_grid requires at least one frame")

    resized: list[np.ndarray] = []
    for frame in frames:
        scale = target_width / frame.shape[1]
        h = int(frame.shape[0] * scale)
        resized.append(cv2.resize(frame, (target_width, h)))

    n = len(resized)
    ncol = 3 if n == 9 else (4 if n in (7, 8, 12) else min(4, n))
    rows: list[np.ndarray] = []
    blank_h = resized[0].shape[0]
    blank_w = target_width

    for start in range(0, n, ncol):
        row = resized[start : start + ncol]
        if len(row) < ncol:
            pad = ncol - len(row)
            row = row + [np.zeros((blank_h, blank_w, 3), dtype=np.uint8)] * pad
        rows.append(cv2.hconcat(row))

    return cv2.vconcat(rows)


def evenly_spaced_indices(max_k: int, count: int) -> list[int]:
    if max_k < 0:
        return []
    if max_k == 0:
        return [0]
    if count <= 1:
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
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"C:\Program Files\Rockstar Games\Grand Theft Auto V Legacy\MTMCT"),
    )
    ap.add_argument("--num-frames", type=int, default=20)
    ap.add_argument(
        "--sync-indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit sync indices k; default: evenly spaced over dataset",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("test_bbox7"))
    ap.add_argument(
        "--num-cameras",
        type=int,
        default=None,
        help="Number of cameras (default: auto-detect cam-0, cam-1, ...)",
    )
    ap.add_argument(
        "--cameras",
        type=int,
        nargs="+",
        default=None,
        help="Camera indices to render (default: 0..num_cameras-1)",
    )
    ap.add_argument(
        "--per-camera",
        action="store_true",
        help="Pick frames per camera from its own CSV (no cross-camera sync k)",
    )
    ap.add_argument(
        "--min-drawn",
        type=int,
        default=1,
        help="Save frame only if at least this many boxes are drawn",
    )
    ap.add_argument(
        "--mosaic-width",
        type=int,
        default=960,
        help="Tile width in all_cams mosaic (larger keeps small boxes visible)",
    )
    return ap.parse_args()


def synced_length_for_cameras(dataset_root: Path, cameras: list[int]) -> int:
    lengths: list[int] = []
    for cam in cameras:
        cam_dir = dataset_root / f"cam-{cam}"
        csv_path = coords_csv_path(cam_dir)
        if not csv_path.is_file():
            lengths.append(0)
            continue
        lengths.append(len(load_snapshots(csv_path)))
    return min(lengths) if lengths else 0


def camera_report(dataset_root: Path, cameras: list[int]) -> None:
    print("\n=== Annotation coverage ===")
    for cam in cameras:
        cam_dir = dataset_root / f"cam-{cam}"
        csv_path = coords_csv_path(cam_dir)
        if not csv_path.is_file():
            print(f"  cam-{cam}: NO CSV")
            continue
        snaps = load_snapshots(csv_path)
        imgs = len(list(cam_dir.glob("image_*.jpg")))
        with_anns = sum(1 for s in snaps if s.annotations)
        print(
            f"  cam-{cam}: csv_frames={len(snaps)} images={imgs}"
            f" frames_with_anns={with_anns} img_without_csv={imgs - len(snaps)}"
        )


def evenly_spaced_snapshot_indices(num_snaps: int, count: int) -> list[int]:
    if num_snaps <= 0:
        return []
    if count <= 1:
        return [0]
    step = max((num_snaps - 1) // (count - 1), 1)
    indices = [min(i * step, num_snaps - 1) for i in range(count)]
    seen: set[int] = set()
    unique: list[int] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique


def run_per_camera(
    dataset_root: Path,
    cameras: list[int],
    out_dir: Path,
    num_frames: int,
    min_drawn: int,
    mosaic_width: int,
) -> int:
    saved = 0
    for cam in cameras:
        cam_dir = dataset_root / f"cam-{cam}"
        snaps = load_snapshots(coords_csv_path(cam_dir))
        candidate_idx = [i for i, s in enumerate(snaps) if s.annotations]
        if not candidate_idx:
            print(f"  cam-{cam}: no frames with CSV annotations")
            continue

        pick = evenly_spaced_snapshot_indices(len(candidate_idx), num_frames)
        cam_frames: list[np.ndarray] = []
        for pick_i in pick:
            snap_i = candidate_idx[pick_i]
            snap = snaps[snap_i]
            image_path = image_path_for_cam_dir(cam_dir, snap.cam_id)
            if not image_path.is_file():
                print(f"  [skip] cam-{cam} missing {image_path.name}")
                continue
            frame, drawn = draw_frame(image_path, snap, snap_i, cam)
            if drawn < min_drawn:
                print(f"  [skip] cam-{cam} snap={snap_i} drawn={drawn} < {min_drawn}")
                continue
            out_path = out_dir / f"cam{cam}_snap{snap_i:05d}_{image_path.stem}.jpg"
            cv2.imwrite(str(out_path), frame)
            cam_frames.append(frame)
            saved += 1
            print(f"  cam-{cam} snap={snap_i:5d} image={image_path.name} csv={len(snap.annotations)} drawn={drawn}")

        if cam_frames:
            mosaic = stack_camera_grid(cam_frames, target_width=mosaic_width)
            cv2.imwrite(str(out_dir / f"cam{cam}_mosaic.jpg"), mosaic)
            saved += 1

    return saved


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if out_dir.exists():
        for old in out_dir.glob("*.jpg"):
            old.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    num_cameras = args.num_cameras
    if num_cameras is None:
        num_cameras = detect_num_cameras(args.dataset_root)
    if num_cameras <= 0:
        raise SystemExit(f"No cam-* folders under {args.dataset_root}")

    cameras = args.cameras if args.cameras is not None else list(range(num_cameras))
    cameras = [c for c in cameras if c < num_cameras]
    if not cameras:
        raise SystemExit("No valid camera indices")

    print(f"Dataset: {args.dataset_root}")
    camera_report(args.dataset_root, cameras)

    if args.per_camera:
        print(f"\nPer-camera mode, cameras={cameras}, min_drawn={args.min_drawn}")
        print(f"Output: {out_dir.resolve()}")
        saved = run_per_camera(
            args.dataset_root,
            cameras,
            out_dir,
            args.num_frames,
            args.min_drawn,
            args.mosaic_width,
        )
        print(f"\nDone. {saved} images in {out_dir.resolve()}")
        return

    dataset = GtaMcmtDataset(args.dataset_root, num_cameras=num_cameras)
    sync_len = synced_length_for_cameras(args.dataset_root, cameras)
    max_k = sync_len - 1
    sync_indices = (
        args.sync_indices
        if args.sync_indices is not None
        else evenly_spaced_indices(max_k, args.num_frames)
    )

    print(f"Cameras: {cameras}  synced length: {sync_len}  sync_k: {sync_indices}")
    print(f"Output: {out_dir.resolve()}")

    if sync_len == 0:
        print(
            "\nNo common sync across selected cameras (some CSV empty or missing)."
            "\nFalling back to --per-camera mode."
        )
        saved = run_per_camera(
            args.dataset_root,
            cameras,
            out_dir,
            args.num_frames,
            args.min_drawn,
            args.mosaic_width,
        )
        print(f"\nDone. {saved} images in {out_dir.resolve()}")
        return

    print("NOTE: synced k pairs k-th CSV frame per camera — sparse cams desync moments.")

    saved = 0
    for k in sync_indices:
        cam_frames: list[np.ndarray] = []
        for cam in cameras:
            snaps = dataset.snapshots_by_cam[cam]
            if k >= len(snaps):
                print(f"[skip] cam-{cam} k={k}: no CSV frame (has {len(snaps)})")
                continue
            snap = snaps[k]
            image_path = image_path_for_cam_dir(dataset.cam_dirs[cam], snap.cam_id)
            if not image_path.is_file():
                print(f"[skip] missing cam-{cam} k={k}: {image_path}")
                continue
            frame, drawn = draw_frame(image_path, snap, k, cam)
            if drawn < args.min_drawn:
                print(f"[skip] cam-{cam} k={k} drawn={drawn}")
                continue
            cam_frames.append(frame)
            out_path = out_dir / f"sync{k:05d}_cam{cam}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

        if len(cam_frames) == len(cameras):
            mosaic = stack_camera_grid(cam_frames, target_width=args.mosaic_width)
            cv2.imwrite(str(out_dir / f"sync{k:05d}_all_cams.jpg"), mosaic)
            saved += 1

        counts = []
        for c in cameras:
            snaps = dataset.snapshots_by_cam[c]
            n = len(snaps[k].annotations) if k < len(snaps) else 0
            counts.append(f"c{c}:{n}")
        print(f"  k={k:5d}  csv_anns=[{', '.join(counts)}]")

    print(f"Done. {saved} images in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
