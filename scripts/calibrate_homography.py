#!/usr/bin/env python3
"""Pick image <-> world point pairs for homography calibration.

Usage:
    python scripts/calibrate_homography.py frame.jpg
    python scripts/calibrate_homography.py datasets/gta_mcmt/cam-0/image_88709.jpg --out datasets/gta_mcmt/cam-0/calibration.txt

Controls (image window):
    Left click      - add point
    Right click     - remove nearest point
    u / Backspace   - undo last point
    r               - clear all points
    Enter / d       - finish picking (use d if Enter closes too fast)

After finishing, numbered points are shown and saved to ``*_calib_points.jpg``.
Console prompts: p1: wx wy  (world / map coordinates for each point).
If --out is set, writes ``calibration.txt`` (H_world_to_image) + ``calibration_i2w.txt``
+ ``calibration_points.json`` (image/world pairs).
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
    homography_image_to_world,
    save_calibration_points,
    save_homography,
    CalibPointPair,
)

WIN = "calibrate_homography"
POINT_RADIUS = 6
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_points(img: np.ndarray, points: list[tuple[int, int]], *, numbered: bool) -> np.ndarray:
    out = img.copy()
    for i, (x, y) in enumerate(points, start=1):
        cv2.circle(out, (x, y), POINT_RADIUS, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), POINT_RADIUS + 2, (0, 0, 0), 2, cv2.LINE_AA)
        if numbered:
            label = str(i)
            cv2.putText(
                out,
                label,
                (x + 10, y - 10),
                FONT,
                0.8,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                label,
                (x + 10, y - 10),
                FONT,
                0.8,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
    return out


def _nearest_index(points: list[tuple[int, int]], x: int, y: int, max_dist: float = 20.0) -> int | None:
    if not points:
        return None
    dists = [np.hypot(px - x, py - y) for px, py in points]
    idx = int(np.argmin(dists))
    return idx if dists[idx] <= max_dist else None


def _drain_keys(ms: int = 300) -> None:
    """Drop queued key events (Enter often fires twice across windows)."""
    t0 = cv2.getTickCount()
    while (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000 < ms:
        cv2.waitKey(30)


def pick_image_points(image_path: Path) -> tuple[np.ndarray, list[tuple[int, int]]]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    points: list[tuple[int, int]] = []
    state = {"done": False}

    def on_mouse(event, x, y, _flags, _userdata) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN:
            idx = _nearest_index(points, x, y)
            if idx is not None:
                points.pop(idx)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)

    print("Controls: LMB=add  RMB=remove nearest  u=undo  r=reset  d=done (Enter also works)")
    while not state["done"]:
        vis = _draw_points(img, points, numbered=False)
        hint = f"points: {len(points)}  |  press d when ready"
        cv2.putText(vis, hint, (12, 28), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, hint, (12, 28), FONT, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(WIN, vis)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, ord("d")):  # Enter / d
            if len(points) < 4:
                print(f"Warning: only {len(points)} points (>=4 recommended for homography).")
            state["done"] = True
        elif key in (8, ord("u")):
            if points:
                points.pop()
        elif key == ord("r"):
            points.clear()
        elif key == ord("q") and len(points) == 0:
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled.")

    cv2.destroyWindow(WIN)
    _drain_keys()
    return img, points


def read_world_points(
    n: int,
    preview: np.ndarray | None = None,
    win_name: str = "numbered_points",
) -> list[tuple[float, float]]:
    world: list[tuple[float, float]] = []
    print("\nEnter world / map coordinates in the CONSOLE (image stays open).")
    print("Close the image window when finished, or press q there after last point.")
    for i in range(1, n + 1):
        while True:
            if preview is not None:
                cv2.imshow(win_name, preview)
                key = cv2.waitKey(50) & 0xFF
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    raise SystemExit("Cancelled.")
            raw = input(f"p{i}: ").strip()
            parts = raw.replace(",", " ").split()
            if len(parts) != 2:
                print("  need two numbers, e.g. 1920 1080")
                continue
            try:
                wx, wy = float(parts[0]), float(parts[1])
            except ValueError:
                print("  invalid number")
                continue
            world.append((wx, wy))
            break
    return world


def compute_homographies(
    image_pts: list[tuple[int, int]],
    world_pts: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, float]:
    src = np.asarray(image_pts, dtype=np.float64)
    dst = np.asarray(world_pts, dtype=np.float64)
    H_i2w, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H_i2w is None:
        raise RuntimeError("cv2.findHomography failed - check point pairs.")
    H_w2i = np.linalg.inv(H_i2w)
    reproj = _reprojection_error(src, dst, H_i2w)
    inliers = int(mask.sum()) if mask is not None else len(image_pts)
    print(f"Homography OK ({inliers}/{len(image_pts)} inliers), mean reproj error: {reproj:.3f}")
    return H_w2i, H_i2w, reproj


def _reprojection_error(
    image_pts: np.ndarray,
    world_pts: np.ndarray,
    H_i2w: np.ndarray,
) -> float:
    ones = np.ones((len(image_pts), 1), dtype=np.float64)
    hom = np.hstack([image_pts, ones])
    proj = (H_i2w @ hom.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(proj - world_pts, axis=1)
    return float(err.mean())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path, help="Camera frame or snapshot")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write calibration.txt (H_world_to_image) here",
    )
    ap.add_argument(
        "--preview",
        type=Path,
        default=None,
        help="Save numbered preview (default: <image>_calib_points.jpg)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image.resolve()
    img, image_pts = pick_image_points(image_path)

    if not image_pts:
        raise SystemExit("No points selected.")

    numbered = _draw_points(img, image_pts, numbered=True)
    hint = "Enter world coords in CONSOLE  |  q=abort"
    cv2.putText(numbered, hint, (12, 28), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(numbered, hint, (12, 28), FONT, 0.65, (0, 255, 0), 1, cv2.LINE_AA)

    preview_path = args.preview or image_path.with_name(f"{image_path.stem}_calib_points.jpg")
    cv2.imwrite(str(preview_path), numbered)
    print(f"\nSaved numbered preview: {preview_path}")
    print("Numbered image window stays open while you type p1, p2, ... in the console.")

    cv2.namedWindow("numbered_points", cv2.WINDOW_NORMAL)
    cv2.imshow("numbered_points", numbered)
    _drain_keys()

    world_pts = read_world_points(len(image_pts), preview=numbered)
    cv2.destroyAllWindows()

    print("\n--- Summary ---")
    for i, ((ix, iy), (wx, wy)) in enumerate(zip(image_pts, world_pts), start=1):
        print(f"p{i}: image=({ix}, {iy})  world=({wx}, {wy})")

    if len(image_pts) >= 4:
        H_w2i, H_i2w, reproj = compute_homographies(image_pts, world_pts)
        if args.out:
            out = args.out.resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            save_homography(out, H_w2i)
            save_homography(out.parent / "calibration_i2w.txt", H_i2w)
            pairs = [
                CalibPointPair(
                    image_x=float(ix),
                    image_y=float(iy),
                    world_x=float(wx),
                    world_y=float(wy),
                )
                for (ix, iy), (wx, wy) in zip(image_pts, world_pts)
            ]
            pts_path = save_calibration_points(
                out,
                image_path=image_path,
                pairs=pairs,
                reprojection_error=reproj,
            )
            text = out.read_text(encoding="utf-8").rstrip()
            out.write_text(f"{text}\nReprojection error: {reproj:.6f}\n", encoding="utf-8")
            print(f"Saved: {out}")
            print(f"Saved: {out.parent / 'calibration_i2w.txt'}")
            print(f"Saved: {pts_path}")
    else:
        print("\nNeed >=4 points to compute homography. Pairs saved in preview only.")


if __name__ == "__main__":
    main()
