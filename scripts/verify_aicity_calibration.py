"""Sanity-check AICity S02 homographies project to plausible lat/lon (GPS meters).

Usage:
    python scripts/verify_aicity_calibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.io.calibration import (  # noqa: E402
    load_homography,
    load_homography_image_to_world,
    project_bbox_bottom_center,
    world_gps_distance_m,
)
from scripts.cityflow_ablation_common import S02_CAM_IDS, S02_ROOT_REL  # noqa: E402

# Approximate scene center from AICity ReadMe (S02.png).
EXPECTED_LAT = 42.491916
EXPECTED_LON = -90.723723
LAT_TOL = 0.05
LON_TOL = 0.05

SAMPLE_POINTS = [
    (640, 540),
    (960, 720),
    (320, 400),
]


def main() -> None:
    root = _ROOT / S02_ROOT_REL
    all_ok = True
    projected: list[tuple[int, float, float]] = []

    print(f"Checking homographies under {S02_ROOT_REL}/")
    for cam in S02_CAM_IDS:
        cal_path = root / f"c{cam:03d}" / "calibration.txt"
        if not cal_path.is_file():
            print(f"[FAIL] missing {cal_path}")
            all_ok = False
            continue

        H_w2i = load_homography(cal_path)
        H_i2w = load_homography_image_to_world(cal_path)
        print(f"\nc{cam:03d}: calibration.txt loaded (world_to_image)")
        print(f"  H_w2i[2,:] = {H_w2i[2].tolist()}")

        for x, y in SAMPLE_POINTS:
            lat, lon = project_bbox_bottom_center(H_i2w, x - 50, y - 50, x + 50, y)
            projected.append((cam, lat, lon))
            lat_ok = abs(lat - EXPECTED_LAT) <= LAT_TOL
            lon_ok = abs(lon - EXPECTED_LON) <= LON_TOL
            status = "OK" if (lat_ok and lon_ok) else "WARN"
            print(f"  [{status}] bottom_center@({x},{y}) -> lat={lat:.6f}, lon={lon:.6f}")
            if status != "OK":
                all_ok = False

    if len(projected) >= 2:
        (c1, lat1, lon1), (c2, lat2, lon2) = projected[0], projected[1]
        d_m = world_gps_distance_m((lat1, lon1), (lat2, lon2))
        print(f"\nSample GPS distance c{c1} vs c{c2}: {d_m:.2f} m")

    if all_ok:
        print("\nCalibration check passed (lat/lon within tolerance).")
    else:
        print("\nCalibration check completed with warnings (review projections above).")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
