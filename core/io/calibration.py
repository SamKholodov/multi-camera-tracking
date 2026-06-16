"""AICity / CityFlow calibration.txt parser.

File layout per camera::

    Homography matrix: a b c;d e f;g h i
    Reprojection error: <float>

``calibration.txt`` stores **H_world_to_image** (world/ground → pixels).
For image → world use **H_image_to_world = inv(H_world_to_image)**.

The inverse may be cached as ``calibration_i2w.txt`` (same format) next to
``calibration.txt`` so it is computed once, not on every projection.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import numpy as np

_CALIB_LINE_PREFIX = "Homography matrix:"


def _parse_homography_text(text: str, path: Union[str, Path]) -> np.ndarray:
    line = next(
        (ln for ln in text.strip().splitlines() if ln.lower().startswith("homography")),
        None,
    )
    if line is None:
        raise ValueError(f"No 'Homography matrix:' line found in {path}")

    payload = line.split(":", 1)[1].strip()
    rows = [r.strip() for r in payload.split(";") if r.strip()]
    if len(rows) != 3:
        raise ValueError(f"Expected 3 rows in homography, got {len(rows)} in {path}")

    matrix = np.array(
        [[float(v) for v in r.split()] for r in rows],
        dtype=np.float64,
    )
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got {matrix.shape} in {path}")
    return matrix


def _format_homography(H: np.ndarray) -> str:
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    rows = [" ".join(f"{v:.15f}" for v in H[i]) for i in range(3)]
    body = ";".join(rows)
    return f"{_CALIB_LINE_PREFIX} {body}\n"


def image_to_world_cache_path(calibration_path: Union[str, Path]) -> Path:
    """Path for cached ``H_image_to_world`` beside ``calibration.txt``."""
    p = Path(calibration_path)
    return p.parent / "calibration_i2w.txt"


def load_homography(path: Union[str, Path]) -> np.ndarray:
    """Load **H_world_to_image** from ``calibration.txt`` (unchanged from dataset)."""
    path = Path(path)
    return _parse_homography_text(path.read_text(encoding="utf-8"), path)


def save_homography(path: Union[str, Path], homography: np.ndarray) -> None:
    """Write a 3x3 matrix in AICity ``Homography matrix:`` format."""
    Path(path).write_text(_format_homography(homography), encoding="utf-8")


def homography_image_to_world(H_world_to_image: np.ndarray) -> np.ndarray:
    """``H_i2w = inv(H_w2i)``."""
    return np.linalg.inv(np.asarray(H_world_to_image, dtype=np.float64))


def load_homography_image_to_world(
    calibration_path: Union[str, Path],
    *,
    write_cache: bool = True,
) -> np.ndarray:
    """Load **H_image_to_world**; use ``calibration_i2w.txt`` if present."""
    calibration_path = Path(calibration_path)
    cache_path = image_to_world_cache_path(calibration_path)

    if cache_path.is_file():
        return load_homography(cache_path)

    H_i2w = homography_image_to_world(load_homography(calibration_path))
    if write_cache:
        save_homography(cache_path, H_i2w)
    return H_i2w


def project_point(
    H_image_to_world: np.ndarray,
    x: float,
    y: float,
    *,
    flip_y: bool = False,
) -> tuple[float, float]:
    """Project image (x, y) to world with **H_image_to_world** (no inverse here)."""
    vec = np.asarray(H_image_to_world, dtype=np.float64) @ np.array([x, y, 1.0], dtype=np.float64)
    w = vec[2] if abs(vec[2]) > 1e-12 else 1e-12
    xw, yw = float(vec[0] / w), float(vec[1] / w)
    if flip_y:
        yw = -yw
    return xw, yw


def project_bbox_bottom_center(
    H_image_to_world: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    flip_y: bool = False,
) -> tuple[float, float]:
    """Project the bottom-center of an image bbox to world coordinates."""
    bcx = (float(x1) + float(x2)) / 2.0
    bcy = float(y2)
    return project_point(H_image_to_world, bcx, bcy, flip_y=flip_y)


def project_world_to_image(
    H_world_to_image: np.ndarray,
    xw: float,
    yw: float,
    *,
    flip_y: bool = False,
) -> tuple[float, float]:
    """Project world (xw, yw) to image with **H_world_to_image**."""
    if flip_y:
        yw = -yw
    vec = np.asarray(H_world_to_image, dtype=np.float64) @ np.array([xw, yw, 1.0], dtype=np.float64)
    w = vec[2] if abs(vec[2]) > 1e-12 else 1e-12
    return float(vec[0] / w), float(vec[1] / w)


def haversine_distance_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    R = 6_371_000.0
    lat1_r, lon1_r = map(math.radians, (lat1, lon1))
    lat2_r, lon2_r = map(math.radians, (lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def world_gps_distance_m(
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    """Distance between two ``(xworld, yworld)`` = ``(lat, lon)`` points."""
    return haversine_distance_m(a[0], a[1], b[0], b[1])


def world_plane_distance(
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    """Euclidean distance in an arbitrary ground-plane coordinate system."""
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def world_distance(
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    metric: str = "gps",
) -> float:
    if metric == "plane":
        return world_plane_distance(a, b)
    if metric == "gps":
        return world_gps_distance_m(a, b)
    raise ValueError(f"Unknown world distance metric: {metric!r}")


@dataclass(frozen=True)
class CalibPointPair:
    image_x: float
    image_y: float
    world_x: float
    world_y: float


def calibration_points_path(calibration_path: Union[str, Path]) -> Path:
    return Path(calibration_path).parent / "calibration_points.json"


def save_calibration_points(
    calibration_path: Union[str, Path],
    *,
    image_path: Union[str, Path],
    pairs: list[CalibPointPair],
    reprojection_error: float | None = None,
) -> Path:
    out = calibration_points_path(calibration_path)
    payload = {
        "image": str(Path(image_path).name),
        "reprojection_error": reprojection_error,
        "pairs": [asdict(p) for p in pairs],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_calibration_points(
    calibration_path: Union[str, Path],
) -> tuple[str | None, list[CalibPointPair], float | None]:
    path = calibration_points_path(calibration_path)
    if not path.is_file():
        return None, [], None
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = [
        CalibPointPair(
            image_x=float(p["image_x"]),
            image_y=float(p["image_y"]),
            world_x=float(p["world_x"]),
            world_y=float(p["world_y"]),
        )
        for p in payload.get("pairs", [])
    ]
    return payload.get("image"), pairs, payload.get("reprojection_error")
