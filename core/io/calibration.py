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
