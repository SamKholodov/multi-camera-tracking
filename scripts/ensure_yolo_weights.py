"""Ensure YOLO weights exist under models/ (download via Ultralytics if needed)."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def ensure_yolo_weights(model_path: str | Path) -> Path:
    dest = Path(model_path)
    if not dest.is_absolute():
        dest = _ROOT / dest
    if dest.is_file():
        print(f"OK: {dest.relative_to(_ROOT).as_posix()}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    weight_name = dest.name
    print(f"Downloading {weight_name} via Ultralytics...")

    from ultralytics import YOLO

    model = YOLO(weight_name)
    src = Path(getattr(model, "ckpt_path", weight_name))
    if not src.is_file():
        src = _ROOT / weight_name
    if not src.is_file():
        raise SystemExit(f"Failed to download {weight_name}")

    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
        print(f"Copied to {dest.relative_to(_ROOT).as_posix()}")
    else:
        print(f"Ready: {dest.relative_to(_ROOT).as_posix()}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "models",
        nargs="*",
        default=["models/yolo26l.pt"],
        help="Model paths relative to repo root (default: models/yolo26l.pt)",
    )
    args = ap.parse_args()
    for model_path in args.models:
        ensure_yolo_weights(model_path)


if __name__ == "__main__":
    main()
