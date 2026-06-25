"""Fine-tune YOLO26l on CityFlow YOLO-exported dataset (single-class vehicle).

Prerequisites:
    python scripts/finetune/export_cityflow_yolo_dataset.py --train-scenes S01 S03 S04 --frame-stride 2 --clean

Usage:
    python scripts/finetune/train_cityflow_yolo26l.py
    python scripts/finetune/train_cityflow_yolo26l.py --epochs 25 --batch 8 --imgsz 768

Note:
    Run only one training process at a time (check GPU with nvidia-smi). Do not start
    a second copy with different --batch/--imgsz while one is already running.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_DATA = _ROOT / "datasets/cityflow_yolo_finetune_train/data.yaml"
DEFAULT_BASE = _ROOT / "models/yolo26l.pt"
DEFAULT_RUN_NAME = "cityflow_yolo26l_finetune_train"
DEFAULT_EXPORT_NAME = "cityflow_yolo26l_finetune_train.pt"


def verify_yolo_dataset(data_yaml: Path) -> None:
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    removed = 0
    for split in ("train", "val"):
        img_dir = root / cfg[split]
        lbl_dir = root / split / "labels"
        if not lbl_dir.is_dir():
            continue
        for lbl in lbl_dir.glob("*.txt"):
            if not (img_dir / f"{lbl.stem}.jpg").is_file():
                lbl.unlink(missing_ok=True)
                removed += 1
    if removed:
        print(f"[WARN] removed {removed} orphan labels without matching images")


def train(
    data_yaml: Path,
    base_weights: Path,
    *,
    run_name: str = DEFAULT_RUN_NAME,
    export_name: str = DEFAULT_EXPORT_NAME,
    epochs: int = 30,
    imgsz: int = 960,
    batch: int = 6,
    patience: int = 15,
    device: str | int = 0,
) -> Path:
    from ultralytics import YOLO

    from scripts.ensure_yolo_weights import ensure_yolo_weights

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Missing {data_yaml}. Run export_cityflow_yolo_dataset.py first."
        )
    verify_yolo_dataset(data_yaml)
    ensure_yolo_weights(base_weights)

    run_dir = _ROOT / "runs/detect" / run_name
    export_path = _ROOT / "models" / export_name

    model = YOLO(str(base_weights))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=str(run_dir.parent),
        name=run_name,
        exist_ok=True,
        device=device,
        workers=0,
        seed=42,
        cache="disk",
        lr0=5e-4,
        freeze=5,
        close_mosaic=10,
    )

    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training finished but best.pt missing: {best}")

    export_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, export_path)
    print(f"Copied best weights -> {export_path}")
    return best.resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    ap.add_argument(
        "--export",
        default=DEFAULT_EXPORT_NAME,
        help="Filename under models/ for best.pt copy",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--device", default=0)
    args = ap.parse_args()

    best = train(
        args.data,
        args.base,
        run_name=args.run_name,
        export_name=args.export,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
    )
    print(f"Best weights: {best}")


if __name__ == "__main__":
    main()
