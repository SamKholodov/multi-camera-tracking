from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.geometry.contact_point.prepare import DEFAULT_TARGET_CLASSES, prepare_contact_point_dataset


def _load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare contact point regression dataset.")
    parser.add_argument("--config", default=None, help="YAML config path.")
    parser.add_argument("--gta-root", default=None, help="GTA dataset root with cam-*/coords_cam_*.csv.")
    parser.add_argument("--output-dir", default=None, help="Prepared dataset output directory.")
    parser.add_argument("--detector", default=None, help="YOLO detector weights path.")
    parser.add_argument(
        "--target-classes",
        nargs="+",
        type=int,
        default=None,
        help="COCO class ids for detector filtering.",
    )
    parser.add_argument("--device", default=None, help="Detector/training device override.")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf-thres", type=float, default=None)
    parser.add_argument("--match-iou", type=float, default=None)
    parser.add_argument("--viz-count", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None, help="Smoke test: limit unique images.")
    parser.add_argument(
        "--skip-detector",
        action="store_true",
        help="Use annotation bbox instead of YOLO matching (debug only).",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config) if args.config else {}
    if args.gta_root is not None:
        cfg["gta_root"] = args.gta_root
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    det_cfg = dict(cfg.get("detector", {}))
    if args.detector is not None:
        det_cfg["model"] = args.detector
    if args.target_classes is not None:
        det_cfg["target_classes"] = args.target_classes
    elif "target_classes" not in det_cfg:
        det_cfg["target_classes"] = list(DEFAULT_TARGET_CLASSES)
    if args.device is not None:
        det_cfg["device"] = args.device
    if args.imgsz is not None:
        det_cfg["imgsz"] = args.imgsz
    if args.conf_thres is not None:
        det_cfg["conf_thres"] = args.conf_thres
    cfg["detector"] = det_cfg
    if args.match_iou is not None:
        cfg["match_iou_threshold"] = args.match_iou
    if args.viz_count is not None:
        cfg["viz_count"] = args.viz_count
    if args.val_fraction is not None:
        cfg["val_fraction"] = args.val_fraction
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.max_images is not None:
        cfg["max_images"] = args.max_images
    if args.skip_detector:
        cfg["skip_detector"] = True

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    result = prepare_contact_point_dataset(cfg)
    print(json_dumps(result))


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
