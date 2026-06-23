"""Run YOLO on synced GTA MCMT frames and save detection previews.

Optionally overlay GT boxes (dashed/green) for comparison with detections (solid).

Usage:
    python scripts/visualize_gta_mcmt_detections.py
    python scripts/visualize_gta_mcmt_detections.py --sync-indices 0 500 1000 --with-gt
    python scripts/visualize_gta_mcmt_detections.py --config config/gta_mcmt_baseline.yaml
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

from core.detector.detector import Detector
from core.io.gta_mcmt import (
    GtaMcmtDataset,
    NUM_CAMERAS,
    center_bbox_to_xyxy,
    image_path_for_cam_dir,
)
from core.visualization.visualizer import Visualizer
from scripts.visualize_gta_mcmt import default_sync_indices, stack_camera_views

COCO_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def load_detector_cfg(config_path: Path | None) -> dict:
    if config_path is None:
        return {
            "model": "models/yolov8m.pt",
            "target_classes": [2, 7],
            "conf_thres": 0.2,
            "imgsz": 640,
            "device": 0,
        }
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg.get("detector", {})


def draw_detections(
    frame: np.ndarray,
    dets: list,
    cam: int,
    sync_k: int,
    image_name: str,
) -> np.ndarray:
    img = frame.copy()
    for i, det in enumerate(dets):
        x1, y1, x2, y2, conf, cls = det[:6]
        cls = int(cls)
        np.random.seed(i + sync_k * 100 + cam)
        color = (
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255)),
        )
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{COCO_NAMES.get(cls, f'c{cls}')} {conf:.2f}"
        cv2.putText(
            img,
            label,
            (int(x1), max(0, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    header = f"sync_k={sync_k}  cam-{cam}  {image_name}  dets={len(dets)}"
    cv2.putText(
        img,
        header,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def draw_gt_overlay(
    img: np.ndarray,
    snapshot,
    *,
    gt_scale: float | None = None,
) -> np.ndarray:
    out = img.copy()
    for ann in snapshot.annotations:
        if gt_scale is not None and gt_scale != 1.0:
            x1, y1, x2, y2 = center_bbox_to_xyxy(ann.cx, ann.cy, ann.w, ann.h)
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            x1, y1, x2, y2 = center_bbox_to_xyxy(
                ann.cx, ann.cy, ann.w, ann.h, scale=gt_scale
            )
            color = (0, 165, 255)  # orange: shrunk GT
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                out,
                f"GT{gt_scale:.2f}:{ann.obj_id}",
                (int(x1), max(0, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        else:
            x1, y1, x2, y2 = center_bbox_to_xyxy(ann.cx, ann.cy, ann.w, ann.h)
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                out,
                f"GT:{ann.obj_id}",
                (int(x1), max(0, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=Path("datasets/gta_mcmt"))
    ap.add_argument("--config", type=Path, default=Path("config/gta_mcmt_baseline.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gta_mcmt_det_preview"))
    ap.add_argument(
        "--sync-indices",
        type=int,
        nargs="+",
        default=None,
        help="Sync indices k (default: 5 evenly spaced)",
    )
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument(
        "--with-gt",
        action="store_true",
        help="Overlay GT boxes under detection boxes",
    )
    ap.add_argument(
        "--gt-scale",
        type=float,
        default=None,
        help="If set (e.g. 0.85), draw full GT green (thin) + scaled GT orange (thick)",
    )
    ap.add_argument("--model", default=None, help="Override detector model path")
    ap.add_argument("--conf", type=float, default=None, help="Override conf threshold")
    ap.add_argument("--imgsz", type=int, default=None, help="Override inference size")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    det_cfg = load_detector_cfg(args.config if args.config.is_file() else None)

    model = args.model or det_cfg.get("model", "models/yolov8m.pt")
    target_classes = det_cfg.get("target_classes", [2, 7])
    conf_thres = args.conf if args.conf is not None else float(det_cfg.get("conf_thres", 0.2))
    imgsz = args.imgsz if args.imgsz is not None else int(det_cfg.get("imgsz", 640))
    device = det_cfg.get("device", 0)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = GtaMcmtDataset(args.dataset_root)
    sync_indices = (
        args.sync_indices
        if args.sync_indices
        else default_sync_indices(len(dataset) - 1, args.count)
    )

    print(f"Model: {model}  classes={target_classes}  conf={conf_thres}  imgsz={imgsz}")
    print(f"Sync indices: {sync_indices}")
    print(f"Output: {out_dir.resolve()}")

    detector = Detector(
        model=model,
        target_classes=target_classes,
        conf_thres=conf_thres,
        imgsz=imgsz,
        device=device,
    )

    for k in sync_indices:
        frames = dataset.read_sync(k)
        dets_batch = detector.detect_batch(frames)
        cam_vis: list[np.ndarray] = []

        for cam in range(NUM_CAMERAS):
            snap = dataset.snapshot(cam, k)
            img_path = image_path_for_cam_dir(dataset.cam_dirs[cam], snap.cam_id)
            dets, _ = dets_batch[cam]
            vis = draw_detections(frames[cam], dets, cam, k, img_path.name)
            if args.with_gt or args.gt_scale is not None:
                vis = draw_gt_overlay(vis, snap, gt_scale=args.gt_scale)
            cam_vis.append(vis)
            cv2.imwrite(str(out_dir / f"k{k:05d}_cam{cam}_det.jpg"), vis)

        mosaic = stack_camera_views(cam_vis)
        cv2.imwrite(str(out_dir / f"k{k:05d}_all_cams_det.jpg"), mosaic)

        gt_counts = [len(dataset.snapshot(c, k).annotations) for c in range(NUM_CAMERAS)]
        det_counts = [len(dets_batch[c][0]) for c in range(NUM_CAMERAS)]
        print(f"k={k:5d}  dets/cam={det_counts}  gt/cam={gt_counts}")

    print("Done.")


if __name__ == "__main__":
    main()
