"""Build GTA MCMT demonstration frames: one vehicle on 3+ cameras, YOLO-refined bboxes.

Usage:
    python scripts/build_gta_demonstration.py --num-examples 6 --device 0
    python scripts/build_gta_demonstration.py --force
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.detector.detector import Detector
from core.geometry.contact_point.dataset import expand_bbox
from core.geometry.contact_point.matching import best_detection_match
from core.io.gta_mcmt import (
    GtaAnnotation,
    GtaMcmtDataset,
    NUM_CAMERAS,
    center_bbox_to_xyxy,
    image_path_for_cam_dir,
)
from core.visualization.visualizer import Visualizer
from scripts.visualize_gta_mcmt import clip_box, stack_camera_views

VEHICLE_CLASSES = {2, 3, 5, 7}
DEFAULT_MODEL = "models/yolo26x.pt"


@dataclass
class CamResult:
    visible: bool = False
    cam: int = -1
    image_path: str = ""
    frame_idx: str = ""
    cam_id: str = ""
    gt_bbox_xyxy: list[float] = field(default_factory=list)
    bbox_xyxy: list[float] = field(default_factory=list)
    bottom_center_px: list[float] = field(default_factory=list)
    roi_xyxy: list[float] = field(default_factory=list)
    match_iou: float = 0.0
    det_score: float = 0.0
    det_class_id: int = -1
    reason: str = ""


@dataclass
class Candidate:
    sync_index: int
    vehicle_id: int
    cam_results: dict[int, CamResult]
    score: float = 0.0

    @property
    def visible_cams(self) -> list[int]:
        return sorted(c for c, r in self.cam_results.items() if r.visible)

    @property
    def yolo_ok_cams(self) -> int:
        return sum(1 for r in self.cam_results.values() if r.visible and r.bbox_xyxy)

    @property
    def mean_match_iou(self) -> float:
        ious = [r.match_iou for r in self.cam_results.values() if r.visible and r.bbox_xyxy]
        return float(np.mean(ious)) if ious else 0.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=Path("datasets/gta_mcmt"))
    ap.add_argument("--out-dir", type=Path, default=Path("gta_demonstration"))
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--num-examples", type=int, default=6)
    ap.add_argument("--min-cameras", type=int, default=3)
    ap.add_argument("--min-bbox-size", type=int, default=40)
    ap.add_argument("--roi-pad", type=float, default=2.5)
    ap.add_argument("--scan-stride", type=int, default=25)
    ap.add_argument("--min-sync-gap", type=int, default=200)
    ap.add_argument("--min-match-iou", type=float, default=0.15)
    ap.add_argument("--conf-thres", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="cuda device index or 'cpu' (default: auto)")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def ann_gt_xyxy(ann: GtaAnnotation) -> tuple[float, float, float, float]:
    return center_bbox_to_xyxy(ann.cx, ann.cy, ann.w, ann.h)


def is_drawable_gt(
    ann: GtaAnnotation, img_w: int, img_h: int, min_size: int
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = ann_gt_xyxy(ann)
    if min(x2 - x1, y2 - y1) < min_size:
        return None
    clipped = clip_box(x1, y1, x2, y2, img_w, img_h)
    if clipped is None:
        return None
    return tuple(float(v) for v in ann_gt_xyxy(ann))


def scan_csv_candidates(
    dataset: GtaMcmtDataset,
    *,
    scan_stride: int,
    min_cameras: int,
    min_bbox_size: int,
) -> list[tuple[int, int, dict[int, GtaAnnotation]]]:
    """Return (sync_index, obj_id, {cam: annotation}) for CSV-only pre-filter."""
    out: list[tuple[int, int, dict[int, GtaAnnotation]]] = []
    for k in range(0, len(dataset), scan_stride):
        by_obj: dict[int, dict[int, GtaAnnotation]] = {}
        for cam in range(dataset.num_cameras):
            snap = dataset.snapshot(cam, k)
            for ann in snap.annotations:
                if ann.obj_class not in VEHICLE_CLASSES:
                    continue
                if min(ann.w, ann.h) < min_bbox_size:
                    continue
                by_obj.setdefault(ann.obj_id, {})[cam] = ann

        for obj_id, cam_anns in by_obj.items():
            if len(cam_anns) >= min_cameras:
                out.append((k, obj_id, cam_anns))

    # Keep one best obj_id per sync_index (most cameras, then largest mean bbox area).
    best_by_k: dict[int, tuple[int, int, dict[int, GtaAnnotation]]] = {}
    for k, obj_id, cam_anns in out:
        mean_area = float(np.mean([a.w * a.h for a in cam_anns.values()]))
        key = (len(cam_anns), mean_area)
        prev = best_by_k.get(k)
        if prev is None or key > (len(prev[2]), float(np.mean([a.w * a.h for a in prev[2].values()]))):
            best_by_k[k] = (k, obj_id, cam_anns)
    return list(best_by_k.values())


def offset_detections(
    detections: list, roi_x1: float, roi_y1: float
) -> list[list[float]]:
    shifted: list[list[float]] = []
    for det in detections:
        if len(det) < 6:
            continue
        shifted.append(
            [
                float(det[0]) + roi_x1,
                float(det[1]) + roi_y1,
                float(det[2]) + roi_x1,
                float(det[3]) + roi_y1,
                float(det[4]),
                int(det[5]),
            ]
        )
    return shifted


def refine_bbox_yolo(
    frame: np.ndarray,
    gt_xyxy: tuple[float, float, float, float],
    detector: Detector,
    roi_pad: float,
    min_match_iou: float,
) -> CamResult | None:
    img_h, img_w = frame.shape[:2]
    roi = expand_bbox(
        gt_xyxy,
        image_width=img_w,
        image_height=img_h,
        pad_ratio=roi_pad,
    )
    rx1, ry1, rx2, ry2 = (int(round(v)) for v in roi)
    rx1 = max(0, min(rx1, img_w - 1))
    ry1 = max(0, min(ry1, img_h - 1))
    rx2 = max(rx1 + 1, min(rx2, img_w))
    ry2 = max(ry1 + 1, min(ry2, img_h))

    crop = frame[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return None

    detections, _ = detector.detect(crop)
    shifted = offset_detections(detections, rx1, ry1)
    _, match = best_detection_match(gt_xyxy, shifted)
    if match.bbox is None or match.iou < min_match_iou:
        return None

    x1, y1, x2, y2 = match.bbox
    return CamResult(
        visible=True,
        gt_bbox_xyxy=list(gt_xyxy),
        bbox_xyxy=[x1, y1, x2, y2],
        bottom_center_px=[(x1 + x2) / 2.0, y2],
        roi_xyxy=[float(rx1), float(ry1), float(rx2), float(ry2)],
        match_iou=float(match.iou),
        det_score=float(match.score),
        det_class_id=int(match.class_id),
    )


def evaluate_candidate(
    dataset: GtaMcmtDataset,
    sync_index: int,
    vehicle_id: int,
    cam_anns: dict[int, GtaAnnotation],
    detector: Detector,
    *,
    roi_pad: float,
    min_match_iou: float,
) -> Candidate | None:
    cam_results: dict[int, CamResult] = {}
    for cam in range(dataset.num_cameras):
        if cam not in cam_anns:
            cam_results[cam] = CamResult(visible=False, cam=cam)
            continue

        snap = dataset.snapshot(cam, sync_index)
        image_path = image_path_for_cam_dir(dataset.cam_dirs[cam], snap.cam_id)
        frame = cv2.imread(str(image_path))
        if frame is None:
            cam_results[cam] = CamResult(visible=False, cam=cam, reason="missing_image")
            return None

        ann = cam_anns[cam]
        gt_xyxy = is_drawable_gt(ann, frame.shape[1], frame.shape[0], min_size=1)
        if gt_xyxy is None:
            cam_results[cam] = CamResult(visible=False, cam=cam, reason="bbox_not_drawable")
            return None

        refined = refine_bbox_yolo(frame, gt_xyxy, detector, roi_pad, min_match_iou)
        if refined is None:
            cam_results[cam] = CamResult(
                visible=True,
                cam=cam,
                image_path=str(image_path),
                frame_idx=snap.frame_idx,
                cam_id=snap.cam_id,
                gt_bbox_xyxy=list(gt_xyxy),
                reason="yolo_no_match",
            )
            return None

        refined.cam = cam
        refined.image_path = str(image_path)
        refined.frame_idx = snap.frame_idx
        refined.cam_id = snap.cam_id
        cam_results[cam] = refined

    areas = [
        (r.bbox_xyxy[2] - r.bbox_xyxy[0]) * (r.bbox_xyxy[3] - r.bbox_xyxy[1])
        for r in cam_results.values()
        if r.visible and r.bbox_xyxy
    ]
    scores = [r.det_score for r in cam_results.values() if r.visible and r.bbox_xyxy]
    n_visible = len(areas)
    mean_area = float(np.mean(areas)) if areas else 0.0
    mean_det = float(np.mean(scores)) if scores else 0.0
    score = n_visible * 1e4 + mean_area * 1e-2 + mean_det * 1e3 + sync_index * 0.01

    return Candidate(
        sync_index=sync_index,
        vehicle_id=vehicle_id,
        cam_results=cam_results,
        score=score,
    )


def select_diverse(candidates: list[Candidate], num: int, min_sync_gap: int) -> list[Candidate]:
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    selected: list[Candidate] = []
    used_sync: list[int] = []
    for cand in ranked:
        if any(abs(cand.sync_index - s) < min_sync_gap for s in used_sync):
            continue
        selected.append(cand)
        used_sync.append(cand.sync_index)
        if len(selected) >= num:
            break
    return selected


def draw_frame(
    frame: np.ndarray,
    result: CamResult,
    *,
    vehicle_id: int,
    sync_index: int,
    cam: int,
) -> np.ndarray:
    img = frame.copy()
    if not result.bbox_xyxy:
        return img

    x1, y1, x2, y2 = (int(round(v)) for v in result.bbox_xyxy)
    color = Visualizer.color_from_id(vehicle_id)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    bx, by = (int(round(result.bottom_center_px[0])), int(round(result.bottom_center_px[1])))
    cv2.circle(img, (bx, by), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (bx, by), 6, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    label = f"ID:{vehicle_id}  cam-{cam}  k={sync_index}"
    cv2.putText(
        img,
        label,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def cam_result_to_json(result: CamResult, *, vehicle_id: int, sync_index: int) -> dict:
    if not result.visible:
        payload: dict = {"visible": False}
        if result.reason:
            payload["reason"] = result.reason
        return payload

    return {
        "visible": True,
        "vehicle_id": vehicle_id,
        "sync_index": sync_index,
        "image_path": result.image_path,
        "frame_idx": result.frame_idx,
        "cam_id": result.cam_id,
        "bbox_xyxy": result.bbox_xyxy,
        "bottom_center_px": result.bottom_center_px,
        "roi_xyxy": result.roi_xyxy,
        "match_iou": result.match_iou,
        "det_score": result.det_score,
        "det_class_id": result.det_class_id,
        "image_source": result.image_path,
    }


def export_candidate(
    cand: Candidate,
    frame_dir: Path,
    dataset: GtaMcmtDataset,
) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    mosaic_frames: list[np.ndarray] = []

    cameras_meta: dict[str, dict] = {}
    for cam in range(dataset.num_cameras):
        result = cand.cam_results.get(cam, CamResult(visible=False, cam=cam))
        cam_key = f"cam{cam}"
        if not result.visible or not result.bbox_xyxy:
            cameras_meta[cam_key] = cam_result_to_json(result, vehicle_id=cand.vehicle_id, sync_index=cand.sync_index)
            continue

        cam_dir = frame_dir / cam_key
        cam_dir.mkdir(parents=True, exist_ok=True)

        image_path = Path(result.image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        annotated = draw_frame(
            frame,
            result,
            vehicle_id=cand.vehicle_id,
            sync_index=cand.sync_index,
            cam=cam,
        )
        cv2.imwrite(str(cam_dir / "frame.jpg"), annotated)
        mosaic_frames.append(annotated)

        meta = cam_result_to_json(result, vehicle_id=cand.vehicle_id, sync_index=cand.sync_index)
        (cam_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        cameras_meta[cam_key] = meta

    if mosaic_frames:
        mosaic = stack_camera_views(mosaic_frames)
        cv2.imwrite(str(frame_dir / "mosaic.jpg"), mosaic)

    frame_meta = {
        "sync_index": cand.sync_index,
        "vehicle_id": cand.vehicle_id,
        "cameras_visible": cand.visible_cams,
        "yolo_ok_cams": cand.yolo_ok_cams,
        "mean_match_iou": cand.mean_match_iou,
        "score": cand.score,
        "cameras": cameras_meta,
    }
    (frame_dir / "meta.json").write_text(json.dumps(frame_meta, indent=2), encoding="utf-8")


def resolve_device(device) -> str | int:
    if device is None:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    if isinstance(device, str) and device.isdigit():
        return int(device)
    return device


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir

    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset_root.resolve()}")
    dataset = GtaMcmtDataset(args.dataset_root)
    print(f"Sync frames: {len(dataset)}")

    print(f"Scanning CSV candidates (stride={args.scan_stride})...")
    raw = scan_csv_candidates(
        dataset,
        scan_stride=args.scan_stride,
        min_cameras=args.min_cameras,
        min_bbox_size=args.min_bbox_size,
    )
    print(f"CSV candidates: {len(raw)}")

    print(f"Loading detector: {args.model}")
    detector = Detector(
        model=args.model,
        target_classes=sorted(VEHICLE_CLASSES),
        conf_thres=args.conf_thres,
        imgsz=args.imgsz,
        device=resolve_device(args.device),
    )

    print("Running YOLO refinement on candidates...")
    valid: list[Candidate] = []
    for i, (k, obj_id, cam_anns) in enumerate(raw):
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(raw)}, valid={len(valid)}")
        cand = evaluate_candidate(
            dataset,
            k,
            obj_id,
            cam_anns,
            detector,
            roi_pad=args.roi_pad,
            min_match_iou=args.min_match_iou,
        )
        if cand is not None:
            valid.append(cand)

    print(f"Valid YOLO-refined candidates: {len(valid)}")
    if not valid:
        raise SystemExit("No valid candidates found. Try lowering --min-match-iou or --scan-stride.")

    selected = select_diverse(valid, args.num_examples, args.min_sync_gap)
    print(f"Selected {len(selected)} examples")

    index: list[dict] = []
    for idx, cand in enumerate(selected, start=1):
        folder = f"frame_{idx:03d}"
        frame_dir = out_dir / folder
        print(
            f"  {folder}: k={cand.sync_index} vehicle_id={cand.vehicle_id} "
            f"cams={cand.visible_cams} score={cand.score:.1f}"
        )
        export_candidate(cand, frame_dir, dataset)
        entry = json.loads((frame_dir / "meta.json").read_text(encoding="utf-8"))
        entry["folder"] = folder
        index.append(entry)

    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Done. Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
