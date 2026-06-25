"""Export CityFlow GT boxes as a single-class YOLO dataset.

S02 validation mode (default): sync-aligned GT from c006-c009 via sync_manifest.json.
Train mode (--train-scenes): official train scenes (S01, S03, S04) from vdo.avi.

Usage:
    # S02 validation export (backward compatible)
    python scripts/finetune/export_cityflow_yolo_dataset.py --clean

    # Official train split export
    python scripts/finetune/export_cityflow_yolo_dataset.py --train-scenes S01 S03 S04 --clean
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.io.roi import ROIFilter
from scripts.cityflow_sync_eval import (
    align_gt_to_sync,
    load_sync_manifest,
    sync_length_frames,
    sync_skip_by_cam,
)
from scripts.gt_to_det import load_gt_mot

DEFAULT_DATASET_ROOT = _ROOT / "datasets/AICity22_Track1_MTMC_Tracking"
DEFAULT_GT_ROOT = DEFAULT_DATASET_ROOT / "validation/S02"
DEFAULT_OUT = _ROOT / "datasets/cityflow_yolo_finetune"
DEFAULT_TRAIN_OUT = _ROOT / "datasets/cityflow_yolo_finetune_train"
DEFAULT_TRAIN_SCENES = ("S01", "S03", "S04")
CAMERAS = (6, 7, 8, 9)
VEHICLE_CLASS = 0
SEED = 42


def mot_to_yolo_lines(gt_rows: np.ndarray, img_w: int, img_h: int) -> list[str]:
    lines: list[str] = []
    for row in gt_rows:
        x, y, w, h = row[2:6]
        if w <= 0 or h <= 0:
            continue
        cx = (x + w / 2.0) / img_w
        cy = (y + h / 2.0) / img_h
        nw = w / img_w
        nh = h / img_h
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)
        if nw <= 0 or nh <= 0:
            continue
        lines.append(f"{VEHICLE_CLASS} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def load_sync_gt_by_frame(
    gt_root: Path,
    cam: int,
    *,
    apply_roi: bool,
) -> dict[int, np.ndarray]:
    gt_path = gt_root / f"c{cam:03d}" / "gt" / "gt.txt"
    manifest = load_sync_manifest(gt_root)
    skips = sync_skip_by_cam(manifest)
    sync_len = sync_length_frames(manifest)
    gt = load_gt_mot(gt_path)
    gt = align_gt_to_sync(gt, skips.get(cam, 0), sync_len)
    if apply_roi:
        roi_path = gt_root / f"c{cam:03d}" / "roi.jpg"
        if roi_path.is_file():
            gt = ROIFilter.from_path(roi_path).filter_mot(gt)

    by_frame: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in gt:
        by_frame[int(row[0])].append(row)
    return {f: np.vstack(rows) for f, rows in by_frame.items()}


def load_raw_gt_by_frame(
    cam_dir: Path,
    *,
    apply_roi: bool,
    frame_stride: int = 1,
) -> dict[int, np.ndarray]:
    gt_path = cam_dir / "gt" / "gt.txt"
    gt = load_gt_mot(gt_path)
    if apply_roi:
        roi_path = cam_dir / "roi.jpg"
        if roi_path.is_file():
            gt = ROIFilter.from_path(roi_path).filter_mot(gt)

    by_frame: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in gt:
        frame_id = int(row[0])
        if frame_stride > 1 and (frame_id - 1) % frame_stride != 0:
            continue
        by_frame[frame_id].append(row)
    return {f: np.vstack(rows) for f, rows in by_frame.items()}


def discover_train_cameras(
    dataset_root: Path,
    scenes: tuple[str, ...],
    *,
    camera_ids: tuple[int, ...] | None = None,
) -> list[tuple[str, int, Path]]:
    wanted = set(camera_ids) if camera_ids else None
    cams: list[tuple[str, int, Path]] = []
    for scene in scenes:
        scene_dir = dataset_root / "train" / scene
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Train scene not found: {scene_dir}")
        for cam_dir in sorted(scene_dir.iterdir()):
            if not cam_dir.is_dir() or not cam_dir.name.startswith("c"):
                continue
            cam_id = int(cam_dir.name[1:])
            if wanted is not None and cam_id not in wanted:
                continue
            cams.append((scene, cam_id, cam_dir))
    return cams


def _split_manifest_path(out_dir: Path) -> Path:
    return out_dir / "split_manifest.json"


def _load_split_manifest(out_dir: Path) -> dict[tuple[int, int], str] | None:
    import json

    path = _split_manifest_path(out_dir)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {(int(cam), int(fid)): split for cam, fid, split in raw}


def _save_split_manifest(
    out_dir: Path,
    split_by_key: dict[tuple[int, int], str],
) -> None:
    import json

    rows = [[cam, fid, split] for (cam, fid), split in sorted(split_by_key.items())]
    _split_manifest_path(out_dir).write_text(
        json.dumps(rows, separators=(",", ":")),
        encoding="utf-8",
    )


def _existing_export(out_dir: Path, stem: str) -> tuple[str, Path] | tuple[None, None]:
    for split in ("train", "val"):
        img_path = out_dir / split / "images" / f"{stem}.jpg"
        lbl_path = out_dir / split / "labels" / f"{stem}.txt"
        if img_path.is_file() and lbl_path.is_file():
            return split, lbl_path
    return None, None


def export_camera_frames(
    video_path: Path,
    cam: int,
    frame_ids: set[int],
    out_dir: Path,
    split_by_key: dict[tuple[int, int], str],
    gt_cache: dict[tuple[int, int], np.ndarray],
    stats: dict[str, int],
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    for frame_id in sorted(frame_ids):
        key = (cam, frame_id)
        split = split_by_key[key]
        stem = f"c{cam:03d}_f{frame_id:05d}"
        existing_split, lbl_path = _existing_export(out_dir, stem)
        if existing_split is not None:
            assert lbl_path is not None
            stats[existing_split] += 1
            stats[f"boxes_{existing_split}"] += len(
                lbl_path.read_text(encoding="utf-8").strip().splitlines()
            )
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_id - 1, 0))
        ok, frame = cap.read()
        if not ok:
            print(f"[WARN] missing frame cam={cam} frame={frame_id}")
            continue
        h, w = frame.shape[:2]
        yolo_lines = mot_to_yolo_lines(gt_cache[key], w, h)
        if not yolo_lines:
            continue
        if frame is None or frame.size == 0:
            print(f"[WARN] empty frame cam={cam} frame={frame_id}")
            continue
        img_path = out_dir / split / "images" / f"{stem}.jpg"
        lbl_path = out_dir / split / "labels" / f"{stem}.txt"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(img_path), frame):
            print(f"[WARN] failed to write {img_path}")
            continue
        lbl_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        stats[split] += 1
        stats[f"boxes_{split}"] += len(yolo_lines)
    cap.release()


def _clear_dir(path: Path) -> None:
    import shutil

    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_file():
            child.unlink(missing_ok=True)
        else:
            shutil.rmtree(child, ignore_errors=True)


def _prepare_out_dir(out_dir: Path, *, clean: bool) -> None:
    if clean and out_dir.exists():
        for sub in ("train/images", "train/labels", "val/images", "val/labels"):
            _clear_dir(out_dir / sub)
        yaml_path = out_dir / "data.yaml"
        if yaml_path.is_file():
            yaml_path.unlink(missing_ok=True)
        manifest = _split_manifest_path(out_dir)
        if manifest.is_file():
            manifest.unlink(missing_ok=True)

    for split in ("train", "val"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def _write_data_yaml(out_dir: Path) -> Path:
    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["vehicle"],
    }
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return yaml_path


def _collect_train_frames(
    dataset_root: Path,
    scenes: tuple[str, ...],
    *,
    apply_roi: bool,
    frame_stride: int,
    camera_ids: tuple[int, ...] | None = None,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], np.ndarray], dict[int, Path], tuple[int, ...]]:
    train_cams = discover_train_cameras(dataset_root, scenes, camera_ids=camera_ids)
    if not train_cams:
        raise SystemExit(f"No cameras found under train scenes: {scenes}")

    frame_keys: list[tuple[int, int]] = []
    gt_cache: dict[tuple[int, int], np.ndarray] = {}
    video_by_cam: dict[int, Path] = {}
    cameras: list[int] = []

    for scene, cam, cam_dir in train_cams:
        by_frame = load_raw_gt_by_frame(
            cam_dir,
            apply_roi=apply_roi,
            frame_stride=frame_stride,
        )
        if not by_frame:
            print(f"[WARN] no GT frames for {scene}/c{cam:03d}")
            continue
        video_path = cam_dir / "vdo.avi"
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing video: {video_path}")
        video_by_cam[cam] = video_path
        cameras.append(cam)
        for frame_id, rows in by_frame.items():
            key = (cam, frame_id)
            frame_keys.append(key)
            gt_cache[key] = rows

    return frame_keys, gt_cache, video_by_cam, tuple(cameras)


def _split_and_export(
    frame_keys: list[tuple[int, int]],
    gt_cache: dict[tuple[int, int], np.ndarray],
    out_dir: Path,
    *,
    val_ratio: float,
    video_by_cam: dict[int, Path],
    cameras: tuple[int, ...],
    manifest_only: bool = False,
) -> dict[str, int]:
    if not frame_keys:
        raise SystemExit("No GT rows after filtering — check dataset paths.")

    split_by_key = _load_split_manifest(out_dir)
    expected_keys = set(frame_keys)
    if split_by_key is None or set(split_by_key.keys()) != expected_keys:
        rng = random.Random(SEED)
        rng.shuffle(frame_keys)
        n_val = max(1, int(round(len(frame_keys) * val_ratio)))
        val_keys = set(frame_keys[:n_val])
        split_by_key = {key: ("val" if key in val_keys else "train") for key in frame_keys}
        _save_split_manifest(out_dir, split_by_key)

    stats = {"train": 0, "val": 0, "boxes_train": 0, "boxes_val": 0}
    if manifest_only:
        return stats

    frames_by_cam: dict[int, set[int]] = defaultdict(set)
    for cam, frame_id in frame_keys:
        frames_by_cam[cam].add(frame_id)

    for cam in cameras:
        export_camera_frames(
            video_by_cam[cam],
            cam,
            frames_by_cam.get(cam, set()),
            out_dir,
            split_by_key,
            gt_cache,
            stats,
        )
    return stats


def export_dataset(
    gt_root: Path,
    out_dir: Path,
    *,
    cameras: tuple[int, ...] = CAMERAS,
    val_ratio: float = 0.2,
    apply_roi: bool = True,
    clean: bool = False,
) -> Path:
    _prepare_out_dir(out_dir, clean=clean)

    frame_keys: list[tuple[int, int]] = []
    gt_cache: dict[tuple[int, int], np.ndarray] = {}
    for cam in cameras:
        by_frame = load_sync_gt_by_frame(gt_root, cam, apply_roi=apply_roi)
        for frame_id, rows in by_frame.items():
            key = (cam, frame_id)
            frame_keys.append(key)
            gt_cache[key] = rows

    video_by_cam = {cam: gt_root / f"c{cam:03d}" / "vdo_synch.avi" for cam in cameras}
    stats = _split_and_export(
        frame_keys,
        gt_cache,
        out_dir,
        val_ratio=val_ratio,
        video_by_cam=video_by_cam,
        cameras=cameras,
    )

    yaml_path = _write_data_yaml(out_dir)
    print(f"Exported {stats['train']} train / {stats['val']} val images")
    print(f"Boxes: train={stats['boxes_train']} val={stats['boxes_val']}")
    print(f"data.yaml -> {yaml_path}")
    return yaml_path


def export_train_dataset(
    dataset_root: Path,
    out_dir: Path,
    *,
    scenes: tuple[str, ...] = DEFAULT_TRAIN_SCENES,
    manifest_scenes: tuple[str, ...] | None = None,
    camera_ids: tuple[int, ...] | None = None,
    val_ratio: float = 0.2,
    apply_roi: bool = True,
    frame_stride: int = 1,
    clean: bool = False,
    manifest_only: bool = False,
) -> Path:
    _prepare_out_dir(out_dir, clean=clean)

    manifest_scenes = manifest_scenes or scenes
    frame_keys, gt_cache, video_by_cam, _ = _collect_train_frames(
        dataset_root,
        manifest_scenes,
        apply_roi=apply_roi,
        frame_stride=frame_stride,
    )

    _, _, export_video_by_cam, export_cameras = _collect_train_frames(
        dataset_root,
        scenes,
        apply_roi=apply_roi,
        frame_stride=frame_stride,
        camera_ids=camera_ids,
    )
    video_by_cam.update(export_video_by_cam)

    stats = _split_and_export(
        frame_keys,
        gt_cache,
        out_dir,
        val_ratio=val_ratio,
        video_by_cam=video_by_cam,
        cameras=() if manifest_only else export_cameras,
        manifest_only=manifest_only,
    )

    yaml_path = _write_data_yaml(out_dir)
    if manifest_only:
        print(f"Train scenes (manifest): {', '.join(manifest_scenes)}")
        print(f"Manifest frames: {len(frame_keys)}")
        print(f"data.yaml -> {yaml_path}")
        return yaml_path

    print(f"Train scenes exported: {', '.join(scenes)} ({len(export_cameras)} cameras)")
    print(f"Exported {stats['train']} train / {stats['val']} val images")
    print(f"Boxes: train={stats['boxes_train']} val={stats['boxes_val']}")
    print(f"data.yaml -> {yaml_path}")
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="AICity dataset root (train mode)",
    )
    ap.add_argument(
        "--train-scenes",
        nargs="+",
        default=None,
        metavar="SCENE",
        help="Export official train scenes (e.g. S01 S03 S04); omit for S02 validation export",
    )
    ap.add_argument(
        "--manifest-scenes",
        nargs="+",
        default=None,
        metavar="SCENE",
        help="Scenes for train/val split manifest (default: same as --train-scenes)",
    )
    ap.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only write split manifest and data.yaml (no frame export)",
    )
    ap.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--cameras", nargs="+", type=int, default=None, help="Camera ids (S02 or train mode)")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--frame-stride", type=int, default=1, help="Keep every Nth GT frame (train mode)")
    ap.add_argument("--no-roi", action="store_true", help="Disable roi.jpg filtering")
    ap.add_argument("--clean", action="store_true", help="Remove output dir before export")
    args = ap.parse_args()

    if args.train_scenes:
        out_dir = args.out_dir or DEFAULT_TRAIN_OUT
        camera_ids = tuple(args.cameras) if args.cameras else None
        export_train_dataset(
            args.dataset_root,
            out_dir,
            scenes=tuple(args.train_scenes),
            manifest_scenes=tuple(args.manifest_scenes) if args.manifest_scenes else None,
            camera_ids=camera_ids,
            val_ratio=args.val_ratio,
            apply_roi=not args.no_roi,
            frame_stride=args.frame_stride,
            clean=args.clean,
            manifest_only=args.manifest_only,
        )
    else:
        out_dir = args.out_dir or DEFAULT_OUT
        cameras = tuple(args.cameras) if args.cameras else CAMERAS
        export_dataset(
            args.gt_root,
            out_dir,
            cameras=cameras,
            val_ratio=args.val_ratio,
            apply_roi=not args.no_roi,
            clean=args.clean,
        )


if __name__ == "__main__":
    main()
