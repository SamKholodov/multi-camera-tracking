"""Visualize overlapping duplicate track pairs and YOLO detections on the same frames."""
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
from core.io.gta_mcmt import GtaMcmtDataset
from core.utils.utilities import Utils
from core.visualization.visualizer import Visualizer
from scripts.duplicate_track_utils import iou_xyxy, load_mot_by_frame, local_to_global

DEFAULT_PAIRS = [
    {"cam": 2, "local_tid1": 727, "local_tid2": 732, "center_frame": 3513},
    {"cam": 0, "local_tid1": 264, "local_tid2": 308, "center_frame": 1537},
]


def load_detector(config_path: Path) -> Detector:
    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    det = cfg.get("detector", {})
    return Detector(
        model=det.get("model", "models/yolo26l_fine_tune_gta.pt"),
        target_classes=det.get("target_classes", [2, 3, 5, 7]),
        device=det.get("device", 0),
        conf_thres=float(det.get("conf_thres", 0.2)),
        imgsz=int(det.get("imgsz", 960)),
    )


def draw_track_pair_frame(
    img: np.ndarray,
    by_frame: dict,
    frame: int,
    tid1: int,
    tid2: int,
    g1: int | None,
    g2: int | None,
    *,
    title: str,
) -> np.ndarray:
    out = img.copy()
    targets = {tid1, tid2}
    for tid, box, conf in by_frame.get(frame, []):
        x1, y1, x2, y2 = map(int, box)
        is_target = tid in targets
        gid = g1 if tid == tid1 else g2 if tid == tid2 else None
        color = (
            (0, 0, 255) if tid == tid1 else (255, 0, 0) if tid == tid2 else (180, 180, 180)
        )
        thickness = 3 if is_target else 1
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if is_target:
            label = f"L{tid}/G{gid} {conf:.2f}"
            cv2.putText(
                out,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
    cv2.putText(
        out,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def draw_detection_frame(
    img: np.ndarray,
    dets: np.ndarray,
    *,
    title: str,
    highlight_iou: float = 0.5,
) -> np.ndarray:
    out = img.copy()
    dup_pairs: list[tuple[int, int, float]] = []
    rows = [d for d in dets]
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            iv = iou_xyxy(rows[i][:4].tolist(), rows[j][:4].tolist())
            if iv >= highlight_iou:
                dup_pairs.append((i, j, iv))

    dup_indices = {i for pair in dup_pairs for i in pair[:2]}
    for idx, det in enumerate(rows):
        x1, y1, x2, y2 = map(int, det[:4])
        conf = float(det[4])
        is_dup = idx in dup_indices
        color = (0, 0, 255) if is_dup else (0, 255, 0)
        thickness = 3 if is_dup else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            out,
            f"{conf:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    header = f"{title}  dets={len(rows)}  dup_pairs={len(dup_pairs)}"
    cv2.putText(
        out,
        header,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def render_pair_clip(
    *,
    run_dir: Path,
    dataset: GtaMcmtDataset,
    detector: Detector,
    cam: int,
    tid1: int,
    tid2: int,
    center_frame: int,
    out_dir: Path,
    half_window: int,
    fps: float,
) -> None:
    per_cam_local = run_dir / "per_cam_local"
    by_frame = load_mot_by_frame(per_cam_local / f"c{cam:03d}.txt")
    g1 = local_to_global(per_cam_local, cam, tid1, center_frame)
    g2 = local_to_global(per_cam_local, cam, tid2, center_frame)

    start = max(1, center_frame - half_window)
    end = center_frame + half_window
    frames = list(range(start, end + 1))

    sample_img = dataset.read_sync(center_frame - 1)[cam]
    h, w = sample_img.shape[:2]
    w, h = w - (w % 2), h - (h % 2)

    tag = f"c{cam:03d}_L{tid1}_L{tid2}_f{center_frame}"
    writers = {
        "tracks": cv2.VideoWriter(
            str(out_dir / f"{tag}_tracks.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        ),
        "dets_raw": cv2.VideoWriter(
            str(out_dir / f"{tag}_dets_raw.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        ),
        "dets_post": cv2.VideoWriter(
            str(out_dir / f"{tag}_dets_post.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        ),
    }

    for frame in frames:
        sync_k = frame - 1
        img = dataset.read_sync(sync_k)[cam]
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h))

        track_vis = draw_track_pair_frame(
            img,
            by_frame,
            frame,
            tid1,
            tid2,
            g1,
            g2,
            title=f"tracks frame={frame}  L{tid1}/G{g1} vs L{tid2}/G{g2}",
        )
        writers["tracks"].write(track_vis)

        raw_dets, _ = detector.detect(img)
        raw_arr = np.asarray(raw_dets, dtype=np.float32) if raw_dets else np.empty((0, 6))
        post_arr = Utils.postprocess_detections(raw_arr.copy())

        writers["dets_raw"].write(
            draw_detection_frame(
                img,
                raw_arr,
                title=f"YOLO raw frame={frame}",
            )
        )
        writers["dets_post"].write(
            draw_detection_frame(
                img,
                post_arr,
                title=f"YOLO post-NMS frame={frame}",
            )
        )

    for writer in writers.values():
        writer.release()

    print(f"Saved clips for {tag} -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/configs_gta/geo_ablation/geo_tight"),
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("configs_gta/geo_ablation/geo_tight.yaml"),
    )
    ap.add_argument("--dataset-root", type=Path, default=Path("datasets/gta_mcmt"))
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <run-dir>/duplicate_analysis/viz",
    )
    ap.add_argument("--half-window", type=int, default=30)
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    out_dir = args.out_dir or (args.run_dir / "duplicate_analysis" / "viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = GtaMcmtDataset(args.dataset_root)
    detector = load_detector(args.config)

    for pair in DEFAULT_PAIRS:
        render_pair_clip(
            run_dir=args.run_dir,
            dataset=dataset,
            detector=detector,
            cam=int(pair["cam"]),
            tid1=int(pair["local_tid1"]),
            tid2=int(pair["local_tid2"]),
            center_frame=int(pair["center_frame"]),
            out_dir=out_dir,
            half_window=args.half_window,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
