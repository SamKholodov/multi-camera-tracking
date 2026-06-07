import argparse
from pathlib import Path

import numpy as np
import yaml

from core.io.calibration import load_homography_image_to_world
from core.io.mot_detections import resolve_cityflow_det_paths
from core.io.roi import resolve_roi_paths
from pipeline import MultiCameraTrackingPipeline
from pipeline import SingleCameraTrackerPipeline


def _load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_target_classes(raw):
    """YAML-friendly: ints for YOLO ``classes=`` (not strings, not one comma-separated scalar)."""
    if raw is None:
        return [0]
    if isinstance(raw, bool):
        return [int(raw)]
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(" ", "").split(",") if p.strip()]
        return [int(p) for p in parts] if parts else [0]
    if isinstance(raw, list):
        out = []
        for item in raw:
            if isinstance(item, bool):
                out.append(int(item))
            elif isinstance(item, int):
                out.append(item)
            elif isinstance(item, str):
                if "," in item:
                    for p in item.split(","):
                        p = p.strip()
                        if p:
                            out.append(int(p))
                else:
                    out.append(int(item))
            else:
                out.append(int(item))
        return out if out else [0]
    return [int(raw)]


def _maybe_homography(matrix_like):
    if matrix_like is None:
        return np.eye(3, dtype=np.float32)
    return np.asarray(matrix_like, dtype=np.float32)


def _looks_like_roi_matrix(value):
    if not isinstance(value, list) or not value:
        return False
    if not all(isinstance(row, list) for row in value):
        return False
    return all(
        not isinstance(cell, (list, tuple, dict))
        for row in value
        for cell in row
    )


def _maybe_roi_paths(sources, roi_cfg):
    """Resolve ROI masks from YAML.

    * ``null`` / omitted — no ROI filtering.
    * ``"auto"`` — use ``<parent_of_video>/roi.jpg`` for each source.
    * string path — one ROI image path for single-camera configs.
    * 2D list — one ROI mask matrix for single-camera configs.
    * list of paths/matrices — explicit per-camera ROI (use null to skip one camera).
    """
    if roi_cfg is None:
        return None
    if isinstance(roi_cfg, str) and roi_cfg.lower() == "auto":
        return [
            str(Path(src).parent / "roi.jpg")
            if (Path(src).parent / "roi.jpg").exists()
            else None
            for src in sources
        ]
    if isinstance(roi_cfg, str):
        if len(sources) != 1:
            raise ValueError("String roi path is only valid for single_camera; use a list for multi_camera")
        return [roi_cfg]
    if isinstance(roi_cfg, list):
        if _looks_like_roi_matrix(roi_cfg):
            if len(sources) != 1:
                raise ValueError("Single ROI matrix is only valid for single_camera; use one matrix per camera")
            return [roi_cfg]
        return resolve_roi_paths(sources, roi_cfg)
    raise ValueError(f"Unsupported roi config: {roi_cfg!r}")


def _resolve_image_to_world(item):
    """Path to ``calibration.txt`` → ``H_image_to_world`` (cached as ``calibration_i2w.txt``)."""
    if isinstance(item, str):
        return np.asarray(load_homography_image_to_world(item), dtype=np.float64)
    return np.asarray(item, dtype=np.float64)


def _maybe_homographies(homos_like):
    """List of ``H_image_to_world`` per camera."""
    if homos_like is None:
        return None
    return [_resolve_image_to_world(item) for item in homos_like]


def _maybe_detection_files(sources, detector_cfg):
    """Resolve per-camera CityFlow ``det/*.txt`` paths when ``detector.source`` is set."""
    source_mode = str(detector_cfg.get("source", "yolo")).lower().strip()
    if source_mode in ("yolo", "ultralytics", ""):
        return None
    if source_mode not in ("cityflow_det", "file", "det"):
        raise ValueError(
            f"Unknown detector.source={source_mode!r}; use 'yolo' or 'cityflow_det'"
        )
    det_files = detector_cfg.get("det_files")
    det_basename = detector_cfg.get("det_basename", "det_mask_rcnn.txt")
    return resolve_cityflow_det_paths(sources, det_files=det_files, det_basename=det_basename)


def run_from_config(config):
    run_mode = config.get("run_mode", "multi_camera")
    detector_cfg = config.get("detector", {})
    tracker_cfg = config.get("tracker", {})
    output_cfg = config.get("output", {})

    detector_model = detector_cfg.get("model", "yolov8m.pt")
    target_classes = _normalize_target_classes(
        detector_cfg.get("target_classes", [0])
    )
    detector_conf_thres = float(detector_cfg.get("conf_thres", 0.3))
    detector_device = detector_cfg.get("device", None)

    visualize = bool(output_cfg.get("visualize", True))
    save_video = bool(output_cfg.get("save_video", False))
    output_path = output_cfg.get("output_path", "outputs/output.mp4")
    save_video_dir = output_cfg.get("save_video_dir")
    if save_video_dir is None and save_video:
        # Default: one full-res video per camera with global ids.
        save_video_dir = str(Path(output_path).parent / "videos")

    if run_mode == "single_camera":
        single_cfg = config.get("single_camera", {})
        source = single_cfg.get("source", "")
        if not source:
            raise ValueError("single_camera.source is empty in config")

        roi_paths = _maybe_roi_paths([source], single_cfg.get("roi"))
        roi_path = roi_paths[0] if roi_paths else None
        det_files = _maybe_detection_files([source], detector_cfg)
        det_file = det_files[0] if det_files else single_cfg.get("detection_file")

        homo_cfg = single_cfg.get("homo")
        H_i2w = (
            _resolve_image_to_world(homo_cfg)
            if homo_cfg is not None
            else np.eye(3, dtype=np.float64)
        )

        pipeline = SingleCameraTrackerPipeline(
            source=source,
            tracker_config=tracker_cfg,
            model=detector_model,
            target_classes=target_classes,
            detector_conf_thres=detector_conf_thres,
            detector_device=detector_device,
            cam_id=int(single_cfg.get("cam_id", 0)),
            homo=H_i2w,
            max_history_gap_frames=int(single_cfg.get("max_history_gap_frames", 30)),
            roi_path=roi_path,
            detection_file=det_file,
        )
        pipeline.run(
            visualize=visualize,
            save=save_video,
            save_tracks=bool(output_cfg.get("save_tracks_json", True)),
            output_path=output_path,
        )
        return

    if run_mode == "multi_camera":
        multi_cfg = config.get("multi_camera", {})
        sources = multi_cfg.get("sources", [])
        if not sources:
            raise ValueError("multi_camera.sources is empty in config")

        detection_files = _maybe_detection_files(sources, detector_cfg)

        pipeline = MultiCameraTrackingPipeline(
            sources=sources,
            tracker_config=tracker_cfg,
            model=detector_model,
            target_classes=target_classes,
            detector_conf_thres=detector_conf_thres,
            detector_device=detector_device,
            homos=_maybe_homographies(multi_cfg.get("homos")),
            association_cost_threshold=float(
                multi_cfg.get("association_cost_threshold", 0.35)
            ),
            association_reid_weight=float(
                multi_cfg.get("association_reid_weight", 0.5)
            ),
            geometry_max_distance=float(
                multi_cfg.get("geometry_max_distance", 25.0)
            ),
            max_cross_cam_gap_frames=int(
                multi_cfg.get("max_cross_cam_gap_frames", 300)
            ),
            max_history_gap_frames=int(multi_cfg.get("max_history_gap_frames", 30)),
            mapping_clear_after_lost_frames=multi_cfg.get(
                "mapping_clear_after_lost_frames"
            ),
            cam_ids=multi_cfg.get("cam_ids"),
            results_dir=multi_cfg.get("results_dir"),
            roi_paths=_maybe_roi_paths(sources, multi_cfg.get("roi")),
            detection_files=detection_files,
            video_fps=float(output_cfg.get("video_fps", 10)),
            detector_batch_inference=bool(detector_cfg.get("batch_inference", True)),
        )
        pipeline.run(
            visualize=visualize,
            save=save_video,
            output_path=output_path,
            save_video_dir=save_video_dir if save_video else None,
        )
        return

    raise ValueError(f"Unknown run_mode: {run_mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/baseline.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    run_from_config(config)


if __name__ == "__main__":
    main()
