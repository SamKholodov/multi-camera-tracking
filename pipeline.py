import numpy as np
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml

from scipy.optimize import linear_sum_assignment

from core.detector.detector import Detector
from core.io.calibration import project_bbox_bottom_center
from core.io.camera_manager import CameraManager
from core.mot.types import (
    TRACK_NCOLS,
    enrich_tracks_world,
    homography_valid,
    world_point_from_row,
)
from core.io.track_history import save_tracks_history_json
from core.io.mcmt_writer import MCMTResultWriter
from core.io.mot_detections import MotDetectionStore
from core.io.roi import ROIFilter
from core.io.zones import ZoneMap
from core.mot.appearance import normalize_appearance_mode
from core.mot.association.cross_camera import (
    association_cost_for_match,
    CrossCameraAssociationConfig,
)
from core.mot.association.same_frame_link import UnionFind, find_same_frame_links
from core.mot.global_track import CamObservation, GlobalTrackStore
from core.mot.local_zone_state import LocalZoneTracker
from core.mot.reid_batch import batch_reid_features
from core.mot.contact_point_batch import batch_contact_point_uv
from core.mot.tracker.bot_sort_tracker import BotSortTracker
from core.mot.tracker.deepocsort_tracker import DeepOcSortTracker
from core.mot.tracker.sort_tracker import SortTracker
from core.utils.fps import FpsCollector, sct_timing_ms
from core.utils.utilities import Utils
from core.visualization.visualizer import Visualizer

def _build_reid_backend(reid_weights, device, half, *, preprocess_name=None):
    """Local ReID backend with ``get_features(xyxys, img)`` for DeepOcSort."""
    from core.reid.core import ReID

    kwargs = dict(device=device, half=half)
    if reid_weights is not None:
        kwargs["weights"] = reid_weights
    if preprocess_name is not None:
        kwargs["preprocess_name"] = preprocess_name
    return ReID(**kwargs).model


def _maybe_build_shared_reid_model(tracker_config, *, share_reid_model=True):
    """Build one shared ReID backend for multi-camera trackers that use ReID."""
    if not share_reid_model:
        return None
    cfg = dict(tracker_config or {})
    tracker_type = str(cfg.get("type", "botsort")).lower().strip()
    if tracker_type in ("deepocsort", "deep_ocsort"):
        if not bool(cfg.get("use_embeddings", False)):
            return None
    elif tracker_type == "botsort":
        if not bool(cfg.get("use_default_reid", True)):
            return None
    else:
        return None
    return _build_reid_backend(
        cfg.get("reid_weights"),
        cfg.get("device", 0),
        bool(cfg.get("half", False)),
        preprocess_name=cfg.get("reid_preprocess"),
    )


def _maybe_build_contact_point_model(contact_cfg):
    """Build shared contact point model when enabled in YAML."""
    cfg = dict(contact_cfg or {})
    if not bool(cfg.get("enabled", False)):
        return None
    weights = cfg.get("weights")
    if not weights:
        return None
    weights_path = Path(weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"contact_point.weights not found: {weights_path}")
    from core.geometry.contact_point.inference import ContactPointInference

    return ContactPointInference(
        weights=weights_path,
        device=cfg.get("device"),
        img_size=int(cfg.get("img_size", 224)),
        bbox_pad_ratio=float(cfg.get("bbox_pad_ratio", 0.0)),
    )


def _create_tracker(tracker_config, *, shared_reid_model=None):
    """Build Sort, BotSort, or DeepOcSort tracker from YAML ``tracker`` dict."""
    cfg = dict(tracker_config or {})
    tracker_type = str(cfg.pop("type", "botsort")).lower().strip()
    reid_keys = (
        "reid_weights",
        "device",
        "half",
        "use_default_reid",
        "use_embeddings",
        "custom_reid_extractor",
        "reid_preprocess",
        "share_reid_model",
        "batch_reid",
    )
    appearance_keys = ("appearance_update", "reid_accum_conf_thresh")
    if tracker_type in ("deepocsort", "deep_ocsort"):
        use_embeddings = bool(cfg.pop("use_embeddings", False))
        reid_weights = cfg.pop("reid_weights", None)
        device = cfg.pop("device", 0)
        half = bool(cfg.pop("half", False))
        cfg.pop("use_default_reid", None)
        cfg.pop("reid_preprocess", None)
        cfg.pop("share_reid_model", None)
        custom_reid_extractor = cfg.pop("custom_reid_extractor", None)
        reid_model = None
        if use_embeddings:
            if shared_reid_model is not None:
                reid_model = shared_reid_model
            else:
                reid_model = _build_reid_backend(
                    reid_weights,
                    device,
                    half,
                    preprocess_name=tracker_config.get("reid_preprocess"),
                )
        return DeepOcSortTracker(
            reid_model=reid_model,
            use_embeddings=use_embeddings,
            custom_reid_extractor=custom_reid_extractor,
            **cfg,
        )
    if tracker_type == "sort":
        for k in (*reid_keys, *appearance_keys):
            cfg.pop(k, None)
        return SortTracker(**cfg)
    if tracker_type == "botsort":
        use_default_reid = bool(cfg.pop("use_default_reid", True))
        reid_weights = cfg.pop("reid_weights", "models/osnet_ibn_x1_0_msmt17.pt")
        device = cfg.pop("device", 0)
        half = bool(cfg.pop("half", False))
        custom_reid_extractor = cfg.pop("custom_reid_extractor", None)
        appearance_update = cfg.pop("appearance_update", "aaf")
        reid_accum_conf_thresh = cfg.pop("reid_accum_conf_thresh", None)
        for k in ("use_embeddings", "reid_preprocess", "share_reid_model", "batch_reid"):
            cfg.pop(k, None)
        reid_model = None
        if use_default_reid:
            if shared_reid_model is not None:
                reid_model = shared_reid_model
            else:
                reid_model = _build_reid_backend(
                    reid_weights,
                    device,
                    half,
                    preprocess_name=tracker_config.get("reid_preprocess"),
                )
        return BotSortTracker(
            reid_model=reid_model,
            use_default_reid=use_default_reid,
            custom_reid_extractor=custom_reid_extractor,
            appearance_update=appearance_update,
            **cfg,
        )
    for k in (*reid_keys, *appearance_keys):
        cfg.pop(k, None)
    raise ValueError(
        f"Unknown tracker.type={tracker_type!r}. "
        "Supported: sort, botsort, deepocsort, deep_ocsort"
    )


class SingleCameraTrackerPipeline:
    def __init__(
        self,
        source,
        tracker_config,
        model="yolov8m.pt",
        target_classes=None,
        detector_conf_thres=0.3,
        detector_device=None,
        detector_imgsz=960,
        cam_id=0,
        homo=np.eye(3, dtype=np.float64),
        max_history_gap_frames=30,
        roi_path=None,
        detection_file=None,
        shared_reid_model=None,
        shared_detector=None,
        world_anchor: str = "bottom_center",
        contact_point_model=None,
        contact_batch_size: int | None = None,
        contact_conf_thresh: float = 0.0,
    ):
        self.source = source
        self.tracker = _create_tracker(
            tracker_config,
            shared_reid_model=shared_reid_model,
        )
        self.detection_store = None
        self.detector = None
        self._shared_detector = None
        if detection_file is not None:
            self.detection_store = MotDetectionStore(
                detection_file, conf_thres=detector_conf_thres
            )
        elif shared_detector is not None:
            self._shared_detector = shared_detector
        else:
            if target_classes is None:
                target_classes = [2, 3, 5, 7]
            self.detector = Detector(
                model=model,
                target_classes=target_classes,
                conf_thres=detector_conf_thres,
                device=detector_device,
                imgsz=detector_imgsz,
            )
        self.visualizer = Visualizer()
        self.tracks = {}
        self.frame_idx = 0
        self.cam_id = cam_id
        # ``homo`` is **H_image_to_world** (see ``core.io.calibration``).
        self.homo = np.asarray(homo, dtype=np.float64)
        # After this many frames without a detection for a track, stop appending
        # placeholder rows (has_detection=0). None = keep appending until video end.
        self.max_history_gap_frames = max_history_gap_frames
        self.roi_filter = ROIFilter.from_spec(roi_path) if roi_path is not None else None
        self.last_frame_ms = None
        self.world_anchor = str(world_anchor).lower().strip()
        self.contact_point_model = contact_point_model
        self.contact_batch_size = contact_batch_size
        self.contact_conf_thresh = float(contact_conf_thresh)

    def _filter_detections_roi(self, dets: np.ndarray) -> np.ndarray:
        if self.roi_filter is None or dets is None or len(dets) == 0:
            return dets
        return self.roi_filter.filter_xyxy_array(dets)

    def _filter_tracks_roi(self, tracks: np.ndarray) -> np.ndarray:
        if self.roi_filter is None or tracks is None or len(tracks) == 0:
            return tracks
        return self.roi_filter.filter_xyxy_array(tracks)

    def _tracker_appearance_maps(self):
        match_map = {}
        raw_map = {}
        count_map = {}
        tracker = self.tracker
        if hasattr(tracker, "get_track_feature_map"):
            match_map = tracker.get_track_feature_map()
        if hasattr(tracker, "get_track_appearance_raw_map"):
            raw_map = tracker.get_track_appearance_raw_map()
        if hasattr(tracker, "get_track_appearance_update_count_map"):
            count_map = tracker.get_track_appearance_update_count_map()
        return match_map, raw_map, count_map

    def _update_tracks_storage(self, tracks):
        _, raw_map, count_map = self._tracker_appearance_maps()
        updated_ids = set()
        for t in tracks:
            x1, y1, x2, y2, tid, conf, det_idx, has_detection = t[:8]
            tid = int(tid)
            updated_ids.add(tid)


            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            bcx = cx
            bcy = float(y2)

            wpt = world_point_from_row(t)
            projected_bc = [wpt[0], wpt[1]] if wpt is not None else None

            if tid not in self.tracks:
                self.tracks[tid] = {
                    "track_id": tid,
                    "start_frame": self.frame_idx,
                    "last_frame": self.frame_idx,
                    "length": 1,
                    "frames": [self.frame_idx],
                    "bboxes": [[float(x1), float(y1), float(x2), float(y2)]],
                    "bbox_centers": [[cx, cy]],
                    "bottom_centers": [[bcx, bcy]],
                    "projected_bcenters": [projected_bc],
                    "state" : "pending",
                    "cam_id": self.cam_id,
                    "reid_emb": None,
                    "appearance_update_count": 0,
                    "has_detection": [int(has_detection)],
                }
            else:
                tr = self.tracks[tid]
                tr["last_frame"] = self.frame_idx
                tr["length"] += 1
                tr["frames"].append(self.frame_idx)
                tr["bboxes"].append([float(x1), float(y1), float(x2), float(y2)])
                tr["bbox_centers"].append([cx, cy])
                tr["bottom_centers"].append([bcx, bcy])
                tr["projected_bcenters"].append(projected_bc)
                tr["has_detection"].append(int(has_detection))

            raw = raw_map.get(tid)
            if raw is not None:
                self.tracks[tid]["reid_emb"] = raw.copy()
            if tid in count_map:
                self.tracks[tid]["appearance_update_count"] = int(count_map[tid])

        return updated_ids

    def _append_missed_detections(self, updated_ids):
        """
        For tracks that were not updated in the current frame, append a
        frame-level record with has_detection=0 to keep history explicit.
        Stops after max_history_gap_frames without a real detection (last_frame).
        """
        for tid, tr in self.tracks.items():
            if tid in updated_ids:
                continue

            gap = self.frame_idx - tr["last_frame"]
            if (
                self.max_history_gap_frames is not None
                and gap > self.max_history_gap_frames
            ):
                continue

            tr["frames"].append(self.frame_idx)
            tr["bboxes"].append(None)
            tr["bbox_centers"].append(None)
            tr["bottom_centers"].append(None)
            tr["projected_bcenters"].append(None)
            tr["has_detection"].append(0)

    def _manage_track_states(self):
        lost_after = (
            self.max_history_gap_frames
            if self.max_history_gap_frames is not None
            else 30
        )
        for tr in self.tracks.values():
            frames_since_update = self.frame_idx - tr["last_frame"]
            if frames_since_update > lost_after:
                tr["state"] = "lost"
            elif tr["length"] < 5:
                tr["state"] = "pending"
            else:
                tr["state"] = "confirmed"

    def process_frame(
        self,
        frame,
        update_storage=True,
        detections_array=None,
        det_ms_override=None,
        embs_override=None,
        contact_uv_override=None,
    ):
        """
        One frame: detection → SCT → (optionally) local history in self.tracks.
        For multicamera, call from a global loop with synchronized frames.
        If frame is None, return empty array, do not increment frame index.
        """
        if frame is None:
            self.last_frame_ms = None
            return np.empty((0, TRACK_NCOLS), dtype=np.float32)

        t0 = time.perf_counter()
        if detections_array is not None:
            detections_array = np.asarray(detections_array, dtype=np.float32)
            if detections_array.ndim == 1 and detections_array.size > 0:
                detections_array = detections_array.reshape(1, -1)
            t1 = t0
        elif self.detection_store is not None:
            # MOT files use 1-based frame ids; first processed frame has frame_idx 0.
            detections_array = self.detection_store.get(self.frame_idx + 1)
            detections_array = Utils.postprocess_detections(detections_array)
            t1 = time.perf_counter()
        else:
            detector = self.detector if self.detector is not None else self._shared_detector
            detections, _ = detector.detect(frame)
            detections_array = (
                Utils.postprocess_detections(np.asarray(detections))
                if detections is not None and len(detections) > 0
                else np.empty((0, 6), dtype=np.float32)
            )
            t1 = time.perf_counter()
        detections_array = self._filter_detections_roi(detections_array)

        if (
            contact_uv_override is None
            and self.contact_point_model is not None
            and self.world_anchor == "contact_point"
            and frame is not None
            and detections_array is not None
            and len(detections_array) > 0
        ):
            contact_uv_override = batch_contact_point_uv(
                self.contact_point_model,
                [frame],
                [detections_array],
                max_batch_size=self.contact_batch_size,
                conf_thresh=self.contact_conf_thresh,
            )[0]

        tracks = self.tracker.update(detections_array, frame, embs=embs_override)
        tracks = self._filter_tracks_roi(tracks)
        tracks = enrich_tracks_world(
            tracks,
            self.homo,
            world_anchor=self.world_anchor,
            contact_uv_by_det=contact_uv_override,
        )
        t2 = time.perf_counter()

        if det_ms_override is not None:
            track_ms = (t2 - t1) * 1000.0
            self.last_frame_ms = {
                "det_ms": float(det_ms_override),
                "track_ms": track_ms,
                "sct_ms": float(det_ms_override) + track_ms,
            }
        else:
            self.last_frame_ms = sct_timing_ms(t0, t1, t2)

        if update_storage:
            updated_ids = self._update_tracks_storage(tracks)
            self._append_missed_detections(updated_ids)
            self._manage_track_states()

        self.frame_idx += 1
        return tracks

    def run(self, visualize=True, save=False, save_tracks=True, output_path="output.mp4"):
        from core.io.camera_manager import read_source_fps

        cap = cv2.VideoCapture(self.source)
        fps = read_source_fps(cap, default=10.0)
        writer = None
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                tracks = self.process_frame(frame, update_storage=True)

                frame_vis = self.visualizer.draw_tracks(
                    frame.copy(),
                    tracks,
                    cam_id=self.cam_id
                )

                writer = self.visualizer.visualize(
                    frames=[frame_vis],
                    visualize=visualize,
                    save_output=save,
                    output_path=output_path,
                    writer=writer,
                    fps=fps,
                )

                if visualize and cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            if cap.isOpened():
                cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
        
        if save_tracks:
            save_tracks_history_json(self.tracks)

class MultiCameraTrackingPipeline:
    """
    Orchestration of multicamera tracking:
    - reading frames from sources (CameraManager)
    - on each camera: detection + SCT (Sort, BotSort or DeepOcSort by tracker.type)
    - cross-camera association: cost matrix + Hungarian algorithm
    """

    def __init__(
        self,
        sources,
        tracker_config,
        model="yolov8m.pt",
        target_classes=None,
        detector_conf_thres=0.3,
        detector_device=None,
        detector_imgsz=960,
        homos=None,
        association_cost_threshold=0.35,
        association_reid_weight=0.5,
        geometry_max_distance=25.0,
        max_cross_cam_gap_frames=300,
        max_history_gap_frames=30,
        mapping_clear_after_lost_frames=None,
        use_temporal_decay=False,
        cam_ids=None,
        results_dir=None,
        roi_paths=None,
        detection_files=None,
        video_fps=10.0,
        share_reid_model=True,
        association_config=None,
        max_frames: int | None = None,
        world_anchor: str = "bottom_center",
        contact_point_config=None,
    ):
        self.sources = list(sources)
        self.video_fps = float(video_fps)
        self.max_frames = int(max_frames) if max_frames is not None else None
        self.camera_manager = CameraManager(
            sources=self.sources, default_fps=self.video_fps
        )
        # ``homos`` are **H_image_to_world** per camera.
        if homos is None:
            self.homos = [np.eye(3, dtype=np.float64) for _ in self.sources]
        else:
            self.homos = [np.asarray(h, dtype=np.float64) for h in homos]

        if cam_ids is None:
            self.cam_ids = list(range(len(self.sources)))
        else:
            self.cam_ids = [int(c) for c in cam_ids]
            if len(self.cam_ids) != len(self.sources):
                raise ValueError("cam_ids length must match sources length")

        if roi_paths is None:
            roi_paths = [None] * len(self.sources)
        elif len(roi_paths) != len(self.sources):
            raise ValueError("roi_paths length must match sources length")

        if detection_files is not None and len(detection_files) != len(self.sources):
            raise ValueError("detection_files length must match sources length")

        tracker_cfg = dict(tracker_config or {})
        self.appearance_update = normalize_appearance_mode(
            tracker_cfg.get("appearance_update", "aaf")
        )
        self.reid_accum_conf_thresh = tracker_cfg.get("reid_accum_conf_thresh")
        batch_size = tracker_cfg.get("reid_batch_size")
        self.reid_batch_size = int(batch_size) if batch_size is not None else None

        if target_classes is None:
            target_classes = [2, 3, 5, 7]

        self.shared_detector = None
        if detection_files is None:
            self.shared_detector = Detector(
                model=model,
                target_classes=target_classes,
                conf_thres=detector_conf_thres,
                device=detector_device,
                imgsz=detector_imgsz,
            )

        self.shared_reid_model = _maybe_build_shared_reid_model(
            tracker_config, share_reid_model=share_reid_model
        )
        self.world_anchor = str(world_anchor).lower().strip()
        contact_cfg = dict(contact_point_config or {})
        self.contact_point_model = _maybe_build_contact_point_model(contact_cfg)
        cp_batch_size = contact_cfg.get("batch_size")
        self.contact_batch_size = int(cp_batch_size) if cp_batch_size is not None else None
        self.contact_conf_thresh = float(contact_cfg.get("conf_thresh", 0.0))
        self._last_batch_reid_ms = None
        self._last_batch_contact_ms = None
        self.per_cam_pipelines = [
            SingleCameraTrackerPipeline(
                source=src,
                tracker_config=tracker_config,
                model=model,
                target_classes=target_classes,
                detector_conf_thres=detector_conf_thres,
                detector_device=detector_device,
                detector_imgsz=detector_imgsz,
                cam_id=self.cam_ids[i],
                homo=self.homos[i],
                max_history_gap_frames=max_history_gap_frames,
                roi_path=roi_paths[i],
                detection_file=(
                    detection_files[i] if detection_files is not None else None
                ),
                shared_reid_model=self.shared_reid_model,
                shared_detector=self.shared_detector,
                world_anchor=self.world_anchor,
                contact_point_model=self.contact_point_model,
                contact_batch_size=self.contact_batch_size,
                contact_conf_thresh=self.contact_conf_thresh,
            )
            for i, src in enumerate(self.sources)
        ]
        self.visualizer = Visualizer()

        self.frame_idx = 0
        if association_config is None:
            association_config = {
                "association_cost_threshold": association_cost_threshold,
                "geometry_max_distance": geometry_max_distance,
                "max_cross_cam_gap_frames": max_cross_cam_gap_frames,
                "video_fps": self.video_fps,
            }
        elif isinstance(association_config, dict):
            association_config = dict(association_config)
            association_config.setdefault("video_fps", self.video_fps)
        self.assoc_cfg = CrossCameraAssociationConfig.from_yaml(association_config)
        self.zone_map = None
        if self.assoc_cfg.zones_path:
            zones_path = Path(self.assoc_cfg.zones_path)
            if not zones_path.is_file():
                raise FileNotFoundError(f"zones_path not found: {zones_path}")
            self.zone_map = ZoneMap.from_yaml(zones_path)
        self._local_zones = LocalZoneTracker(self.zone_map)
        self.association_cost_threshold = self.assoc_cfg.reid_cost_threshold
        self.association_reid_weight = float(np.clip(association_reid_weight, 0.0, 1.0))
        self.max_cross_cam_gap_frames = self.assoc_cfg.max_cross_cam_gap_frames
        self.global_delete_after_frames = self.assoc_cfg.global_delete_after_frames
        self.mapping_clear_after_lost_frames = int(
            mapping_clear_after_lost_frames
            if mapping_clear_after_lost_frames is not None
            else max_history_gap_frames
        )
        self.use_temporal_decay = bool(use_temporal_decay)

        # global_id -> GlobalTrack with per-frame multicamera history.
        self.global_tracks = {}
        self._global_track_store = GlobalTrackStore(self.global_tracks)
        self._next_global_id = 1
        # (cam_index, local_track_id) -> global_id
        self.local_to_global = {}
        # number of consecutive frames a local_tid has been missing from SCT output
        self._local_absent_frames: dict[tuple[int, int], int] = {}

        self.results_dir = Path(results_dir) if results_dir else None
        self.mcmt_writer = (
            MCMTResultWriter(self.results_dir, self.cam_ids, self.homos)
            if self.results_dir is not None
            else None
        )

    @staticmethod
    def _detections_to_array(detections) -> np.ndarray:
        if detections is None or len(detections) == 0:
            return np.empty((0, 6), dtype=np.float32)
        return Utils.postprocess_detections(np.asarray(detections, dtype=np.float32))

    def _parallel_read_enabled(self) -> bool:
        return len(self.sources) > 1

    def _process_one_camera(self, args):
        cam_id, frame, detections_array, det_ms_override, embs_override, contact_uv_override = args
        return self.per_cam_pipelines[cam_id].process_frame(
            frame,
            update_storage=True,
            detections_array=detections_array,
            det_ms_override=det_ms_override,
            embs_override=embs_override,
            contact_uv_override=contact_uv_override,
        )

    def _process_frames_per_cam(
        self,
        frames,
        *,
        detections_per_cam=None,
        det_ms_per_cam=None,
        embs_per_cam=None,
        contact_uv_per_cam=None,
    ):
        n = len(frames)
        detections_per_cam = detections_per_cam or [None for _ in range(n)]
        embs_per_cam = embs_per_cam or [None for _ in range(n)]
        contact_uv_per_cam = contact_uv_per_cam or [None for _ in range(n)]
        tasks = [
            (
                cam_id,
                frames[cam_id],
                detections_per_cam[cam_id],
                det_ms_per_cam if frames[cam_id] is not None else None,
                embs_per_cam[cam_id],
                contact_uv_per_cam[cam_id],
            )
            for cam_id in range(n)
        ]

        if n <= 1:
            return [self._process_one_camera(task) for task in tasks]

        with ThreadPoolExecutor(max_workers=n) as pool:
            return list(pool.map(self._process_one_camera, tasks))

    def _step_sct(self, frames):
        """Batched YOLO + batched ReID + contact point, then parallel SCT per camera."""
        self._last_batch_reid_ms = None
        self._last_batch_contact_ms = None
        n = len(frames)

        if self.shared_detector is not None:
            t_det0 = time.perf_counter()
            batch_results = self.shared_detector.detect_batch(frames)
            batch_det_ms = (time.perf_counter() - t_det0) * 1000.0
            n_active = sum(1 for frame in frames if frame is not None)
            det_ms_per_cam = batch_det_ms / max(n_active, 1)
            detections_per_cam = [
                self._detections_to_array(detections)
                for detections, _ in batch_results
            ]
            for cam_id, dets in enumerate(detections_per_cam):
                detections_per_cam[cam_id] = self.per_cam_pipelines[
                    cam_id
                ]._filter_detections_roi(dets)
        else:
            det_ms_per_cam = None
            detections_per_cam = []
            for cam in self.per_cam_pipelines:
                if cam.detection_store is not None:
                    dets = self._detections_to_array(
                        cam.detection_store.get(self.frame_idx + 1)
                    )
                    detections_per_cam.append(cam._filter_detections_roi(dets))
                else:
                    detections_per_cam.append(None)

        embs_per_cam = [None for _ in range(n)]
        if self.shared_reid_model is not None:
            t_reid0 = time.perf_counter()
            embs_per_cam = batch_reid_features(
                self.shared_reid_model,
                frames,
                detections_per_cam,
                max_batch_size=self.reid_batch_size,
                conf_thresh=self.reid_accum_conf_thresh,
            )
            self._last_batch_reid_ms = (time.perf_counter() - t_reid0) * 1000.0

        contact_uv_per_cam = [None for _ in range(n)]
        if self.contact_point_model is not None and self.world_anchor == "contact_point":
            t_cp0 = time.perf_counter()
            contact_uv_per_cam = batch_contact_point_uv(
                self.contact_point_model,
                frames,
                detections_per_cam,
                max_batch_size=self.contact_batch_size,
                conf_thresh=self.contact_conf_thresh,
            )
            self._last_batch_contact_ms = (time.perf_counter() - t_cp0) * 1000.0

        return self._process_frames_per_cam(
            frames,
            detections_per_cam=detections_per_cam,
            det_ms_per_cam=det_ms_per_cam,
            embs_per_cam=embs_per_cam,
            contact_uv_per_cam=contact_uv_per_cam,
        )

    def _cam_appearance_maps(self, cam_id):
        tracker = self.per_cam_pipelines[cam_id].tracker
        match_map = (
            tracker.get_track_feature_map()
            if hasattr(tracker, "get_track_feature_map")
            else {}
        )
        raw_map = (
            tracker.get_track_appearance_raw_map()
            if hasattr(tracker, "get_track_appearance_raw_map")
            else {}
        )
        return match_map, raw_map

    def _world_point(self, cam_id, row):
        """World coords from enriched row (cols 8–9), else project if homography set."""
        wpt = world_point_from_row(row)
        if wpt is not None:
            return wpt
        H = self.homos[cam_id]
        if homography_valid(H):
            return project_bbox_bottom_center(H, row[0], row[1], row[2], row[3])
        return None

    @staticmethod
    def _global_last_frame(track) -> int | None:
        return track.last_frame

    def _assoc_cost(self, fvec, gid, query_cam, wpt, query_bbox=None, query_local_tid=None):
        """Association cost after hard gates and soft penalties."""
        gmeta = self.global_tracks[gid]
        zone_cam = int(self.cam_ids[query_cam])
        query_zone_entry = (
            self._local_zones.entry_zone((query_cam, query_local_tid))
            if query_local_tid is not None
            else None
        )
        return association_cost_for_match(
            self.assoc_cfg,
            query_cam,
            wpt,
            fvec,
            gmeta,
            self.frame_idx,
            appearance_update=self.appearance_update,
            query_bbox=query_bbox,
            zone_map=self.zone_map,
            zone_cam=zone_cam,
            query_zone_entry=query_zone_entry,
        )

    def _make_global_observation(self, row, local_tid, raw_feat, wpt) -> CamObservation:
        x1, y1, x2, y2 = map(float, row[:4])
        conf = float(row[5]) if len(row) > 5 else 0.0
        has_detection = int(row[7]) if len(row) > 7 else 1
        return CamObservation(
            local_tid=int(local_tid),
            bbox=(x1, y1, x2, y2),
            wpt=wpt,
            reid_raw=raw_feat,
            conf=conf,
            has_detection=has_detection,
        )

    def _update_local_state(self, gid, cam_id, local_tid, raw_feat, wpt, row=None):
        """Update appearance and per-camera world point for one local track."""
        if row is None:
            bbox = None
            conf = 0.0
            has_detection = 1
        else:
            bbox = tuple(float(v) for v in row[:4])
            conf = float(row[5]) if len(row) > 5 else 0.0
            has_detection = int(row[7]) if len(row) > 7 else 1
        obs = CamObservation(
            local_tid=int(local_tid),
            bbox=bbox,
            wpt=wpt,
            reid_raw=raw_feat,
            conf=conf,
            has_detection=has_detection,
        )
        self._global_track_store.append_observation(gid, self.frame_idx, cam_id, obs)

    def _refresh_active_cameras(self, per_cam_tracks):
        """Recompute active_cameras per global id; update last_seen on deactivation."""
        self._global_track_store.refresh_active(
            self.local_to_global,
            per_cam_tracks,
            zone_map=getattr(self, "zone_map", None),
            cam_ids=getattr(self, "cam_ids", None),
            local_zone_tracker=getattr(self, "_local_zones", None),
        )

    def _update_local_zone_states(self, per_cam_tracks):
        if self.zone_map is None or self.zone_map.mode != "tracklet":
            return
        for cam_idx, tracks in enumerate(per_cam_tracks):
            if tracks is None or len(tracks) == 0:
                continue
            real_cam = int(self.cam_ids[cam_idx])
            for row in tracks:
                local_tid = int(row[4])
                bbox = tuple(float(v) for v in row[:4])
                self._local_zones.update((cam_idx, local_tid), real_cam, bbox)

    def _valid_candidates(self, query_cam, candidates):
        """Filter global-id candidates (velocity, cam graph, etc.). Stub in v1."""
        return candidates

    def _candidate_globals(self, cam_id):
        """Global hypotheses not too old (same-cam allowed under tiered geometry)."""
        out = []
        if self.assoc_cfg.temporal_mode == "strict":
            max_gap = self.assoc_cfg.max_cross_cam_gap_frames
        else:
            max_gap = self.assoc_cfg.global_delete_after_frames
        for gid, meta in self.global_tracks.items():
            if meta.state == "lost":
                continue
            last_f = self._global_last_frame(meta)
            if last_f is None:
                continue
            if self.frame_idx - last_f > max_gap:
                continue
            out.append(gid)
        return out

    def _merge_global_ids(self, gid_keep: int, gid_drop: int) -> None:
        if gid_keep == gid_drop or gid_drop not in self.global_tracks:
            return
        self._global_track_store.merge(gid_keep, gid_drop)
        for key, gid in list(self.local_to_global.items()):
            if gid == gid_drop:
                self.local_to_global[key] = gid_keep

    def _apply_same_frame_linking(self, per_cam_tracks) -> None:
        if not self.assoc_cfg.same_frame_linking:
            return

        detections: list[tuple[int, int, tuple[float, float] | None, object, object]] = []
        row_by_key: dict[tuple[int, int], object] = {}
        raw_by_key: dict[tuple[int, int], object] = {}
        for cam_id, tracks in enumerate(per_cam_tracks):
            if tracks is None or len(tracks) == 0:
                continue
            match_map, raw_map = self._cam_appearance_maps(cam_id)
            for row in tracks:
                local_tid = int(row[4])
                key = (cam_id, local_tid)
                wpt = self._world_point(cam_id, row)
                raw_feat = raw_map.get(local_tid)
                detections.append((cam_id, local_tid, wpt, raw_feat))
                row_by_key[key] = row
                raw_by_key[key] = raw_feat

        if len(detections) < 2:
            return

        links = find_same_frame_links(
            detections,
            t_min_m=self.assoc_cfg.geometry_t_min_m,
            reid_threshold=self.assoc_cfg.reid_strong_reject_threshold,
            metric=self.assoc_cfg.geometry_distance_metric,
            appearance_mode=self.appearance_update,
        )
        if not links:
            return

        keys = [(cam, tid) for cam, tid, _, _ in detections]
        key_to_idx = {k: i for i, k in enumerate(keys)}
        uf = UnionFind(len(keys))
        for key_a, key_b in links:
            uf.union(key_to_idx[key_a], key_to_idx[key_b])

        for comp_indices in uf.components():
            if len(comp_indices) < 2:
                continue
            comp_keys = [keys[i] for i in comp_indices]
            existing_gids = {
                self.local_to_global[k] for k in comp_keys if k in self.local_to_global
            }
            if len(existing_gids) > 1:
                gid_keep = min(existing_gids)
                for gid_drop in existing_gids:
                    if gid_drop != gid_keep:
                        self._merge_global_ids(gid_keep, gid_drop)
            elif len(existing_gids) == 1:
                gid_keep = next(iter(existing_gids))
            else:
                first_key = comp_keys[0]
                cam_id, local_tid = first_key
                match_map, raw_map = self._cam_appearance_maps(cam_id)
                row = row_by_key[first_key]
                self._new_global(
                    cam_id,
                    local_tid,
                    match_map.get(local_tid),
                    raw_map.get(local_tid),
                    detections[key_to_idx[first_key]][2],
                    row,
                )
                gid_keep = self.local_to_global[first_key]

            for key in comp_keys:
                if key in self.local_to_global:
                    continue
                cam_id, local_tid = key
                row = row_by_key[key]
                self.local_to_global[key] = gid_keep
                self._local_absent_frames.pop(key, None)
                self._update_local_state(
                    gid_keep,
                    cam_id,
                    local_tid,
                    raw_by_key[key],
                    detections[key_to_idx[key]][2],
                    row,
                )

    def _associate_cross_camera(self, per_cam_tracks):
        """
        Unmatched (cam, local_tid) try to attach to existing global_id
        through linear_sum_assignment; otherwise create a new global_id.
        """
        self._refresh_active_cameras(per_cam_tracks)

        COST_INF = 1e9
        taken_per_cam: dict[int, set] = {}

        unmatched = []
        for cam_id, tracks in enumerate(per_cam_tracks):
            if tracks is None or len(tracks) == 0:
                continue
            for row in tracks:
                local_tid = int(row[4])
                key = (cam_id, local_tid)
                wpt = self._world_point(cam_id, row)
                if key in self.local_to_global:
                    gid = self.local_to_global[key]
                    taken = taken_per_cam.setdefault(cam_id, set())
                    if gid in taken:
                        # Two active local_tid on the same camera with the same global_id.
                        match_map, raw_map = self._cam_appearance_maps(cam_id)
                        self._new_global(
                            cam_id,
                            local_tid,
                            match_map.get(local_tid),
                            raw_map.get(local_tid),
                            wpt,
                            row,
                        )
                        gid = self.local_to_global[key]
                    taken.add(gid)
                    match_map, raw_map = self._cam_appearance_maps(cam_id)
                    self._update_local_state(
                        gid,
                        cam_id,
                        local_tid,
                        raw_map.get(local_tid),
                        wpt,
                        row,
                    )
                    continue
                match_map, raw_map = self._cam_appearance_maps(cam_id)
                fvec = match_map.get(local_tid)
                unmatched.append((cam_id, local_tid, fvec, raw_map.get(local_tid), wpt, row))

        if not unmatched:
            self._resolve_per_cam_gid_conflicts(per_cam_tracks)
            self._refresh_active_cameras(per_cam_tracks)
            return

        raw_candidates = self._candidate_globals_for_unmatched(
            {c for c, _, _, _, _, _ in unmatched}
        )
        if not raw_candidates:
            for cam_id, local_tid, fvec, raw_feat, wpt, row in unmatched:
                self._new_global(cam_id, local_tid, fvec, raw_feat, wpt, row)
            self._resolve_per_cam_gid_conflicts(per_cam_tracks)
            self._refresh_active_cameras(per_cam_tracks)
            return

        n_u = len(unmatched)
        col_entries: dict[int, list[tuple[int, int]]] = {}
        for i, (cam_id, local_tid, fvec, _raw_feat, wpt, row) in enumerate(unmatched):
            cam_taken = taken_per_cam.get(cam_id, set())
            candidates = self._valid_candidates(cam_id, raw_candidates)
            query_bbox = tuple(float(v) for v in row[:4]) if row is not None else None
            for gid in candidates:
                if gid in cam_taken:
                    continue
                cost = self._assoc_cost(
                    fvec,
                    gid,
                    cam_id,
                    wpt,
                    query_bbox,
                    query_local_tid=local_tid,
                )
                if cost is None or not np.isfinite(cost):
                    continue
                col_entries.setdefault(gid, []).append((i, float(cost)))

        col_gid = sorted(col_entries.keys())
        n_g = len(col_gid)
        if n_g == 0:
            for cam_id, local_tid, fvec, raw_feat, wpt, row in unmatched:
                self._new_global(cam_id, local_tid, fvec, raw_feat, wpt, row)
            self._resolve_per_cam_gid_conflicts(per_cam_tracks)
            self._refresh_active_cameras(per_cam_tracks)
            return

        C = np.full((n_u, n_g), COST_INF, dtype=np.float64)
        for j, gid in enumerate(col_gid):
            for i, cost in col_entries[gid]:
                C[i, j] = cost

        s = max(n_u, n_g)
        P = np.full((s, s), COST_INF, dtype=np.float64)
        P[:n_u, :n_g] = C
        bad = ~np.isfinite(P)
        if bad.any():
            P = P.copy()
            P[bad] = COST_INF
        ri, ci = linear_sum_assignment(P)

        used_local = set()
        used_global = set()
        for r, c in zip(ri, ci):
            if r >= n_u or c >= n_g:
                continue
            if P[r, c] >= COST_INF / 2:
                continue
            if P[r, c] > self.assoc_cfg.reid_cost_threshold:
                continue
            cam_id, local_tid, fvec, raw_feat, wpt, row = unmatched[r]
            gid = col_gid[c]
            if gid in taken_per_cam.get(cam_id, set()):
                continue
            self.local_to_global[(cam_id, local_tid)] = gid
            taken_per_cam.setdefault(cam_id, set()).add(gid)
            used_local.add(r)
            used_global.add(c)
            self._update_local_state(gid, cam_id, local_tid, raw_feat, wpt, row)

        for i, (cam_id, local_tid, fvec, raw_feat, wpt, row) in enumerate(unmatched):
            if i in used_local:
                continue
            self._new_global(cam_id, local_tid, fvec, raw_feat, wpt, row)

        self._resolve_per_cam_gid_conflicts(per_cam_tracks)
        self._refresh_active_cameras(per_cam_tracks)

    def _resolve_per_cam_gid_conflicts(self, per_cam_tracks):
        """Per camera per frame: one global_id maps to one local_tid (others get a new gid)."""
        for cam_id, tracks in enumerate(per_cam_tracks):
            if tracks is None or len(tracks) == 0:
                continue
            by_gid: dict[int, list] = {}
            for row in tracks:
                local_tid = int(row[4])
                gid = self.local_to_global.get((cam_id, local_tid))
                if gid is None:
                    continue
                conf = float(row[5]) if len(row) > 5 else 0.0
                by_gid.setdefault(gid, []).append((local_tid, conf))

            for gid, items in by_gid.items():
                if len(items) <= 1:
                    continue
                items.sort(key=lambda x: x[1], reverse=True)
                match_map, raw_map = self._cam_appearance_maps(cam_id)
                for local_tid, _conf in items[1:]:
                    wpt = None
                    for row in tracks:
                        if int(row[4]) == local_tid:
                            wpt = self._world_point(cam_id, row)
                            break
                    self._new_global(
                        cam_id,
                        local_tid,
                        match_map.get(local_tid),
                        raw_map.get(local_tid),
                        wpt,
                        row,
                    )

    def _candidate_globals_for_unmatched(self, cam_ids):
        """Union of candidates across all cameras that have unmatched tracks."""
        seen = set()
        for cam_id in cam_ids:
            for gid in self._candidate_globals(cam_id):
                seen.add(gid)
        return sorted(seen)

    def _new_global(self, cam_id, local_tid, fvec, raw_feat=None, wpt=None, row=None):
        gid = self._next_global_id
        self._next_global_id += 1
        key = (cam_id, local_tid)
        self.local_to_global[key] = gid
        self._local_absent_frames.pop(key, None)
        stored = raw_feat if raw_feat is not None else fvec
        if row is None:
            obs = CamObservation(
                local_tid=int(local_tid),
                bbox=None,
                wpt=wpt,
                reid_raw=stored,
                conf=0.0,
                has_detection=1,
            )
        else:
            obs = self._make_global_observation(row, local_tid, stored, wpt)
        self._global_track_store.create(gid, self.frame_idx, cam_id, obs)

    def _clear_local_mapping(self, key: tuple[int, int]) -> None:
        gid = self.local_to_global.get(key)
        self.local_to_global.pop(key, None)
        self._local_absent_frames.pop(key, None)
        self._local_zones.pop(key)
        if gid is not None and gid in self.global_tracks:
            self._global_track_store.remove_local_appearance_key(gid, key[0], key[1])

    def _prune_stale_local_mappings(self, per_cam_tracks):
        """Drop (cam, local_tid)->gid mappings based on the SCT track state.

        * ``state == "lost"`` and >= N frames passed since ``last_frame`` — the
          object on this camera is considered finished, the mapping is removed.
        * ``local_tid`` reappeared after being absent — likely id reuse by the
          tracker, so the mapping is reset until a new association.
        * N = ``mapping_clear_after_lost_frames`` (defaults to the SCT
          ``max_history_gap_frames``, usually 30).
        """
        n_clear = self.mapping_clear_after_lost_frames

        for cam_id, tracks in enumerate(per_cam_tracks):
            active = (
                {int(row[4]) for row in tracks}
                if tracks is not None and len(tracks) > 0
                else set()
            )
            sct = self.per_cam_pipelines[cam_id]
            cam_keys = [k for k in list(self.local_to_global) if k[0] == cam_id]

            for key in cam_keys:
                local_tid = key[1]
                tr = sct.tracks.get(local_tid)

                if local_tid in active:
                    if self._local_absent_frames.pop(key, 0) > 0:
                        self._clear_local_mapping(key)
                    continue

                absent = self._local_absent_frames.get(key, 0) + 1
                self._local_absent_frames[key] = absent

                if tr is None:
                    self._clear_local_mapping(key)
                    continue

                gap = sct.frame_idx - int(tr["last_frame"])
                if tr.get("state") == "lost" and gap >= n_clear:
                    self._clear_local_mapping(key)
                elif absent >= n_clear:
                    self._clear_local_mapping(key)

    def global_id_for(self, cam_id, local_tid):
        return self.local_to_global.get((cam_id, int(local_tid)))

    def _finalize_global_tracks_frame(self, per_cam_tracks) -> None:
        deleted = self._global_track_store.finalize_frame(
            self.frame_idx,
            per_cam_tracks,
            self.local_to_global,
            lost_after=self.mapping_clear_after_lost_frames,
            delete_after=self.global_delete_after_frames,
            zone_map=self.zone_map,
            cam_ids=self.cam_ids,
            local_zone_tracker=self._local_zones,
        )
        if deleted:
            deleted_set = set(deleted)
            for key, gid in list(self.local_to_global.items()):
                if gid in deleted_set:
                    self.local_to_global.pop(key, None)
                    self._local_absent_frames.pop(key, None)
                    self._local_zones.pop(key)

    def run(
        self,
        visualize=True,
        save=False,
        output_path="multicam_output.mp4",
        save_video_dir=None,
    ):
        """Run MCMT loop.

        * ``save_video_dir`` set — one MP4 per camera with **global** ids
          (``c006.mp4``, …) under that directory.
        * ``save`` without ``save_video_dir`` — legacy single stacked video at
          ``output_path`` (all cameras vertically in one file).
        """
        writer = None
        per_cam_writers: dict[int, cv2.VideoWriter] = {}
        video_dir = Path(save_video_dir) if save_video_dir else None
        if save and video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)

        fps = FpsCollector(num_cameras=len(self.sources))

        try:
            while True:
                t_frame = fps.begin_frame()
                frames = self.camera_manager.read_frames(
                    parallel=self._parallel_read_enabled()
                )
                if all(f is None for f in frames):
                    break

                per_cam_tracks = self._step_sct(frames)
                t_after_sct = time.perf_counter()
                self._update_local_zone_states(per_cam_tracks)
                self._prune_stale_local_mappings(per_cam_tracks)
                self._apply_same_frame_linking(per_cam_tracks)
                self._associate_cross_camera(per_cam_tracks)
                self._finalize_global_tracks_frame(per_cam_tracks)
                t_after_mcmt = time.perf_counter()

                if self.mcmt_writer is not None:
                    for cam_idx, tracks in enumerate(per_cam_tracks):
                        if frames[cam_idx] is None:
                            continue
                        self.mcmt_writer.add_frame(
                            cam_index=cam_idx,
                            frame_id=self.frame_idx + 1,
                            tracks=tracks,
                            local_to_global=self.local_to_global,
                        )

                rendered = []
                for cam_idx, frame in enumerate(frames):
                    if frame is None:
                        continue
                    tracks = per_cam_tracks[cam_idx]
                    ids_dict = {}
                    if len(tracks) > 0:
                        for row in tracks:
                            lid = int(row[4])
                            gid = self.global_id_for(cam_idx, lid)
                            if gid is not None:
                                ids_dict[lid] = gid
                    cam_label = self.cam_ids[cam_idx]
                    vis = self.visualizer.draw_tracks(
                        frame.copy(),
                        tracks,
                        cam_id=cam_label,
                        ids_dict=ids_dict,
                    )

                    if save and video_dir is not None:
                        if cam_idx not in per_cam_writers:
                            out_mp4 = video_dir / f"c{cam_label:03d}.mp4"
                            per_cam_writers[cam_idx] = (
                                self.visualizer.create_video_writer(
                                    out_mp4,
                                    vis,
                                    fps=self.camera_manager.fps_list[cam_idx],
                                )
                            )
                        per_cam_writers[cam_idx].write(vis)
                    elif save:
                        rendered.append(vis)

                    if visualize:
                        cv2.imshow(f"Cam c{cam_label:03d}", vis)

                if rendered and save and video_dir is None:
                    writer = self.visualizer.visualize(
                        frames=rendered,
                        visualize=False,
                        save_output=True,
                        output_path=output_path,
                        writer=writer,
                        fps=self.video_fps,
                    )

                self.frame_idx += 1
                if self.max_frames is not None and self.frame_idx >= self.max_frames:
                    break
                fps.end_frame(
                    t_frame,
                    t_after_sct,
                    t_after_mcmt,
                    frames,
                    self.per_cam_pipelines,
                    batch_reid_ms=self._last_batch_reid_ms,
                )
                if visualize and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.camera_manager.release()
            for w in per_cam_writers.values():
                w.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            if video_dir is not None and per_cam_writers:
                print(f"[MCMT] Per-cam videos (global ids): {video_dir}")
            if self.mcmt_writer is not None:
                paths = self.mcmt_writer.finalize()
                print(f"[MCMT] AICity track file: {paths['aicity']}")
                print(f"[MCMT] Per-cam (global ids) dir: {paths['per_cam'][self.cam_ids[0]].parent}")
                print(f"[MCMT] Per-cam (local ids) dir:  {paths['per_cam_local'][self.cam_ids[0]].parent}")
            fps.save(self.results_dir)

def load_pipeline_config(config_path):
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
