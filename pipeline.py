import numpy as np
import cv2
import json
from pathlib import Path
import yaml

from scipy.optimize import linear_sum_assignment

from core.detector.detector import Detector
from core.geometry.homography_transformer import HomographyTransformer
from core.io.camera_manager import CameraManager
from core.io.mcmt_writer import MCMTResultWriter
from core.io.mot_detections import MotDetectionStore
from core.io.roi import ROIFilter
from core.mot.tracker.bot_sort_tracker import BotSortTracker
from core.mot.tracker.deepocsort_tracker import DeepOcSortTracker
from core.utils.utilities import Utils
from core.visualization.visualizer import Visualizer


def _create_tracker(tracker_config):
    """Build BotSort or DeepOcSort tracker from YAML ``tracker`` dict."""
    cfg = dict(tracker_config or {})
    tracker_type = str(cfg.pop("type", "botsort")).lower().strip()
    if tracker_type in ("deepocsort", "deep_ocsort"):
        for k in ("reid_weights", "device", "half", "use_default_reid"):
            cfg.pop(k, None)
        return DeepOcSortTracker(**cfg)
    return BotSortTracker(**cfg)


class SingleCameraTrackerPipeline:
    def __init__(
        self,
        source,
        tracker_config,
        model="yolov8m.pt",
        target_classes=None,
        detector_conf_thres=0.3,
        detector_device=None,
        cam_id=0,
        homo=np.eye(3, dtype=np.float64),
        max_history_gap_frames=30,
        roi_path=None,
        detection_file=None,
    ):
        self.source = source
        self.tracker = _create_tracker(tracker_config)
        self.detection_store = None
        if detection_file is not None:
            self.detection_store = MotDetectionStore(
                detection_file, conf_thres=detector_conf_thres
            )
            self.detector = None
        else:
            if target_classes is None:
                target_classes = [2, 3, 5, 7]
            self.detector = Detector(
                model=model,
                target_classes=target_classes,
                conf_thres=detector_conf_thres,
                device=detector_device,
            )
        self.visualizer = Visualizer()
        self.tracks = {}
        self.frame_idx = 0
        self.cam_id = cam_id
        # ``homo`` is **H_image_to_world** (see ``core.io.calibration``).
        self.homo = np.asarray(homo, dtype=np.float64)
        self.homography_transformer = HomographyTransformer()
        # After this many frames without a detection for a track, stop appending
        # placeholder rows (has_detection=0). None = keep appending until video end.
        self.max_history_gap_frames = max_history_gap_frames
        self.roi_filter = ROIFilter.from_spec(roi_path) if roi_path is not None else None

    def _filter_detections_roi(self, dets: np.ndarray) -> np.ndarray:
        if self.roi_filter is None or dets is None or len(dets) == 0:
            return dets
        return self.roi_filter.filter_xyxy_array(dets)

    def _filter_tracks_roi(self, tracks: np.ndarray) -> np.ndarray:
        if self.roi_filter is None or tracks is None or len(tracks) == 0:
            return tracks
        return self.roi_filter.filter_xyxy_array(tracks)

    def _update_tracks_storage(self, tracks):
        updated_ids = set()
        for t in tracks:
            x1, y1, x2, y2, tid, conf, det_idx, has_detection = t
            tid = int(tid)
            updated_ids.add(tid)


            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            bcx = cx
            bcy = float(y2)

            projected_bc = self.homography_transformer.apply_homo_to_point(
                [bcx, bcy], self.homo
            )

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
                    "reid_emb": [],
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

    def process_frame(self, frame, update_storage=True):
        """
        Один кадр: детекция → SCT → (опционально) локальная история в self.tracks.
        Для мультикамеры вызывайте из общего цикла с синхронными кадрами.
        При frame is None возвращает пустой массив, индекс кадра не увеличивает.
        """
        if frame is None:
            return np.empty((0, 8), dtype=np.float32)

        if self.detection_store is not None:
            # MOT files use 1-based frame ids; first processed frame has frame_idx 0.
            detections_array = self.detection_store.get(self.frame_idx + 1)
            detections_array = Utils.filter_detections(detections_array)
        else:
            detections, _ = self.detector.detect(frame)
            detections_array = (
                Utils.filter_detections(np.asarray(detections))
                if detections is not None and len(detections) > 0
                else np.empty((0, 6), dtype=np.float32)
            )
        detections_array = self._filter_detections_roi(detections_array)

        tracks = self.tracker.update(detections_array, frame)
        tracks = self._filter_tracks_roi(tracks)

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
            self._save_tracks_to_json()

    def _save_tracks_to_json(self, output_path="tracks_history.json", indent=2):
        """
        Save accumulated track history to a JSON file.
        """
        def _to_jsonable(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {str(k): _to_jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_to_jsonable(v) for v in value]
            return value

        serializable_tracks = {
            str(int(track_id)): _to_jsonable(track_data)
            for track_id, track_data in self.tracks.items()
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_tracks, f, ensure_ascii=False, indent=indent)

class MultiCameraTrackingPipeline:
    """
    Оркестрация мультикамерного трекинга:
    - чтение кадров из sources (CameraManager)
    - на каждой камере: детекция + SCT (BotSort или DeepOCSORT по tracker.type)
    - межкамерная ассоциация: матрица стоимостей + венгерский алгоритм
    """

    def __init__(
        self,
        sources,
        tracker_config,
        model="yolov8m.pt",
        target_classes=None,
        detector_conf_thres=0.3,
        detector_device=None,
        homos=None,
        association_cost_threshold=0.35,
        max_cross_cam_gap_frames=300,
        max_history_gap_frames=30,
        mapping_clear_after_lost_frames=None,
        cam_ids=None,
        results_dir=None,
        roi_paths=None,
        detection_files=None,
        video_fps=10.0,
    ):
        self.sources = list(sources)
        self.video_fps = float(video_fps)
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

        self.per_cam_pipelines = [
            SingleCameraTrackerPipeline(
                source=src,
                tracker_config=tracker_config,
                model=model,
                target_classes=target_classes,
                detector_conf_thres=detector_conf_thres,
                detector_device=detector_device,
                cam_id=self.cam_ids[i],
                homo=self.homos[i],
                max_history_gap_frames=max_history_gap_frames,
                roi_path=roi_paths[i],
                detection_file=(
                    detection_files[i] if detection_files is not None else None
                ),
            )
            for i, src in enumerate(self.sources)
        ]
        self.visualizer = Visualizer()

        self.frame_idx = 0
        self.association_cost_threshold = float(association_cost_threshold)
        self.max_cross_cam_gap_frames = int(max_cross_cam_gap_frames)
        self.mapping_clear_after_lost_frames = int(
            mapping_clear_after_lost_frames
            if mapping_clear_after_lost_frames is not None
            else max_history_gap_frames
        )

        # global_id -> {smooth_feat, last_cam, last_frame}
        self.global_tracks = {}
        self._next_global_id = 1
        # (cam_index, local_track_id) -> global_id
        self.local_to_global = {}
        # сколько кадров подряд local_tid не было в выходе SCT на этом кадре
        self._local_absent_frames: dict[tuple[int, int], int] = {}

        self.results_dir = Path(results_dir) if results_dir else None
        self.mcmt_writer = (
            MCMTResultWriter(self.results_dir, self.cam_ids, self.homos)
            if self.results_dir is not None
            else None
        )

    def _step_sct(self, frames):
        """На каждой камере — тот же путь, что в SingleCameraTrackerPipeline.process_frame."""
        per_cam_tracks = []
        for cam_id, frame in enumerate(frames):
            tracks = self.per_cam_pipelines[cam_id].process_frame(
                frame, update_storage=True
            )
            per_cam_tracks.append(tracks)
        return per_cam_tracks

    def _emb_dist(self, a, b):
        """Косинусное расстояние для L2-нормированных векторов."""
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        if a.size == 0 or b.size == 0:
            return 1e9
        return float(1.0 - np.clip(np.dot(a, b), -1.0, 1.0))

    def _candidate_globals(self, cam_id):
        """Глобальные гипотезы с другой камеры и не слишком старые."""
        out = []
        for gid, meta in self.global_tracks.items():
            if meta["last_cam"] == cam_id:
                continue
            if self.frame_idx - meta["last_frame"] > self.max_cross_cam_gap_frames:
                continue
            out.append(gid)
        return out

    def _associate_cross_camera(self, per_cam_tracks):
        """
        Несопоставленные (cam, local_tid) пытаемся привязать к существующим global_id
        через linear_sum_assignment; иначе создаём новый global_id.
        """
        COST_INF = 1e9

        # gid, уже занятые активным локальным треком на этой камере в этом кадре.
        # Нужно, чтобы венгерка не назначила тот же gid второму local_tid на той же
        # камере → иначе MCMT пишет один global_id с несколькими bbox в кадре.
        taken_per_cam: dict[int, set] = {}

        unmatched = []
        for cam_id, tracks in enumerate(per_cam_tracks):
            if tracks is None or len(tracks) == 0:
                continue
            for row in tracks:
                local_tid = int(row[4])
                key = (cam_id, local_tid)
                if key in self.local_to_global:
                    gid = self.local_to_global[key]
                    taken = taken_per_cam.setdefault(cam_id, set())
                    if gid in taken:
                        # Два активных local_tid на одной камере с одним global_id.
                        feats = self.per_cam_pipelines[cam_id].tracker.get_track_feature_map()
                        self._new_global(cam_id, local_tid, feats.get(local_tid))
                        gid = self.local_to_global[key]
                    taken.add(gid)
                    feats = self.per_cam_pipelines[cam_id].tracker.get_track_feature_map()
                    fvec = feats.get(local_tid)
                    g = self.global_tracks[gid]
                    if fvec is not None:
                        if g["smooth_feat"] is None:
                            g["smooth_feat"] = np.asarray(fvec, dtype=np.float32).copy()
                        else:
                            alpha = 0.9
                            g["smooth_feat"] = (
                                alpha * g["smooth_feat"]
                                + (1.0 - alpha) * np.asarray(fvec, dtype=np.float32)
                            )
                            g["smooth_feat"] /= (
                                np.linalg.norm(g["smooth_feat"]) + 1e-12
                            )
                    g["last_cam"] = cam_id
                    g["last_frame"] = self.frame_idx
                    continue
                feats = self.per_cam_pipelines[cam_id].tracker.get_track_feature_map()
                fvec = feats.get(local_tid)
                unmatched.append((cam_id, local_tid, fvec))

        if not unmatched:
            self._resolve_per_cam_gid_conflicts(per_cam_tracks)
            return

        candidates = self._candidate_globals_for_unmatched({c for c, _, _ in unmatched})
        if not candidates:
            for cam_id, local_tid, _f in unmatched:
                self._new_global(cam_id, local_tid, _f)
            self._resolve_per_cam_gid_conflicts(per_cam_tracks)
            return

        n_u = len(unmatched)
        n_g = len(candidates)
        C = np.full((n_u, n_g), COST_INF, dtype=np.float64)
        for i, (cam_id, local_tid, fvec) in enumerate(unmatched):
            cam_taken = taken_per_cam.get(cam_id, set())
            for j, gid in enumerate(candidates):
                # Запрещаем уже занятые на этой камере global_id.
                if gid in cam_taken:
                    continue
                gfeat = self.global_tracks[gid]["smooth_feat"]
                if fvec is None or gfeat is None:
                    continue
                C[i, j] = self._emb_dist(fvec, gfeat)

        s = max(n_u, n_g)
        P = np.full((s, s), COST_INF, dtype=np.float64)
        P[:n_u, :n_g] = C
        ri, ci = linear_sum_assignment(P)

        used_local = set()
        used_global = set()
        for r, c in zip(ri, ci):
            if r >= n_u or c >= n_g:
                continue
            if P[r, c] >= COST_INF / 2:
                continue
            if P[r, c] > self.association_cost_threshold:
                continue
            cam_id, local_tid, fvec = unmatched[r]
            gid = candidates[c]
            if gid in taken_per_cam.get(cam_id, set()):
                continue
            self.local_to_global[(cam_id, local_tid)] = gid
            taken_per_cam.setdefault(cam_id, set()).add(gid)
            used_local.add(r)
            used_global.add(c)
            g = self.global_tracks[gid]
            if fvec is not None:
                if g["smooth_feat"] is None:
                    g["smooth_feat"] = np.asarray(fvec, dtype=np.float32).copy()
                else:
                    alpha = 0.9
                    g["smooth_feat"] = (
                        alpha * g["smooth_feat"]
                        + (1.0 - alpha) * np.asarray(fvec, dtype=np.float32)
                    )
                    g["smooth_feat"] /= np.linalg.norm(g["smooth_feat"]) + 1e-12
            g["last_cam"] = cam_id
            g["last_frame"] = self.frame_idx

        for i, (cam_id, local_tid, fvec) in enumerate(unmatched):
            if i in used_local:
                continue
            self._new_global(cam_id, local_tid, fvec)

        self._resolve_per_cam_gid_conflicts(per_cam_tracks)

    def _resolve_per_cam_gid_conflicts(self, per_cam_tracks):
        """На камере в кадре: один global_id — один local_tid (остальным — новый gid)."""
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
                feats = self.per_cam_pipelines[cam_id].tracker.get_track_feature_map()
                for local_tid, _conf in items[1:]:
                    self._new_global(cam_id, local_tid, feats.get(local_tid))

    def _candidate_globals_for_unmatched(self, cam_ids):
        """Объединение кандидатов по всем камерам, с которых есть unmatched."""
        seen = set()
        for cam_id in cam_ids:
            for gid in self._candidate_globals(cam_id):
                seen.add(gid)
        return sorted(seen)

    def _new_global(self, cam_id, local_tid, fvec):
        gid = self._next_global_id
        self._next_global_id += 1
        key = (cam_id, local_tid)
        self.local_to_global[key] = gid
        self._local_absent_frames.pop(key, None)
        feat = None
        if fvec is not None:
            feat = np.asarray(fvec, dtype=np.float32).copy()
            feat /= np.linalg.norm(feat) + 1e-12
        self.global_tracks[gid] = {
            "smooth_feat": feat,
            "last_cam": cam_id,
            "last_frame": self.frame_idx,
        }

    def _clear_local_mapping(self, key: tuple[int, int]) -> None:
        self.local_to_global.pop(key, None)
        self._local_absent_frames.pop(key, None)

    def _prune_stale_local_mappings(self, per_cam_tracks):
        """Снимаем (cam, local_tid)→gid по состоянию SCT-трека.

        * ``state == "lost"`` и с ``last_frame`` прошло >= N кадров — объект
          на этой камере считаем завершённым, маппинг удаляем.
        * ``local_tid`` снова появился после пропажи — вероятное переиспользование
          id трекером, маппинг сбрасываем до новой ассоциации.
        * N = ``mapping_clear_after_lost_frames`` (по умолчанию как
          ``max_history_gap_frames`` в SCT, обычно 30).
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

        try:
            while True:
                frames = self.camera_manager.read_frames()
                if all(f is None for f in frames):
                    break

                per_cam_tracks = self._step_sct(frames)
                self._prune_stale_local_mappings(per_cam_tracks)
                self._associate_cross_camera(per_cam_tracks)

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

def load_pipeline_config(config_path):
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
