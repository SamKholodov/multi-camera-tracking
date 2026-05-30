"""Pipeline FPS sampling and reporting for MCMT runs."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Optional

FPS_WARMUP_FRAMES = 10


def sct_timing_ms(t_start: float, t_after_det: float, t_after_track: float) -> dict[str, float]:
    """Build per-camera SCT timing dict from ``perf_counter`` timestamps."""
    return {
        "det_ms": (t_after_det - t_start) * 1000.0,
        "track_ms": (t_after_track - t_after_det) * 1000.0,
        "sct_ms": (t_after_track - t_start) * 1000.0,
    }


def build_fps_report(
    frame_times_sec: Iterable[float],
    num_cameras: int,
    det_ms_per_frame: Optional[Iterable[float]] = None,
    track_ms_per_frame: Optional[Iterable[float]] = None,
    mcmt_ms_per_frame: Optional[Iterable[float]] = None,
    warmup_frames: int = FPS_WARMUP_FRAMES,
) -> Optional[dict[str, Any]]:
    times = list(frame_times_sec)
    if not times:
        return None

    total = float(sum(times))
    n = len(times)
    report: dict[str, Any] = {
        "warmup_frames": warmup_frames,
        "frames_timed": n,
        "total_sec": round(total, 3),
        "pipeline_fps": round(n / total, 3) if total > 0 else 0.0,
        "ms_per_sync_frame": round(1000.0 * total / n, 2),
        "ms_per_camera_frame": round(1000.0 * total / n / max(num_cameras, 1), 2),
        "num_cameras": num_cameras,
    }

    if det_ms_per_frame:
        det = list(det_ms_per_frame)
        report["avg_det_ms_per_cam"] = round(float(sum(det)) / len(det), 2)
    if track_ms_per_frame:
        track = list(track_ms_per_frame)
        report["avg_track_ms_per_cam"] = round(float(sum(track)) / len(track), 2)
    if mcmt_ms_per_frame:
        mcmt = list(mcmt_ms_per_frame)
        report["avg_mcmt_ms"] = round(float(sum(mcmt)) / len(mcmt), 2)
    return report


def save_fps_report(results_dir, report: Optional[dict[str, Any]]) -> Optional[Path]:
    if results_dir is None or not report:
        return None

    out_path = Path(results_dir) / "fps.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(
        f"[FPS] {report['pipeline_fps']:.2f} sync-fps "
        f"({report['ms_per_sync_frame']:.1f} ms/frame, "
        f"{report['ms_per_camera_frame']:.1f} ms/cam) -> {out_path}"
    )
    return out_path


class FpsCollector:
    """Accumulate MCMT frame timings; skip warmup frames."""

    def __init__(self, num_cameras: int, warmup_frames: int = FPS_WARMUP_FRAMES):
        self.num_cameras = num_cameras
        self.warmup_frames = warmup_frames
        self._loop_i = 0
        self.frame_times: list[float] = []
        self.det_ms_samples: list[float] = []
        self.track_ms_samples: list[float] = []
        self.mcmt_ms_samples: list[float] = []

    def begin_frame(self) -> float:
        return time.perf_counter()

    def end_frame(
        self,
        t_frame: float,
        t_after_sct: float,
        t_after_mcmt: float,
        frames: list,
        per_cam_pipelines: list,
    ) -> None:
        self._loop_i += 1
        if self._loop_i <= self.warmup_frames:
            return

        self.frame_times.append(time.perf_counter() - t_frame)
        self.mcmt_ms_samples.append((t_after_mcmt - t_after_sct) * 1000.0)
        for cam_id, frame in enumerate(frames):
            if frame is None:
                continue
            ms = getattr(per_cam_pipelines[cam_id], "last_frame_ms", None)
            if ms is None:
                continue
            self.det_ms_samples.append(ms["det_ms"])
            self.track_ms_samples.append(ms["track_ms"])

    def build_report(self) -> Optional[dict[str, Any]]:
        return build_fps_report(
            self.frame_times,
            num_cameras=self.num_cameras,
            det_ms_per_frame=self.det_ms_samples or None,
            track_ms_per_frame=self.track_ms_samples or None,
            mcmt_ms_per_frame=self.mcmt_ms_samples or None,
            warmup_frames=self.warmup_frames,
        )

    def save(self, results_dir) -> Optional[Path]:
        return save_fps_report(results_dir, self.build_report())
