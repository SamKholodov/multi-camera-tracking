from pathlib import Path

import cv2
from concurrent.futures import ThreadPoolExecutor

from core.io.gta_mcmt import GtaMcmtFrameSource, is_gta_mcmt_cam_dir


def read_source_fps(cap, default: float = 10.0) -> float:
    """FPS for VideoWriter; CityFlow S02 is 10 FPS (bad AVI metadata is common)."""
    if isinstance(cap, GtaMcmtFrameSource):
        return default
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1.0 or fps > 15.0:
        return default
    return fps


def open_frame_source(source: str | Path, default_fps: float = 10.0):
    """Video file -> VideoCapture; GTA MCMT cam folder -> GtaMcmtFrameSource."""
    path = Path(source)
    if is_gta_mcmt_cam_dir(path):
        return GtaMcmtFrameSource(path)
    return cv2.VideoCapture(str(source))


class CameraManager:
    def __init__(self, sources, default_fps: float = 10.0):
        self.sources = sources
        self.caps = []
        self.fps_list = []
        for source in sources:
            cap = open_frame_source(source, default_fps=default_fps)
            self.caps.append(cap)
            self.fps_list.append(read_source_fps(cap, default=default_fps))

    @staticmethod
    def _read_one(cap):
        ret, frame = cap.read()
        return frame if ret else None

    def read_frames(self, parallel: bool = True):
        if not parallel or len(self.caps) <= 1:
            return [self._read_one(cap) for cap in self.caps]

        with ThreadPoolExecutor(max_workers=len(self.caps)) as pool:
            return list(pool.map(self._read_one, self.caps))

    def release(self):
        for cap in self.caps:
            if hasattr(cap, "isOpened") and cap.isOpened():
                cap.release()
