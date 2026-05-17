import cv2


def read_source_fps(cap, default: float = 10.0) -> float:
    """FPS for VideoWriter; CityFlow S02 is 10 FPS (bad AVI metadata is common)."""
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1.0 or fps > 15.0:
        return default
    return fps


class CameraManager:
    def __init__(self, sources, default_fps: float = 10.0):
        self.sources = sources
        self.caps = []
        self.fps_list = []
        for source in sources:
            cap = cv2.VideoCapture(source)
            self.caps.append(cap)
            self.fps_list.append(read_source_fps(cap, default=default_fps))

    def read_frames(self):
        frames = []
        for cap in self.caps:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                frames.append(None)
        return frames

    def release(self):
        for cap in self.caps:
            if cap.isOpened():
                cap.release()
