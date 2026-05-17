import cv2


class CameraManager:
    def __init__(self, sources):
        self.sources = sources
        self.caps = []
        for source in sources:
            cap = cv2.VideoCapture(source)
            self.caps.append(cap)

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
