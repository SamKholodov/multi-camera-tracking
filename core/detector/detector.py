import os
from typing import Optional, Union

import torch
from ultralytics import RTDETR
from ultralytics import YOLO


class Detector:
    def __init__(
        self,
        model: str = "rtdetr-l.pt",
        target_classes: Union[int, list[int], None] = 0,
        device: Optional[Union[str, int]] = None,
        conf_thres: float = 0.3,
    ):
        model_stem = os.path.splitext(os.path.basename(model))[0]
        if model_stem.startswith("rtdetr"):
            self.model = RTDETR(model)
        else:
            self.model = YOLO(model)

        self.conf_thres = conf_thres
        self.target_classes = target_classes

        self.model_name = model_stem
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)
        self.model.fuse()

    def detect(self, frame):
        if frame is None:
            return [], []

        results = self.model(
            frame,
            imgsz=960,
            conf=self.conf_thres,
            verbose=False,
            classes=self.target_classes,
        )

        res = results[0]
        if res.boxes is None:
            return [], []

        detections = []
        labels = []
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls in zip(boxes, scores, classes):
            detections.append([*box, float(score), cls])
            labels.append(cls)
        return detections, labels
