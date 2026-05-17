import os

import cv2
import numpy as np


class Utils:
    @staticmethod
    def filter_detections(dets):
        if len(dets) == 0:
            return dets

        valid = []
        for d in dets:
            x1, y1, x2, y2 = d[:4]

            if not np.isfinite(d).all():
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) < 2 or (y2 - y1) < 2:
                continue

            valid.append(d)

        return np.array(valid) if valid else np.empty((0, dets.shape[1]))

    @staticmethod
    def convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h + 1e-6)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def nms1(bboxes, threshold):
        if bboxes is None or len(bboxes) == 0:
            return np.empty((0, 5))

        boxes = bboxes[:, :4].tolist()
        scores = bboxes[:, 4].tolist()

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            score_threshold=0.3,
            nms_threshold=threshold,
            eta=1.0,
            top_k=0,
        )

        if len(indices) > 0:
            indices = indices.flatten()
            return bboxes[indices]
        return np.empty((0, 5))

    @staticmethod
    def compute_iou(box, boxes):
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union = area1 + area2 - inter
        return inter / (union + 1e-7)

    @staticmethod
    def compute_giou(box, boxes):
        iou = Utils.compute_iou(box, boxes)

        x1_c = np.minimum(box[0], boxes[:, 0])
        y1_c = np.minimum(box[1], boxes[:, 1])
        x2_c = np.maximum(box[2], boxes[:, 2])
        y2_c = np.maximum(box[3], boxes[:, 3])

        area_c = (x2_c - x1_c) * (y2_c - y1_c)

        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area1 + area2 - inter

        return iou - (area_c - union) / (area_c + 1e-7)

    @staticmethod
    def compute_diou(box, boxes):
        iou = Utils.compute_iou(box, boxes)

        center_x1 = (box[0] + box[2]) / 2
        center_y1 = (box[1] + box[3]) / 2

        center_x2 = (boxes[:, 0] + boxes[:, 2]) / 2
        center_y2 = (boxes[:, 1] + boxes[:, 3]) / 2

        rho2 = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2

        x1_c = np.minimum(box[0], boxes[:, 0])
        y1_c = np.minimum(box[1], boxes[:, 1])
        x2_c = np.maximum(box[2], boxes[:, 2])
        y2_c = np.maximum(box[3], boxes[:, 3])

        c2 = (x2_c - x1_c) ** 2 + (y2_c - y1_c) ** 2

        return iou - rho2 / (c2 + 1e-7)

    @staticmethod
    def compute_ciou(box, boxes):
        diou = Utils.compute_diou(box, boxes)

        w1 = box[2] - box[0]
        h1 = box[3] - box[1]
        w2 = boxes[:, 2] - boxes[:, 0]
        h2 = boxes[:, 3] - boxes[:, 1]

        v = (4 / (np.pi**2)) * (
            np.arctan(w2 / (h2 + 1e-7)) - np.arctan(w1 / (h1 + 1e-7))
        ) ** 2
        iou = Utils.compute_iou(box, boxes)
        alpha = v / (1 - iou + v + 1e-7)

        return diou - alpha * v

    @staticmethod
    def nms(bboxes, threshold=0.5, metric="iou"):
        if bboxes is None or len(bboxes) == 0:
            return np.empty((0, 5))

        bboxes = bboxes[np.argsort(-bboxes[:, 4])]
        keep = []

        while len(bboxes) > 0:
            current = bboxes[0]
            keep.append(current)

            if len(bboxes) == 1:
                break

            rest = bboxes[1:]

            if metric == "iou":
                overlaps = Utils.compute_iou(current[:4], rest[:, :4])
            elif metric == "giou":
                overlaps = Utils.compute_giou(current[:4], rest[:, :4])
            elif metric == "diou":
                overlaps = Utils.compute_diou(current[:4], rest[:, :4])
            elif metric == "ciou":
                overlaps = Utils.compute_ciou(current[:4], rest[:, :4])
            else:
                raise ValueError("Unknown metric")

            bboxes = rest[overlaps < threshold]

        return np.array(keep)

    @staticmethod
    def collect_sources(root_dir):
        sources = []
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d != "detections"]

            for f in files:
                if not f.lower().endswith(".mp4"):
                    continue
                if f.endswith("_det.mp4"):
                    continue
                sources.append(os.path.join(root, f))

        return sources
