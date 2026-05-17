import numpy as np


class HomographyTransformer:
    @staticmethod
    def apply_homo_to_bbox(bbox, homography):
        x1, y1, x2, y2, conf, cls = bbox

        points = np.array(
            [[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float32
        ).T

        projected = homography @ points
        projected = projected / projected[2]

        x_min = int(np.min(projected[0]))
        x_max = int(np.max(projected[0]))
        y_min = int(np.min(projected[1]))
        y_max = int(np.max(projected[1]))

        return [x_min, y_min, x_max, y_max, conf, cls]

    @staticmethod
    def apply_homo_to_point(point, H_image_to_world):
        """Image → world; pass **H_image_to_world** (``calibration_i2w``)."""
        from core.io.calibration import project_point

        x, y = point
        xw, yw = project_point(H_image_to_world, float(x), float(y))
        return [xw, yw]
