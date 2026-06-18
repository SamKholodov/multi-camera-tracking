"""Runtime inference for contact point regression."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from .dataset import IMAGENET_MEAN, IMAGENET_STD, expand_bbox
from .model import ContactPointRegressor


class ContactPointInference:
    """Load ``best.pth`` and predict normalized contact ``(u, v)`` from vehicle crops."""

    def __init__(
        self,
        weights: str | Path,
        device: str | int | torch.device | None = None,
        *,
        img_size: int = 224,
        bbox_pad_ratio: float = 0.0,
        pretrained_backbone: bool = False,
    ):
        self.weights_path = Path(weights)
        self.img_size = int(img_size)
        self.bbox_pad_ratio = float(bbox_pad_ratio)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        ckpt_img_size = int(checkpoint.get("img_size", self.img_size))
        self.img_size = ckpt_img_size
        self.model = ContactPointRegressor(pretrained=pretrained_backbone).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.transform = T.Compose(
            [
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def get_crops(self, xyxys: np.ndarray, frame_bgr: np.ndarray) -> torch.Tensor:
        """Return a batch ``(N, 3, H, W)`` tensor on CPU (caller may move to device)."""
        boxes = np.asarray(xyxys, dtype=np.float32)
        if boxes.size == 0:
            return torch.empty((0, 3, self.img_size, self.img_size), dtype=torch.float32)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        boxes = boxes[:, :4]

        image_h, image_w = frame_bgr.shape[:2]
        crops: list[torch.Tensor] = []
        for x1, y1, x2, y2 in boxes:
            bbox = expand_bbox(
                (float(x1), float(y1), float(x2), float(y2)),
                image_width=image_w,
                image_height=image_h,
                pad_ratio=self.bbox_pad_ratio,
            )
            bx1, by1, bx2, by2 = [int(round(v)) for v in bbox]
            crop = frame_bgr[by1:by2, bx1:bx2]
            if crop.size == 0:
                crop = np.zeros((1, 1, 3), dtype=np.uint8)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(crop_rgb)
            crops.append(self.transform(pil))

        return torch.stack(crops, dim=0)

    @torch.no_grad()
    def predict_uv_batch(self, crops: torch.Tensor) -> np.ndarray:
        if crops.numel() == 0:
            return np.empty((0, 2), dtype=np.float32)
        crops = crops.to(self.device, non_blocking=True)
        pred = self.model(crops)
        return pred.detach().cpu().numpy().astype(np.float32)
