"""PyTorch dataset for contact point regression."""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_split_indices(splits_path: str | Path, split: str) -> list[int] | None:
    path = Path(splits_path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get(split)


def expand_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
    pad_ratio: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    px, py = w * pad_ratio, h * pad_ratio
    return (
        max(0.0, x1 - px),
        max(0.0, y1 - py),
        min(float(image_width - 1), x2 + px),
        min(float(image_height - 1), y2 + py),
    )


class ContactPointDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str = "train",
        splits_path: str | Path | None = None,
        img_size: int = 224,
        train: bool | None = None,
        bbox_pad_ratio: float = 0.05,
    ):
        rows = load_manifest(manifest_path)
        if splits_path is None:
            splits_path = Path(manifest_path).parent / "splits.json"
        indices = load_split_indices(splits_path, split)
        if indices is not None:
            rows = [rows[i] for i in indices if i < len(rows)]
        self.rows = rows
        self.split = split
        self.img_size = int(img_size)
        self.train = (split == "train") if train is None else bool(train)
        self.bbox_pad_ratio = float(bbox_pad_ratio)
        self.transform = self._build_transform()

    def _build_transform(self):
        transforms = [T.Resize((self.img_size, self.img_size))]
        if self.train:
            transforms.extend(
                [
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
                    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.25),
                ]
            )
        transforms.extend([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        return T.Compose(transforms)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(f"Missing image: {row['image_path']}")
        image_h, image_w = image.shape[:2]
        pad = random.uniform(0.0, self.bbox_pad_ratio) if self.train else 0.0
        bbox = expand_bbox(row["bbox"], image_width=image_w, image_height=image_h, pad_ratio=pad)
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError(f"Empty crop for row {idx}: {bbox}")
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(crop_rgb)
        target_u = (float(row["contact_point_x"]) - float(bbox[0])) / max(float(bbox[2] - bbox[0]), 1e-6)
        target_v = (float(row["contact_point_y"]) - float(bbox[1])) / max(float(bbox[3] - bbox[1]), 1e-6)
        return {
            "image": self.transform(pil),
            "target_uv": torch.tensor([target_u, target_v], dtype=torch.float32),
            "bbox_wh": torch.tensor([float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])], dtype=torch.float32),
            "baseline_uv": torch.tensor([0.5, 1.0], dtype=torch.float32),
        }
