"""MobileNetV3 contact point regressor."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ContactPointRegressor(nn.Module):
    """Predict normalized contact point coordinates ``(u, v)`` from a vehicle crop."""

    def __init__(self, *, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def uv_to_pixel(
    uv: tuple[float, float] | list[float],
    bbox: tuple[float, float, float, float] | list[float],
) -> tuple[float, float]:
    u, v = float(uv[0]), float(uv[1])
    x1, y1, x2, y2 = [float(x) for x in bbox]
    return x1 + u * (x2 - x1), y1 + v * (y2 - y1)
