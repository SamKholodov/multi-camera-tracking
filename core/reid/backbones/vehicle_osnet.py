"""Vehicle-specific dual-head OSNet models."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.reid.backbones.osnet import osnet_x1_0
from core.reid.utils import logger as LOGGER

VIEW_CLASS_NAMES = (
    "front",
    "rear",
    "left",
    "left_front",
    "left_rear",
    "right",
    "right_front",
    "right_rear",
)


def extract_embedding(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the ReID embedding tensor from a model output."""
    if isinstance(output, dict):
        return output["embedding"]
    return output


class VehicleOSNet(nn.Module):
    """OSNet x1.0 with a ReID BNNeck branch and a vehicle-view MLP head."""

    def __init__(
        self,
        num_classes: int,
        *,
        num_view_classes: int = 8,
        pretrained: bool = True,
        pretrained_path: str | Path | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_view_classes = num_view_classes
        self.feature_dim = 512

        self.backbone = osnet_x1_0(
            num_classes=1000,
            pretrained=False,
            loss="triplet",
        )
        self.bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.feature_dim, num_classes, bias=False)
        self.id_classifier = self.classifier
        self.view_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_view_classes),
        )
        self._init_heads()

        if pretrained:
            self.load_imagenet_weights(pretrained_path)

    def _init_heads(self) -> None:
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias, 0.0)
        nn.init.normal_(self.classifier.weight, std=0.001)
        for module in self.view_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def load_imagenet_weights(self, pretrained_path: str | Path | None = None) -> None:
        path = Path(pretrained_path or "osnet_x1_0_imagenet.pth")
        if not path.is_file():
            LOGGER.warning(f"ImageNet OSNet weights not found: {path}")
            return

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        backbone_state = self.backbone.state_dict()
        matched = OrderedDict()
        discarded: list[str] = []

        for key, value in state_dict.items():
            key = key[7:] if key.startswith("module.") else key
            if key in backbone_state and backbone_state[key].shape == value.shape:
                matched[key] = value
            else:
                discarded.append(key)

        backbone_state.update(matched)
        self.backbone.load_state_dict(backbone_state)
        LOGGER.info(f"Loaded {len(matched)} OSNet ImageNet layers from {path}")
        if discarded:
            LOGGER.debug(f"Discarded OSNet ImageNet layers: {discarded}")

    def _shared_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.featuremaps(x)
        x = self.backbone.global_avgpool(x)
        x = x.flatten(1)
        if self.backbone.fc is not None:
            x = self.backbone.fc(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        shared = self._shared_features(x)
        feat_bn = self.bottleneck(shared)
        embedding = F.normalize(feat_bn, p=2, dim=1)

        view_logits = self.view_head(shared)

        if not self.training and not return_dict:
            return {
                "embedding": embedding,
                "view_logits": view_logits,
            }

        return {
            "id_logits": self.classifier(feat_bn),
            "embedding": embedding,
            "view_logits": view_logits,
        }

    @torch.no_grad()
    def predict_view(self, x: torch.Tensor) -> torch.Tensor:
        output = self.forward(x)
        if isinstance(output, dict):
            return output["view_logits"]
        raise TypeError("Expected dict output from VehicleOSNet.forward()")


def vehicle_osnet_x1_0(
    num_classes: int = 1000,
    pretrained: bool = True,
    loss: str = "triplet",
    use_gpu: bool = True,
    **kwargs: Any,
) -> VehicleOSNet:
    return VehicleOSNet(num_classes=num_classes, pretrained=pretrained, **kwargs)
