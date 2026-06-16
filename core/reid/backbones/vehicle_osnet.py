"""Vehicle-specific dual-head OSNet models."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
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

VIEW_LAYER_ORDER = ("conv3", "conv4", "fc")
VIEW_LAYER_INPUT_DIM = {
    "conv3": 384,
    "conv4": 512,
    "fc": 512,
}


def normalize_view_layers(
    layers: Sequence[str] | str | None,
) -> tuple[str, ...]:
    """Normalize and validate configured view-head attachment points."""
    if layers is None:
        return ("fc",)
    if isinstance(layers, str):
        layers = [layers]
    normalized: list[str] = []
    for layer in layers:
        key = str(layer).lower().strip()
        if key not in VIEW_LAYER_ORDER:
            allowed = ", ".join(VIEW_LAYER_ORDER)
            raise ValueError(f"Unknown view layer '{layer}'. Expected one of: {allowed}")
        if key not in normalized:
            normalized.append(key)
    if not normalized:
        raise ValueError("view_layers cannot be empty")
    normalized.sort(key=VIEW_LAYER_ORDER.index)
    return tuple(normalized)


def normalize_view_layer_weights(
    weights: Mapping[str, float] | None,
    view_layers: Sequence[str],
) -> dict[str, float]:
    if not weights:
        return {layer: 1.0 for layer in view_layers}
    normalized: dict[str, float] = {}
    for layer in view_layers:
        if layer in weights:
            normalized[layer] = float(weights[layer])
    for key, value in weights.items():
        layer = str(key).lower().strip()
        if layer in view_layers and layer not in normalized:
            normalized[layer] = float(value)
    for layer in view_layers:
        normalized.setdefault(layer, 1.0)
    return normalized


def remap_vehicle_view_state_dict(state_dict: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """Map legacy single ``view_head`` checkpoints to ``view_heads.fc``."""
    remapped = OrderedDict()
    has_view_heads = any(key.startswith("view_heads.") for key in state_dict)
    for key, value in state_dict.items():
        if not has_view_heads and key.startswith("view_head."):
            remapped[f"view_heads.fc.{key[len('view_head.'):]}"] = value
        else:
            remapped[key] = value
    return remapped


def extract_embedding(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the ReID embedding tensor from a model output."""
    if isinstance(output, dict):
        return output["embedding"]
    return output


def _make_view_head(in_features: int, num_view_classes: int) -> nn.Sequential:
    hidden = min(256, in_features)
    mid = min(128, hidden)
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(hidden, mid),
        nn.BatchNorm1d(mid),
        nn.ReLU(inplace=True),
        nn.Linear(mid, num_view_classes),
    )


class VehicleOSNet(nn.Module):
    """OSNet x1.0 with a ReID BNNeck branch and configurable view heads."""

    def __init__(
        self,
        num_classes: int,
        *,
        num_view_classes: int = 8,
        view_layers: Sequence[str] | str | None = None,
        pretrained: bool = True,
        pretrained_path: str | Path | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_view_classes = num_view_classes
        self.view_layers = normalize_view_layers(view_layers)
        self.primary_view_layer = self.view_layers[-1]
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
        self.view_heads = nn.ModuleDict(
            {
                layer: _make_view_head(VIEW_LAYER_INPUT_DIM[layer], num_view_classes)
                for layer in self.view_layers
            }
        )
        self._init_heads()

        if pretrained:
            self.load_imagenet_weights(pretrained_path)

    @property
    def view_head(self) -> nn.Module:
        """Backward-compatible alias for the deepest configured view head."""
        return self.view_heads[self.primary_view_layer]

    def _init_heads(self) -> None:
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias, 0.0)
        nn.init.normal_(self.classifier.weight, std=0.001)
        for head in self.view_heads.values():
            for module in head.modules():
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

    def _forward_stages(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.backbone.conv1(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.conv2(x)
        stages = {
            "conv3": self.backbone.conv3(x),
        }
        stages["conv4"] = self.backbone.conv4(stages["conv3"])
        stages["conv5"] = self.backbone.conv5(stages["conv4"])
        pooled = self.backbone.global_avgpool(stages["conv5"]).flatten(1)
        stages["fc"] = self.backbone.fc(pooled) if self.backbone.fc is not None else pooled
        return stages

    def _view_features(self, stages: dict[str, torch.Tensor], layer: str) -> torch.Tensor:
        if layer == "fc":
            return stages["fc"]
        return self.backbone.global_avgpool(stages[layer]).flatten(1)

    def _compute_view_logits(self, stages: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            layer: self.view_heads[layer](self._view_features(stages, layer))
            for layer in self.view_layers
        }

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        stages = self._forward_stages(x)
        shared = stages["fc"]
        feat_bn = self.bottleneck(shared)
        embedding = F.normalize(feat_bn, p=2, dim=1)

        view_logits_by_layer = self._compute_view_logits(stages)
        view_logits = view_logits_by_layer[self.primary_view_layer]

        if not self.training and not return_dict:
            return {
                "embedding": embedding,
                "view_logits": view_logits,
                "view_logits_by_layer": view_logits_by_layer,
            }

        return {
            "id_logits": self.classifier(feat_bn),
            "embedding": embedding,
            "view_logits": view_logits,
            "view_logits_by_layer": view_logits_by_layer,
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
