# model_registry.py
from __future__ import annotations

from collections import OrderedDict

import torch

from core.reid.core.config import MODEL_TYPES, NR_CLASSES_DICT, TRAINED_URLS
from core.reid.core.factory import MODEL_FACTORY
from core.reid.utils import logger as LOGGER


class ReIDModelRegistry:
    """Encapsulates model registration and related utilities."""

    @staticmethod
    def show_downloadable_models():
        LOGGER.info("Available .pt ReID models for automatic download")
        LOGGER.info(list(TRAINED_URLS.keys()))

    @staticmethod
    def get_model_name(model):
        for name in MODEL_TYPES:
            if name in model.name:
                return name
        try:
            checkpoint = torch.load(
                model,
                map_location="cpu",
                weights_only=False,
                encoding="latin1",
            )
            if isinstance(checkpoint, dict):
                return checkpoint.get("model_name")
        except Exception:
            pass
        return None

    @staticmethod
    def get_model_url(model):
        return TRAINED_URLS.get(model.name, None)

    @staticmethod
    def get_checkpoint_preprocess(weight_path) -> str | None:
        """Return the preprocessing method stored in a checkpoint, or None."""
        try:
            checkpoint = torch.load(
                weight_path,
                map_location="cpu",
                weights_only=False,
                encoding="latin1",
            )
            if isinstance(checkpoint, dict):
                return checkpoint.get("preprocess")
        except Exception:
            pass
        return None

    @staticmethod
    def get_checkpoint_img_size(weight_path) -> tuple[int, int] | None:
        """Return checkpoint image size as (H, W), or None."""
        try:
            checkpoint = torch.load(
                weight_path,
                map_location="cpu",
                weights_only=False,
                encoding="latin1",
            )
            if isinstance(checkpoint, dict) and "img_size" in checkpoint:
                img_size = checkpoint["img_size"]
                return int(img_size[0]), int(img_size[1])
        except Exception:
            pass
        return None

    @staticmethod
    def get_checkpoint_num_view_classes(weight_path) -> int | None:
        try:
            checkpoint = torch.load(
                weight_path,
                map_location="cpu",
                weights_only=False,
                encoding="latin1",
            )
            if isinstance(checkpoint, dict) and "num_view_classes" in checkpoint:
                return int(checkpoint["num_view_classes"])
        except Exception:
            pass
        return None

    @staticmethod
    def load_pretrained_weights(model, weight_path):
        """
        Loads pretrained weights into a model.
        Chooses the proper map_location based on CUDA availability.
        """
        checkpoint = torch.load(
            weight_path,
            map_location="cpu",
            weights_only=False,
            encoding='latin1',
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_dict = model.state_dict()

        new_state_dict = OrderedDict()
        matched_layers, discarded_layers = [], []
        for k, v in state_dict.items():
            # Remove 'module.' prefix if present
            key = k[7:] if k.startswith("module.") else k
            if key in model_dict and model_dict[key].size() == v.size():
                new_state_dict[key] = v
                matched_layers.append(key)
            else:
                discarded_layers.append(key)
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)

        if not matched_layers:
            LOGGER.debug(
                f"Pretrained weights from {weight_path} cannot be loaded. Check key names manually."
            )
        else:
            LOGGER.info(f"Loaded pretrained weights from {weight_path}")

        if discarded_layers:
            LOGGER.debug(
                f"Discarded layers due to unmatched keys or size: {discarded_layers}"
            )

    @staticmethod
    def show_available_models():
        LOGGER.info("Available models:")
        LOGGER.info(list(MODEL_FACTORY.keys()))

    @staticmethod
    def get_nr_classes(weights):
        try:
            checkpoint = torch.load(
                weights,
                map_location="cpu",
                weights_only=False,
                encoding="latin1",
            )
            if isinstance(checkpoint, dict) and "num_classes" in checkpoint:
                return int(checkpoint["num_classes"])
        except Exception:
            pass
        # Extract dataset name from weights name, then look up in the class dictionary
        dataset_key = weights.name.split("_")[1]
        return NR_CLASSES_DICT.get(dataset_key, 1)

    @staticmethod
    def build_model(
        name,
        weights,
        num_classes,
        loss="softmax",
        pretrained=True,
        use_gpu=True,
        **kwargs,
    ):
        if name not in MODEL_FACTORY:
            available = list(MODEL_FACTORY.keys())
            raise KeyError(f"Unknown model '{name}'. Must be one of {available}")

        return MODEL_FACTORY[name](
            num_classes=num_classes,
            loss=loss,
            pretrained=pretrained,
            use_gpu=use_gpu,
            **kwargs,
        )
