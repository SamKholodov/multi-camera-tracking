"""ReID dataset registry and dataloader construction."""

from __future__ import annotations

from typing import Any, Dict, List, Type

from core.reid.datasets.base import BaseReIDDataset, CombinedReIDDataset
from core.reid.datasets.veri776 import VeRi776
from core.reid.datasets.vric import VRIC

DATASET_REGISTRY: Dict[str, Type[BaseReIDDataset]] = {
    "veri": VeRi776,
    "veri776": VeRi776,
    "vric": VRIC,
}


def build_dataset(name: str, root: str, **kwargs: Any) -> BaseReIDDataset:
    """Instantiate a ReID dataset by name."""
    key = name.lower().replace("-", "").replace("_", "")
    if key in ("veri776", "veri"):
        key = "veri"
    if key == "vric":
        key = "vric"
    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {sorted(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[key](root=root, **kwargs)


def build_combined_dataset(names: List[str], root: str, **kwargs: Any) -> CombinedReIDDataset:
    """Load multiple datasets and combine their train splits with PID remapping.

    Query/gallery splits are kept per-dataset for separate evaluation.
    """
    datasets = [build_dataset(n.strip(), root, **kwargs) for n in names]
    return CombinedReIDDataset(datasets)


__all__ = (
    "DATASET_REGISTRY",
    "BaseReIDDataset",
    "CombinedReIDDataset",
    "VeRi776",
    "VRIC",
    "build_dataset",
    "build_combined_dataset",
)
