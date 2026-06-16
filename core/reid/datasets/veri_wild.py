"""VeRI-Wild vehicle re-identification dataset.

Directory layout (official README)::

    VeRI-Wild/
        train_test_split/
            train_list.txt
            test_3000.txt
            test_3000_query.txt
            test_5000.txt
            test_5000_query.txt
            test_10000.txt
            test_10000_query.txt
        images/
            <vehicle_id>/*.jpg
        README.md
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from core.reid.datasets.base import BaseReIDDataset, ReIDSample

_README_SPLIT_FILES = {
    "train": "train_list.txt",
    "query": {
        3000: "test_3000_query.txt",
        5000: "test_5000_query.txt",
        10000: "test_10000_query.txt",
    },
    "gallery": {
        3000: "test_3000.txt",
        5000: "test_5000.txt",
        10000: "test_10000.txt",
    },
}


class VeRIWild(BaseReIDDataset):
    name = "veri_wild"

    _SUBDIRS = ("VeRI-Wild", "veri_wild", "VeRIWild", "veriwild")

    def __init__(self, root: str, *, test_size: int = 3000, **kwargs):
        self.test_size = int(test_size)
        resolved = self._resolve_root(root)
        super().__init__(str(resolved), **kwargs)

    def _resolve_root(self, root: str) -> Path:
        p = Path(root)
        if self._has_layout(p):
            return p
        for sub in self._SUBDIRS:
            candidate = p / sub
            if self._has_layout(candidate):
                return candidate
        raise FileNotFoundError(
            f"Cannot find VeRI-Wild dataset under {root}. "
            f"Expected README layout: train_test_split/train_list.txt and images/."
        )

    @staticmethod
    def _has_layout(path: Path) -> bool:
        if not path.is_dir():
            return False
        split_dir = path / "train_test_split"
        images_dir = path / "images"
        return (
            split_dir.is_dir()
            and images_dir.is_dir()
            and (split_dir / _README_SPLIT_FILES["train"]).is_file()
        )

    def _split_file(self, split: str) -> Path:
        split_dir = self.root / "train_test_split"
        if split == "train":
            filename = _README_SPLIT_FILES["train"]
        elif split in ("query", "gallery"):
            filenames = _README_SPLIT_FILES[split]
            if self.test_size not in filenames:
                allowed = sorted(filenames)
                raise ValueError(
                    f"Unsupported VeRI-Wild test_size={self.test_size}. "
                    f"Expected one of {allowed}."
                )
            filename = filenames[self.test_size]
        else:
            raise KeyError(f"Unknown split '{split}'")

        path = split_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing VeRI-Wild split file: {path}")
        return path

    def _image_root(self) -> Path:
        return self.root / "images"

    def _load_split(self, split: str) -> List[ReIDSample]:
        return _parse_veri_wild_file(
            self._split_file(split),
            self._image_root(),
            source=self.name,
        )


def _parse_veri_wild_file(
    annotation_path: Path,
    image_root: Path,
    *,
    source: str,
) -> List[ReIDSample]:
    samples: List[ReIDSample] = []
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing VeRI-Wild annotation file: {annotation_path}")

    for line in annotation_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        img_rel, pid_raw, cam_raw = parts[:3]
        img_path = image_root / Path(img_rel.replace("\\", "/"))
        if not img_path.is_file():
            continue

        samples.append(
            ReIDSample(
                img_path=str(img_path),
                pid=int(pid_raw),
                camid=int(cam_raw),
                view_id=-1,
                has_view=False,
                source=source,
                source_id=0,
            )
        )

    return samples
