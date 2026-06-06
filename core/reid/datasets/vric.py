"""VRIC vehicle re-identification dataset."""

from __future__ import annotations

from pathlib import Path
from typing import List

from core.reid.datasets.base import BaseReIDDataset, ReIDSample


class VRIC(BaseReIDDataset):
    name = "vric"

    _SUBDIRS = ("VRIC", "vric")

    def __init__(self, root: str, **kwargs):
        resolved = self._resolve_root(root)
        super().__init__(str(resolved), **kwargs)

    def _resolve_root(self, root: str) -> Path:
        p = Path(root)
        if (p / "vric_train.txt").is_file():
            return p
        for sub in self._SUBDIRS:
            candidate = p / sub
            if (candidate / "vric_train.txt").is_file():
                return candidate
        raise FileNotFoundError(
            f"Cannot find VRIC dataset under {root}. Expected vric_train.txt."
        )

    def _load_split(self, split: str) -> List[ReIDSample]:
        image_dirs = {
            "train": "train_images",
            "query": "probe_images",
            "gallery": "gallery_images",
        }
        annotation_files = {
            "train": "vric_train.txt",
            "query": "vric_probe.txt",
            "gallery": "vric_gallery.txt",
        }
        return _parse_vric_file(
            self.root / annotation_files[split],
            self.root / image_dirs[split],
        )


def _parse_vric_file(annotation_path: Path, image_dir: Path) -> List[ReIDSample]:
    samples: List[ReIDSample] = []
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing VRIC annotation file: {annotation_path}")

    for line in annotation_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        img_name, pid_raw, cam_raw = parts[:3]
        img_path = Path(img_name)
        if not img_path.is_absolute():
            img_path = image_dir / img_path.name

        samples.append(
            ReIDSample(
                img_path=str(img_path),
                pid=int(pid_raw),
                camid=int(cam_raw),
                view_id=-1,
                has_view=False,
                source="vric",
                source_id=1,
            )
        )

    return samples
