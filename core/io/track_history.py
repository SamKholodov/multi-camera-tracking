"""Serialize SCT track history dicts to JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import numpy as np


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def save_tracks_history_json(
    tracks: dict,
    output_path: Union[str, Path] = "tracks_history.json",
    *,
    indent: int = 2,
) -> Path:
    """Write ``SingleCameraTrackerPipeline.tracks`` to a JSON file."""
    path = Path(output_path)
    serializable_tracks = {
        str(int(track_id)): _to_jsonable(track_data)
        for track_id, track_data in tracks.items()
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable_tracks, f, ensure_ascii=False, indent=indent)
    return path
