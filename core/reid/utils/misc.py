from pathlib import Path

from core.reid.utils import WEIGHTS


def resolve_model_path(model_path, default_dir: Path = WEIGHTS) -> Path:
    path = Path(model_path)
    candidates = [path]
    if not path.is_absolute() and path.parent == Path("."):
        candidates.append(default_dir / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in candidates:
        parent = candidate.parent
        if not parent.exists():
            continue
        lowered_name = candidate.name.lower()
        for sibling in parent.iterdir():
            if sibling.name.lower() == lowered_name:
                return sibling

    return candidates[-1]
