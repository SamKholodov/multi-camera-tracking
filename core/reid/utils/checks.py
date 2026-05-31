from importlib.metadata import PackageNotFoundError, version
from typing import Iterable

from packaging.requirements import Requirement

from core.reid.utils import logger as LOGGER


class RequirementsChecker:
    """Lightweight runtime dependency check (no auto-install)."""

    def check_packages(self, requirements: Iterable[str], extra_args=None):
        missing = []
        for req in [Requirement(r) for r in requirements]:
            if req.marker is not None and not req.marker.evaluate():
                continue
            try:
                inst_ver = version(req.name)
                if req.specifier and not req.specifier.contains(inst_ver, prereleases=True):
                    missing.append(str(req))
            except PackageNotFoundError:
                missing.append(str(req))
        if missing:
            raise ImportError(
                "Missing packages for this ReID backend: "
                + ", ".join(missing)
                + ". Install them with pip."
            )
