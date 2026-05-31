from pathlib import Path
from urllib.request import urlretrieve

from core.reid.utils import logger as LOGGER


def download_file(url: str, dest: Path) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = str(url)

    if "drive.google.com" in url or "drive.usercontent.google.com" in url:
        try:
            import gdown
        except ImportError as e:
            raise ImportError(
                "Google Drive weight download requires gdown: pip install gdown"
            ) from e
        LOGGER.info(f"Downloading {url} -> {dest}")
        gdown.download(url, str(dest), quiet=False)
        return

    LOGGER.info(f"Downloading {url} -> {dest}")
    urlretrieve(url, dest)
