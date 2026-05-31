from pathlib import Path

from core.reid.utils.logger import LOGGER

logger = LOGGER

ROOT = Path(__file__).resolve().parents[3]
REID_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models"
BOXMOT = REID_ROOT

__all__ = ("LOGGER", "logger", "ROOT", "REID_ROOT", "WEIGHTS", "BOXMOT")
