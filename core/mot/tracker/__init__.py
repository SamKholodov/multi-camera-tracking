from .base import BaseTracker
from .deepocsort import DeepOcSort
from .deepocsort_tracker import DeepOcSortTracker

try:
    from .bot_sort_tracker import BotSortTracker
except Exception:  # Optional dependency: boxmot
    BotSortTracker = None

__all__ = ["BaseTracker", "BotSortTracker", "DeepOcSort", "DeepOcSortTracker"]
