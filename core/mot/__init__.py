from .tracker.deepocsort import DeepOcSort
from .tracker.deepocsort_tracker import DeepOcSortTracker
from .tracker.sort_tracker import SortTracker

try:
    from .tracker.bot_sort_tracker import BotSortTracker
except Exception:  # Optional dependency: boxmot
    BotSortTracker = None

__all__ = ["BotSortTracker", "DeepOcSort", "DeepOcSortTracker", "SortTracker"]