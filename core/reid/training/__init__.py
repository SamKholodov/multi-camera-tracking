"""ReID model training utilities: losses, trainer, and evaluation."""

from core.reid.training.evaluator import evaluate_ranking
from core.reid.training.losses import CenterLoss, CrossEntropyLabelSmooth, TripletLoss
from core.reid.training.vehicle_trainer import VehicleReIDTrainer

__all__ = (
    "CenterLoss",
    "CrossEntropyLabelSmooth",
    "TripletLoss",
    "VehicleReIDTrainer",
    "evaluate_ranking",
)
