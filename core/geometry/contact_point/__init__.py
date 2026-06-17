"""Contact point regression data preparation and training utilities."""

from .annotations import GtaContactRecord, load_contact_records
from .validation import RejectReason, ValidationConfig, compute_uv

__all__ = [
    "GtaContactRecord",
    "RejectReason",
    "ValidationConfig",
    "compute_uv",
    "load_contact_records",
]
