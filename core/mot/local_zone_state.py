from __future__ import annotations

from dataclasses import dataclass, field

from core.io.zones import ZoneMap


@dataclass
class LocalTrackZoneState:
    zone_entry: int | None = None
    zone_exit: int | None = None
    _pending_entry: list[int | None] = field(default_factory=list)
    _pending_exit: list[int | None] = field(default_factory=list)
    _exit_candidate: int | None = None

    @property
    def effective_out(self) -> int | None:
        return self.zone_exit if self.zone_exit is not None else self.zone_entry


class LocalZoneTracker:
    """Stable [zone_entry, zone_exit] state for local SCT tracklets."""

    def __init__(self, zone_map: ZoneMap | None):
        self.zone_map = zone_map
        self._states: dict[tuple[int, int], LocalTrackZoneState] = {}

    def get(self, key: tuple[int, int]) -> LocalTrackZoneState | None:
        return self._states.get((int(key[0]), int(key[1])))

    def pop(self, key: tuple[int, int]) -> None:
        self._states.pop((int(key[0]), int(key[1])), None)

    def entry_zone(self, key: tuple[int, int]) -> int | None:
        state = self.get(key)
        return state.zone_entry if state is not None else None

    def effective_out(self, key: tuple[int, int]) -> int | None:
        state = self.get(key)
        return state.effective_out if state is not None else None

    def update(
        self,
        key: tuple[int, int],
        cam: int,
        bbox: tuple[float, float, float, float] | None,
    ) -> None:
        if self.zone_map is None or self.zone_map.mode != "tracklet":
            return
        key = (int(key[0]), int(key[1]))
        state = self._states.setdefault(key, LocalTrackZoneState())
        zone = self.zone_map.zone_at_bbox(int(cam), bbox)
        if state.zone_entry is None:
            state._pending_entry.append(zone)
            fixed = self.zone_map.resolve_stable_zone(state._pending_entry)
            if fixed is not None:
                state.zone_entry = fixed
                state._pending_entry.clear()
            return

        if zone is None or zone == state.effective_out:
            state._pending_exit.clear()
            state._exit_candidate = None
            return

        if state.zone_exit is None and zone == state.zone_entry:
            state._pending_exit.clear()
            state._exit_candidate = None
            return

        source_zone = state.effective_out
        if not self.zone_map.allows_transition(source_zone, zone):
            state._pending_exit.clear()
            state._exit_candidate = None
            return

        if state._exit_candidate != zone:
            state._exit_candidate = zone
            state._pending_exit = [zone]
        else:
            state._pending_exit.append(zone)

        fixed = self.zone_map.resolve_stable_zone(state._pending_exit)
        if fixed is not None:
            state.zone_exit = fixed
            state._pending_exit.clear()
            state._exit_candidate = None
