"""Analyze GTA MCMT GT zone transitions vs entry/exit hypothesis."""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.io.gta_mcmt import GtaMcmtDataset, center_bbox_to_xyxy
from core.io.zones import ZoneMap

ENTRY = {2, 4, 6, 8}
EXIT = {1, 3, 5, 7}
STABLE = 5
CONFIG = _ROOT / "config/gta_mcmt_zone_polygons.yaml"
DATASET = _ROOT / "datasets/gta_mcmt"


def stable_labels(raw: list[int | None]) -> list[int | None]:
    out: list[int | None] = []
    cur: int | None = None
    run = 0
    for z in raw:
        if z is None:
            cur, run = None, 0
            out.append(None)
            continue
        if z == cur:
            run += 1
        else:
            cur, run = z, 1
        out.append(cur if run >= STABLE else None)
    return out


def transitions_from_stable(stable: list[int | None]) -> list[tuple[int, int]]:
    trans: list[tuple[int, int]] = []
    prev: int | None = None
    for z in stable:
        if z is None:
            continue
        if prev is not None and z != prev:
            trans.append((prev, z))
        prev = z
    return trans


def classify_transition(a: int, b: int) -> str:
    if a in ENTRY and b in EXIT:
        return "ok_entry_exit"
    if a in EXIT and b in ENTRY:
        return "viol_exit_to_entry"
    if a in EXIT and b in EXIT:
        return "viol_exit_to_exit"
    if a in ENTRY and b in ENTRY:
        return "viol_entry_to_entry"
    if a in ENTRY and b not in EXIT:
        return "viol_entry_to_other"
    return "other"


def main() -> None:
    zone_map = ZoneMap.from_yaml(CONFIG)
    dataset = GtaMcmtDataset(DATASET)

    all_trans: Counter[tuple[int, int]] = Counter()
    first_zone: Counter[int] = Counter()
    last_zone: Counter[int] = Counter()
    viol: Counter[str] = Counter()
    total_trans = 0

    global_timeline: dict[int, list[tuple[int, int, int | None]]] = defaultdict(list)

    for cam in range(4):
        snaps = dataset.snapshots_by_cam[cam]
        by_obj: dict[int, list[tuple[int, int | None]]] = defaultdict(list)
        for si, snap in enumerate(snaps):
            for ann in snap.annotations:
                x1, y1, x2, y2 = center_bbox_to_xyxy(ann.cx, ann.cy, ann.w, ann.h)
                z = zone_map.zone_at_bbox(cam, (x1, y1, x2, y2))
                by_obj[ann.obj_id].append((si, z))
                global_timeline[ann.obj_id].append((si, cam, z))

        for seq in by_obj.values():
            seq.sort(key=lambda x: x[0])
            stable = stable_labels([z for _, z in seq])
            nz = [z for z in stable if z is not None]
            if nz:
                first_zone[nz[0]] += 1
                last_zone[nz[-1]] += 1
            for ab in transitions_from_stable(stable):
                all_trans[ab] += 1
                total_trans += 1
                viol[classify_transition(*ab)] += 1

    global_trans: Counter[tuple[int, int]] = Counter()
    global_viol: Counter[str] = Counter()
    for obs in global_timeline.values():
        obs.sort(key=lambda x: x[0])
        stable = stable_labels([z for _, _, z in obs])
        for ab in transitions_from_stable(stable):
            global_trans[ab] += 1
            global_viol[classify_transition(*ab)] += 1

    print(f"Hypothesis: ENTRY={sorted(ENTRY)} -> EXIT={sorted(EXIT)} only")
    print(f"Stable frames: {STABLE}\n")

    print("=== Per-camera tracklet transitions ===")
    print(f"Total: {total_trans}")
    print(f"By class: {dict(viol)}")
    ok_pct = 100.0 * viol["ok_entry_exit"] / total_trans if total_trans else 0
    print(f"OK entry->exit: {viol['ok_entry_exit']} ({ok_pct:.1f}%)\n")

    print("Transition matrix (count):")
    for ab, c in sorted(all_trans.items(), key=lambda x: (-x[1], x[0])):
        a, b = ab
        cls = classify_transition(a, b)
        mark = "OK" if cls == "ok_entry_exit" else "!!"
        print(f"  [{mark}] {a} -> {b}: {c}  ({cls})")

    print("\nFirst zone on cam track (where vehicles appear):")
    for z in sorted(first_zone):
        kind = "ENTRY" if z in ENTRY else "EXIT"
        print(f"  Z{z} ({kind}): {first_zone[z]}")

    print("\nLast zone on cam track (where vehicles leave FOV):")
    for z in sorted(last_zone):
        kind = "ENTRY" if z in ENTRY else "EXIT"
        print(f"  Z{z} ({kind}): {last_zone[z]}")

    print("\n=== Global timeline (merged sync_index, cross-cam) ===")
    gtot = sum(global_trans.values())
    print(f"Total: {gtot}")
    print(f"By class: {dict(global_viol)}")
    if gtot:
        print(f"OK entry->exit: {100.0 * global_viol['ok_entry_exit'] / gtot:.1f}%")
    for ab, c in sorted(global_trans.items(), key=lambda x: (-x[1], x[0])):
        a, b = ab
        cls = classify_transition(a, b)
        mark = "OK" if cls == "ok_entry_exit" else "!!"
        print(f"  [{mark}] {a} -> {b}: {c}")


if __name__ == "__main__":
    main()
