"""GT cross-camera handoff frame gaps for temporal association tuning.

Estimates ``max_cross_cam_gap_frames`` and ``temporal_mid_penalty`` from GTA MCMT
annotations. Matches pipeline handoff logic: gap = query_frame - last_global_frame.

Usage:
    python scripts/analyze_gta_temporal_gt.py
    python scripts/analyze_gta_temporal_gt.py --json --output outputs/configs_gta/temporal_gt_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.io.gta_mcmt import GtaMcmtDataset  # noqa: E402


def _stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def collect_handoff_gaps(dataset_root: Path, *, max_frames: int | None) -> dict:
    ds = GtaMcmtDataset(dataset_root)
    n_frames = ds.length if max_frames is None else min(ds.length, max_frames)

    # obj_id -> sorted unique (frame, cam) presence
    per_obj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    per_obj_frames: dict[int, set[int]] = defaultdict(set)

    for frame_idx in range(n_frames):
        for cam in range(ds.num_cameras):
            snap = ds.snapshot(cam, frame_idx)
            for ann in snap.annotations:
                per_obj_frames[ann.obj_id].add(frame_idx)
                per_obj[ann.obj_id].append((frame_idx, cam))

    handoff_gaps: list[int] = []
    consecutive_switch_gaps: list[int] = []
    reappear_gaps: list[int] = []
    pair_gaps: list[int] = []

    for _obj_id, events in per_obj.items():
        events.sort(key=lambda x: (x[0], x[1]))
        deduped: list[tuple[int, int]] = []
        for item in events:
            if deduped and deduped[-1] == item:
                continue
            deduped.append(item)

        for (f0, c0), (f1, c1) in zip(deduped, deduped[1:]):
            gap = int(f1) - int(f0)
            if gap <= 0:
                continue
            if c0 != c1:
                consecutive_switch_gaps.append(gap)

        seen_cams: set[int] = set()
        for frame_idx, cam in deduped:
            if cam in seen_cams:
                continue
            prior = [f for f, c in deduped if f < frame_idx]
            if not prior:
                seen_cams.add(cam)
                continue
            last_global = max(prior)
            gap = int(frame_idx) - int(last_global)
            if gap > 0:
                reappear_gaps.append(gap)
            seen_cams.add(cam)

        by_cam: dict[int, list[int]] = defaultdict(list)
        for frame_idx, cam in deduped:
            by_cam[cam].append(frame_idx)
        cams = sorted(by_cam)
        for cam_a in cams:
            last_a = max(by_cam[cam_a])
            for cam_b in cams:
                if cam_b == cam_a:
                    continue
                after_a = [f for f in by_cam[cam_b] if f > last_a]
                if not after_a:
                    continue
                pair_gaps.append(int(min(after_a) - last_a))

    # Sequential handoff: last frame on cam A -> first frame on cam B after A ends.
    handoff_gaps = pair_gaps if pair_gaps else reappear_gaps
    # Gap when object vanishes from all cameras then reappears (stale global track).
    occlusion_gaps: list[int] = []
    for frames in per_obj_frames.values():
        fs = sorted(frames)
        for f0, f1 in zip(fs, fs[1:]):
            gap = int(f1) - int(f0)
            if gap > 1:
                occlusion_gaps.append(gap)

    st_handoff = _stats(handoff_gaps)
    st_occlusion = _stats(occlusion_gaps)
    # Use occlusion gaps for max_cross_cam_gap — matches stale-track re-association.
    st = st_occlusion if occlusion_gaps else st_handoff
    reid_thresh = 0.7
    # Penalty should be material vs reid_cost_threshold; sweep around 15–40% of threshold.
    penalty_candidates = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    penalty_rationale = {
        str(p): f"{100 * p / reid_thresh:.0f}% of reid_cost_threshold={reid_thresh}"
        for p in penalty_candidates
    }

    fps = 30.0
    suggested = {
        "max_cross_cam_gap_frames": {
            "penalty_only_p90": int(st.get("p90", 44)),
            "penalty_only_p95": int(st.get("p95", 105)),
            "strict_p95": int(st.get("p95", 105)),
            "strict_p99": int(min(st.get("p99", 300), 600)),
            "legacy_N60": 60,
            "legacy_N150": 150,
            "legacy_N300": 300,
        },
        "max_cross_cam_gap_seconds_at_30fps": {
            "p90": round(float(st.get("p90", 0)) / fps, 2),
            "p95": round(float(st.get("p95", 0)) / fps, 2),
            "p99": round(float(st.get("p99", 0)) / fps, 2),
        },
        "temporal_mid_penalty_candidates": penalty_candidates,
        "temporal_mid_penalty_recommended": [0.15, 0.25, 0.35],
        "temporal_mid_penalty_rationale": penalty_rationale,
        "reid_cost_threshold_ref": reid_thresh,
    }

    return {
        "dataset_root": str(dataset_root),
        "frames": int(n_frames),
        "fps": fps,
        "handoff_gap_frames": st_handoff,
        "occlusion_gap_frames": st_occlusion,
        "reappear_gap_frames": _stats(reappear_gaps),
        "consecutive_switch_gap_frames": _stats(consecutive_switch_gaps),
        "pairwise_handoff_gap_frames": _stats(pair_gaps),
        "suggested": suggested,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=_ROOT / "datasets" / "gta_mcmt")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    result = collect_handoff_gaps(args.dataset_root, max_frames=args.max_frames)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(result, indent=2))
        return

    st = result["occlusion_gap_frames"]
    st_h = result["handoff_gap_frames"]
    sug = result["suggested"]
    print(f"GTA temporal gaps ({result['frames']} sync frames, fps={result['fps']})")
    print("\nocclusion_gap_frames (off all cams -> reappear; use for N tuning):")
    for k, v in st.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v}")
    print("\ncross_cam_handoff_gap_frames (cam A last -> cam B after):")
    for k, v in st_h.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v}")

    print("\nSuggested max_cross_cam_gap_frames:")
    for k, v in sug["max_cross_cam_gap_frames"].items():
        print(f"  {k}: {v}")

    print("\nSuggested temporal_mid_penalty sweep (geo_tight + penalty_only):")
    print(f"  {sug['temporal_mid_penalty_recommended']}")
    print(f"  (default in code was 0.10 — likely too weak vs reid_cost_threshold={sug['reid_cost_threshold_ref']})")


if __name__ == "__main__":
    main()
