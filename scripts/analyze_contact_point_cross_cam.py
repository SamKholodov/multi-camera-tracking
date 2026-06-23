"""Analyze cross-camera distance gain from contact point vs bottom-center (pred bbox).

Usage:
    python scripts/analyze_contact_point_cross_cam.py --dataset both
    python scripts/analyze_contact_point_cross_cam.py --dataset gta --full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.geometry.contact_point.inference import ContactPointInference
from core.io.gta_mcmt import GtaMcmtDataset
from scripts.build_contact_point_geo_demo import (
    CONTACT_WEIGHTS,
    DEFAULT_RUNS,
    GTA_CAMERAS,
    GTA_GT_ROOT,
    load_cityflow_frame,
    load_gta_frame,
    release_cityflow_caps,
)
from scripts.cityflow_ablation_common import S02_CAM_IDS
from scripts.contact_point_cross_cam_stats import collect_cross_cam_distance_stats

METHODOLOGY = {
    "same_vehicle": "One gt_id visible on >=2 cameras in the same sync frame",
    "bbox_source": "MCMT pred bbox after greedy match to GT (IoU >= 0.5)",
    "bottom_anchor": "bbox bottom-center projected via H_image_to_world",
    "contact_anchor": "contact point UV from ContactPointInference on pred crop",
    "pair_distance": "world_distance between views on different cameras only",
    "improvement_m": "d_bottom - d_contact (positive means closer with contact point)",
    "improvement_relative_pct": "(mean_bottom - mean_contact) / mean_bottom * 100",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["gta", "cityflow", "both"], default="both")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--scan-stride", type=int, default=None)
    ap.add_argument("--full", action="store_true", help="Scan every frame (stride=1) on both datasets")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "outputs/analysis/contact_point_cross_cam.json",
    )
    ap.add_argument("--contact-weights", type=Path, default=CONTACT_WEIGHTS)
    return ap.parse_args()


def _stride_for(dataset_name: str, args: argparse.Namespace) -> int:
    if args.full:
        return 1
    if args.scan_stride is not None:
        return args.scan_stride
    return 1 if dataset_name == "cityflow" else 10


def _print_row(dataset_name: str, stats: dict) -> None:
    bottom = stats.get("bottom", {})
    contact = stats.get("contact", {})
    imp = stats.get("improvement", {})
    print(f"\n=== {dataset_name.upper()} ({stats.get('metric')}) ===")
    print(f"  run_stride={stats.get('scan_stride')}  frames={stats.get('frames_scanned')}  "
          f"vehicles={stats.get('vehicles_total')}  pairs={stats.get('pairs_total')}")
    print(f"  bottom (before):  mean={bottom.get('mean_m', float('nan')):.3f} m  "
          f"median={bottom.get('median_m', float('nan')):.3f} m")
    print(f"  contact (after):  mean={contact.get('mean_m', float('nan')):.3f} m  "
          f"median={contact.get('median_m', float('nan')):.3f} m")
    print(f"  gain Δ:           mean={imp.get('mean_m', float('nan')):.3f} m  "
          f"median={imp.get('median_m', float('nan')):.3f} m  "
          f"relative={stats.get('improvement_relative_pct', float('nan')):.1f}%")
    print(f"  pairs improved:   {stats.get('pairs_improved_pct', float('nan')):.1f}%")
    print(f"  note: {stats.get('contact_model_note')}")


def analyze_dataset(
    dataset_name: str,
    run_dir: Path,
    contact: ContactPointInference,
    args: argparse.Namespace,
) -> dict:
    pred_file = run_dir / "per_cam_local" / ("c000.txt" if dataset_name == "gta" else "c006.txt")
    if not pred_file.is_file():
        raise SystemExit(f"Missing predictions: {pred_file}")

    stride = _stride_for(dataset_name, args)
    min_side = 40.0 if dataset_name == "gta" else 20.0
    min_area = min_side * min_side
    metric = "plane" if dataset_name == "gta" else "gps"
    cameras = GTA_CAMERAS if dataset_name == "gta" else S02_CAM_IDS
    gta_dataset = GtaMcmtDataset(GTA_GT_ROOT) if dataset_name == "gta" else None

    def load_frame(cam: int, frame: int):
        if dataset_name == "gta":
            assert gta_dataset is not None
            return load_gta_frame(gta_dataset, cam, frame)
        return load_cityflow_frame(cam, frame)

    progress_every = 50 if dataset_name == "gta" else 20
    print(f"[{dataset_name}] Scanning stride={stride} run={run_dir}...", flush=True)
    result = collect_cross_cam_distance_stats(
        dataset_name=dataset_name,
        run_dir=run_dir,
        contact=contact,
        scan_stride=stride,
        min_side_px=min_side,
        min_area=min_area,
        metric=metric,
        cameras=cameras,
        load_frame_fn=load_frame,
        max_frames=args.max_frames,
        collect_examples=False,
        progress_every=progress_every,
    )

    if dataset_name == "cityflow":
        release_cityflow_caps()

    stats = result.stats
    stats["run_dir"] = str(run_dir)
    stats["scan_stride"] = stride
    stats["methodology"] = METHODOLOGY
    _print_row(dataset_name, stats)
    return stats


def main() -> None:
    args = parse_args()
    if not args.contact_weights.is_file():
        raise SystemExit(f"Contact weights not found: {args.contact_weights}")

    print(f"Loading contact model: {args.contact_weights}", flush=True)
    contact = ContactPointInference(
        weights=args.contact_weights,
        device=0,
        bbox_pad_ratio=0.05,
        pretrained_backbone=False,
    )

    datasets = ["gta", "cityflow"] if args.dataset == "both" else [args.dataset]
    payload: dict = {
        "methodology": METHODOLOGY,
        "contact_weights": str(args.contact_weights),
        "datasets": {},
    }

    for ds in datasets:
        run_dir = args.run_dir if args.run_dir is not None else DEFAULT_RUNS[ds]
        if args.run_dir is not None and len(datasets) > 1:
            raise SystemExit("--run-dir only with single --dataset")
        payload["datasets"][ds] = analyze_dataset(ds, run_dir, contact, args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
