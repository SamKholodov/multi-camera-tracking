"""Generate GTA conf_thres ablation configs.

Usage:
    python scripts/generate_conf_ablation_configs.py
    python scripts/generate_conf_ablation_configs.py --values 0.1,0.2,0.3,0.4,0.5,0.6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUT_DIR = _ROOT / "configs_gta" / "conf_ablation"
DEFAULT_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def _conf_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "_")


def build_config(conf: float) -> dict:
    tag = _conf_tag(conf)
    return {
        "run_mode": "multi_camera",
        "detector": {
            "model": "models/yolo26l.pt",
            "target_classes": [2, 3, 5, 7],
            "conf_thres": conf,
            "imgsz": 640,
            "device": 0,
        },
        "tracker": {
            "type": "deepocsort",
            "use_embeddings": True,
            "reid_weights": "models/osnet_ibn_x1_0_msmt17.pt",
            "device": 0,
            "half": False,
            "det_thresh": conf,
            "max_age": 30,
            "min_hits": 3,
            "iou_threshold": 0.3,
            "appearance_update": "aaf",
            "reid_accum_conf_thresh": conf,
        },
        "multi_camera": {
            "cam_ids": [0, 1, 2, 3],
            "sources": [f"datasets/gta_mcmt/cam-{i}" for i in range(4)],
            "homos": None,
            "roi": "auto",
            "max_frames": 2000,
            "association": {
                "gates": {
                    "temporal": False,
                    "cam_transition": False,
                    "zone_transition": False,
                },
                "reid_matching": True,
                "geometry_distance_metric": "plane",
                "geometry_t_min_m": 14.0,
                "geometry_t_distant_m": 0.0,
                "geometry_mid_penalty": 0.15,
                "same_cam_cost_add": 0.25,
                "max_cross_cam_gap_frames": 300,
                "reid_cost_threshold": 0.25,
            },
            "max_history_gap_frames": 30,
            "results_dir": f"outputs/configs_gta/conf_ablation/conf_{tag}",
        },
        "output": {
            "video_fps": 10,
            "visualize": False,
            "save_video": False,
            "output_path": f"outputs/configs_gta/conf_ablation/conf_{tag}/multicam.mp4",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--values",
        type=str,
        default=",".join(str(v) for v in DEFAULT_VALUES),
        help="Comma-separated detector confidence thresholds",
    )
    args = ap.parse_args()
    values = [float(v.strip()) for v in args.values.split(",") if v.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for conf in values:
        tag = _conf_tag(conf)
        dst = OUT_DIR / f"conf_{tag}.yaml"
        header = (
            f"# GTA conf_thres ablation: detector.conf_thres={conf} "
            f"(det_thresh and reid_accum_conf_thresh aligned).\n"
            f"# Run: python run.py --config configs_gta/conf_ablation/conf_{tag}.yaml\n\n"
        )
        body = yaml.dump(build_config(conf), sort_keys=False, allow_unicode=True)
        dst.write_text(header + body, encoding="utf-8")
        print(f"Wrote {dst.relative_to(_ROOT).as_posix()}")

    print(f"\nDone: {len(values)} configs in {OUT_DIR.relative_to(_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
