"""Shared constants and helpers for CityFlow S02 ablation config generation."""
from __future__ import annotations

import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "configs_cityflow"

AICITY_ROOT_REL = "datasets/AICity22_Track1_MTMC_Tracking"
S02_ROOT_REL = f"{AICITY_ROOT_REL}/validation/S02"
GT_ROOT = ROOT / AICITY_ROOT_REL / "validation" / "S02"

S02_CAM_IDS = [6, 7, 8, 9]
ZONES_PATH = "zones/s02/full/zone_tracklet.yaml"

VIDEO_FPS = 10.0
GLOBAL_DELETE_FRAMES = 200
MAX_HISTORY_GAP = 100
DEFAULT_CROSS_CAM_GAP = 100
TEMPORAL_GAPS = (20, 50, 100)
SPEED_LIMITS = (25, 35)
TRAJECTORY_KS = (1, 3)
TRAJECTORY_THRESHOLDS = {1: 0.75, 3: 0.07}
CONF_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

DETECTOR_YOLO26L = {
    "model": "models/yolo26l.pt",
    "target_classes": [2, 3, 5, 7],
    "conf_thres": 0.2,
    "imgsz": 960,
    "device": 0,
}

DETECTOR_YOLO26L_640 = {
    **DETECTOR_YOLO26L,
    "imgsz": 640,
}

TRACKER_MODERN = {
    "type": "deepocsort",
    "use_embeddings": True,
    "reid_weights": "runs/vehicle_reid/osnet_x1_0_veri_vric_wild/epoch_120.pth",
    "reid_preprocess": "pad_ratio_resize",
    "device": 0,
    "half": False,
    "det_thresh": 0.3,
    "max_age": 30,
    "min_hits": 5,
    "iou_threshold": 0.3,
    "appearance_update": "ema",
    "reid_accum_conf_thresh": 0.5,
}

TRACKER_LEGACY = {
    "type": "deepocsort",
    "use_embeddings": True,
    "reid_weights": "models/osnet_ibn_x1_0_msmt17.pt",
    "device": 0,
    "half": False,
    "det_thresh": 0.3,
    "max_age": 30,
    "min_hits": 5,
    "iou_threshold": 0.3,
    "appearance_update": "aaf",
    "reid_accum_conf_thresh": 0.6,
}

REID_CHECKPOINTS: dict[str, dict] = {
    "osnet_ibn_msmt17": {
        "reid_weights": "models/osnet_ibn_x1_0_msmt17.pt",
        "reid_cost_threshold": 0.18,
    },
    "vehicle_osnet_veri_vric": {
        "reid_weights": "runs/vehicle_reid/osnet_x1_0_veri_vric_1_without_view/best.pth",
        "reid_cost_threshold": 0.18,
    },
    "vehicle_osnet_view_finetune": {
        "reid_weights": "runs/vehicle_reid/osnet_x1_0_veri_vric_with_01_view/best.pth",
        "reid_preprocess": "pad_ratio_resize",
        "reid_cost_threshold": 0.18,
    },
    "vehicle_osnet_veri_vric_wild_epoch120": {
        "reid_weights": "runs/vehicle_reid/osnet_x1_0_veri_vric_wild/epoch_120.pth",
        "reid_preprocess": "pad_ratio_resize",
        "reid_cost_threshold": 0.5,
    },
    "vehicle_osnet_veri_vric_wild_epoch120_asso070": {
        "reid_weights": "runs/vehicle_reid/osnet_x1_0_veri_vric_wild/epoch_120.pth",
        "reid_preprocess": "pad_ratio_resize",
        "reid_cost_threshold": 0.7,
    },
}

GEO_SWEEPS: list[tuple[str, float, float]] = [
    ("geo_tight", 10.0, 25.0),
    ("geo_baseline", 14.0, 38.0),
    ("geo_loose", 14.0, 55.0),
]

GEO_TIGHT_TEMPORAL: list[tuple[str, int, float | None]] = [
    ("geo_tight_temporal_N50", 50, None),
    ("geo_tight_temporal_N15_p015", 15, 0.15),
    ("geo_tight_temporal_N15_p025", 15, 0.25),
    ("geo_tight_temporal_N15_p035", 15, 0.35),
    ("geo_tight_temporal_strict_N35", 35, None),
]


def s02_homos() -> list[str]:
    return [f"{S02_ROOT_REL}/c{c:03d}/calibration.txt" for c in S02_CAM_IDS]


def s02_sources(use_synch: bool = True) -> list[str]:
    """Video paths under repo root (prefer vdo_synch.avi when use_synch)."""
    out: list[str] = []
    for cam in S02_CAM_IDS:
        cam_dir = ROOT / S02_ROOT_REL / f"c{cam:03d}"
        if use_synch:
            synch = cam_dir / "vdo_synch.avi"
            raw = cam_dir / "vdo.avi"
            rel = synch if synch.is_file() else raw
        else:
            rel = cam_dir / "vdo.avi"
        out.append(rel.relative_to(ROOT).as_posix())
    return out


def results_rel(group: str, name: str) -> str:
    return f"configs_cityflow/{group}/{name}"


def output_block(results_rel_path: str) -> dict:
    base = f"outputs/{results_rel_path}"
    return {
        "video_fps": VIDEO_FPS,
        "visualize": False,
        "save_video": False,
        "save_video_dir": f"{base}/videos",
        "output_path": f"{base}/multicam.mp4",
    }


def write_yaml(path: Path, cfg: dict, header: str = "") -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    path.write_text((header + body) if header else body, encoding="utf-8")


def prune_stale(out_dir: Path, keep: set[str]) -> None:
    if not out_dir.is_dir():
        return
    for stale in out_dir.glob("*.yaml"):
        if stale.name not in keep:
            stale.unlink(missing_ok=True)


def association_block(
    *,
    t_min_m: float = 14.0,
    t_distant_m: float = 38.0,
    temporal_mode: str = "off",
    temporal_gap: int | None = None,
    speed_enabled: bool = False,
    speed_mps: float = 25.0,
    trajectory_enabled: bool = False,
    trajectory_k: int = 1,
    trajectory_thresh: float = 0.75,
    same_cam_add: float = 0.25,
    mid_penalty: float = 0.15,
    zone_transition: bool = False,
    zones_path: str | None = None,
    reid_cost_threshold: float = 0.7,
    temporal_mid_penalty: float | None = None,
    use_gps: bool = True,
) -> dict:
    gap = int(temporal_gap if temporal_gap is not None else DEFAULT_CROSS_CAM_GAP)
    assoc = {
        "gates": {
            "temporal": temporal_mode != "off",
            "cam_transition": False,
            "zone_transition": zone_transition,
        },
        "reid_matching": True,
        "geometry_distance_metric": "gps" if use_gps else "plane",
        "geometry_t_min_m": float(t_min_m),
        "geometry_t_distant_m": float(t_distant_m),
        "geometry_mid_penalty": float(mid_penalty),
        "same_cam_cost_add": float(same_cam_add),
        "same_frame_linking": False,
        "reid_strong_reject_threshold": 0.5,
        "max_cross_cam_gap_frames": gap,
        "reid_cost_threshold": float(reid_cost_threshold),
        "video_fps": VIDEO_FPS,
        "temporal_mode": temporal_mode,
        "speed_limit_enabled": speed_enabled,
        "trajectory_enabled": trajectory_enabled,
        "global_delete_after_frames": GLOBAL_DELETE_FRAMES,
    }
    if zones_path:
        assoc["zones_path"] = zones_path
    if speed_enabled:
        assoc["speed_limit_mode"] = "penalty"
        assoc["speed_v_max_mps"] = float(speed_mps)
    if trajectory_enabled:
        assoc["trajectory_mode"] = "linear"
        assoc["trajectory_history_k"] = int(trajectory_k)
        assoc["trajectory_threshold_m"] = float(trajectory_thresh)
    if temporal_mid_penalty is not None:
        assoc["temporal_mid_penalty"] = float(temporal_mid_penalty)
    return assoc


def multi_camera_block(
    *,
    homos: list[str] | None,
    results_rel_path: str,
    association: dict | None = None,
    legacy: bool = False,
    max_frames: int | None = None,
) -> dict:
    mc: dict = {
        "cam_ids": list(S02_CAM_IDS),
        "sources": s02_sources(),
        "homos": homos,
        "roi": "auto",
        "max_frames": max_frames,
        "results_dir": f"outputs/{results_rel_path}",
    }
    if association is not None:
        mc["association"] = association
        mc["association_cost_threshold"] = 0.7
        mc["max_history_gap_frames"] = MAX_HISTORY_GAP
        mc["video_fps"] = VIDEO_FPS
    elif legacy:
        mc["association_cost_threshold"] = 0.25
        mc["association_reid_weight"] = 0.5
        mc["geometry_max_distance"] = 0.0
        mc["max_cross_cam_gap_frames"] = DEFAULT_CROSS_CAM_GAP
        mc["max_history_gap_frames"] = 30
    return mc


def modern_run(
    *,
    group: str,
    name: str,
    homos: list[str] | None,
    association: dict,
    world_anchor: str = "bottom_center",
    detector: dict | None = None,
    tracker: dict | None = None,
    header: str = "",
) -> None:
    rel = results_rel(group, name)
    cfg = {
        "run_mode": "multi_camera",
        "world_anchor": world_anchor,
        "detector": copy.deepcopy(detector or DETECTOR_YOLO26L),
        "tracker": copy.deepcopy(tracker or TRACKER_MODERN),
        "multi_camera": multi_camera_block(
            homos=homos,
            results_rel_path=rel,
            association=association,
        ),
        "output": output_block(rel),
    }
    write_yaml(OUT_ROOT / group / f"{name}.yaml", cfg, header=header)


def legacy_run(
    *,
    group: str,
    name: str,
    tracker: dict | None = None,
    detector: dict | None = None,
    tracker_type: str | None = None,
    extra_tracker: dict | None = None,
    header: str = "",
) -> None:
    rel = results_rel(group, name)
    tr = copy.deepcopy(tracker or TRACKER_LEGACY)
    if tracker_type:
        tr["type"] = tracker_type
        tr.pop("use_embeddings", None)
        tr.pop("reid_weights", None)
    if extra_tracker:
        tr.update(extra_tracker)
    cfg = {
        "run_mode": "multi_camera",
        "detector": copy.deepcopy(detector or DETECTOR_YOLO26L_640),
        "tracker": tr,
        "multi_camera": multi_camera_block(
            homos=None,
            results_rel_path=rel,
            legacy=True,
        ),
        "output": output_block(rel),
    }
    write_yaml(OUT_ROOT / group / f"{name}.yaml", cfg, header=header)


def scale_gta_frames(n_at_30fps: int) -> int:
    """Convert GTA frame count at 30 fps to CityFlow 10 fps."""
    return max(1, round(int(n_at_30fps) / 3.0))
