"""Generate full configs_cityflow/ suite (mirror of configs_gta for S02).

Usage:
    python scripts/generate_configs_cityflow.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cityflow_ablation_common import (  # noqa: E402
    CONF_VALUES,
    DEFAULT_CROSS_CAM_GAP,
    DETECTOR_YOLO26L,
    DETECTOR_YOLO26L_640,
    GEO_SWEEPS,
    GEO_TIGHT_TEMPORAL,
    OUT_ROOT,
    REID_CHECKPOINTS,
    ROOT,
    SPEED_LIMITS,
    TEMPORAL_GAPS,
    TRACKER_LEGACY,
    TRACKER_MODERN,
    TRAJECTORY_KS,
    TRAJECTORY_THRESHOLDS,
    VIDEO_FPS,
    ZONES_PATH,
    association_block,
    legacy_run,
    modern_run,
    multi_camera_block,
    output_block,
    prune_stale,
    results_rel,
    s02_homos,
    write_yaml,
)

def _conf_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "_")


def generate_baseline() -> None:
    rel = "configs_cityflow/baseline"
    cfg = {
        "run_mode": "multi_camera",
        "detector": copy.deepcopy(DETECTOR_YOLO26L_640),
        "tracker": copy.deepcopy(TRACKER_LEGACY),
        "multi_camera": multi_camera_block(homos=None, results_rel_path=rel, legacy=True),
        "output": output_block(rel),
    }
    header = (
        "# CityFlow S02 legacy baseline (msmt17 ReID, AAF, no geometry tiers).\n"
        "# Run: python run.py --config configs_cityflow/baseline.yaml\n\n"
    )
    write_yaml(OUT_ROOT / "baseline.yaml", cfg, header=header)


def generate_assoc_ablation() -> None:
    out_dir = OUT_ROOT / "assoc_ablation"
    keep = set()

    modern_run(
        group="assoc_ablation",
        name="reid_only",
        homos=None,
        association=association_block(
            t_distant_m=0.0, same_cam_add=0.0, mid_penalty=0.0, use_gps=False
        ),
        header="# CityFlow S02: ReID-only cross-camera association.\n\n",
    )
    keep.add("reid_only.yaml")

    modern_run(
        group="assoc_ablation",
        name="+zone_tracklet",
        homos=None,
        association=association_block(
            t_distant_m=0.0,
            zone_transition=True,
            zones_path=ZONES_PATH,
            reid_cost_threshold=0.25,
            use_gps=False,
        ),
        tracker=copy.deepcopy(TRACKER_MODERN),
        header="# CityFlow S02: ReID + zone tracklet graph.\n\n",
    )
    keep.add("+zone_tracklet.yaml")

    assoc = association_block(t_min_m=14.0, t_distant_m=38.0)
    modern_run(
        group="assoc_ablation",
        name="no_different_cam_geo_tiers",
        homos=s02_homos(),
        association=assoc,
        header="# CityFlow S02: tiered geometry (14/38 m GPS), no legacy gates.\n\n",
    )
    keep.add("no_different_cam_geo_tiers.yaml")
    prune_stale(out_dir, keep)


def generate_reid_ablation() -> None:
    out_dir = OUT_ROOT / "reid_ablation"
    keep: set[str] = set()
    for name, ckpt in REID_CHECKPOINTS.items():
        tr = copy.deepcopy(TRACKER_MODERN)
        tr["reid_weights"] = ckpt["reid_weights"]
        if "reid_preprocess" in ckpt:
            tr["reid_preprocess"] = ckpt["reid_preprocess"]
        else:
            tr.pop("reid_preprocess", None)
        modern_run(
            group="reid_ablation",
            name=name,
            homos=None,
            association=association_block(
                t_distant_m=0.0,
                reid_cost_threshold=ckpt["reid_cost_threshold"],
                use_gps=False,
            ),
            tracker=tr,
            header=f"# CityFlow S02 ReID ablation: {name}\n\n",
        )
        keep.add(f"{name}.yaml")
    prune_stale(out_dir, keep)


def generate_temporal_ablation() -> None:
    out_dir = OUT_ROOT / "temporal_ablation"
    keep = {"temporal_off.yaml"}
    homos = s02_homos()

    modern_run(
        group="temporal_ablation",
        name="temporal_off",
        homos=homos,
        association=association_block(t_min_m=14.0, t_distant_m=38.0),
    )

    for gap in TEMPORAL_GAPS:
        name = f"temporal_penalty_N{gap}"
        keep.add(f"{name}.yaml")
        modern_run(
            group="temporal_ablation",
            name=name,
            homos=homos,
            association=association_block(
                t_min_m=14.0,
                t_distant_m=38.0,
                temporal_mode="penalty_only",
                temporal_gap=gap,
            ),
        )
    prune_stale(out_dir, keep)


def generate_kinematic_ablation() -> None:
    out_dir = OUT_ROOT / "kinematic_ablation"
    keep: set[str] = set()
    homos = s02_homos()
    for speed in SPEED_LIMITS:
        name = f"speed_penalty_v{speed}"
        keep.add(f"{name}.yaml")
        modern_run(
            group="kinematic_ablation",
            name=name,
            homos=homos,
            association=association_block(
                t_min_m=14.0,
                t_distant_m=38.0,
                speed_enabled=True,
                speed_mps=float(speed),
            ),
        )
    prune_stale(out_dir, keep)


def generate_trajectory_ablation() -> None:
    out_dir = OUT_ROOT / "trajectory_ablation"
    keep: set[str] = set()
    homos = s02_homos()
    for k in TRAJECTORY_KS:
        name = f"traj_linear_K{k}"
        keep.add(f"{name}.yaml")
        modern_run(
            group="trajectory_ablation",
            name=name,
            homos=homos,
            association=association_block(
                t_min_m=14.0,
                t_distant_m=38.0,
                trajectory_enabled=True,
                trajectory_k=k,
                trajectory_thresh=TRAJECTORY_THRESHOLDS[k],
            ),
        )
    prune_stale(out_dir, keep)


def generate_geo_ablation() -> None:
    out_dir = OUT_ROOT / "geo_ablation"
    keep: set[str] = set()
    homos = s02_homos()

    for stem, t_min, t_distant in GEO_SWEEPS:
        keep.add(f"{stem}.yaml")
        modern_run(
            group="geo_ablation",
            name=stem,
            homos=homos,
            association=association_block(t_min_m=t_min, t_distant_m=t_distant),
        )

    modern_run(
        group="geo_ablation",
        name="contact_point_world_bottom_center",
        homos=homos,
        association=association_block(t_min_m=14.0, t_distant_m=38.0),
        world_anchor="bottom_center",
        header="# Anchor comparison: bottom_center (no learned contact point).\n\n",
    )
    keep.add("contact_point_world_bottom_center.yaml")

    for stem, gap, mid_pen in GEO_TIGHT_TEMPORAL:
        keep.add(f"{stem}.yaml")
        strict = "strict" in stem
        mode = "strict" if strict else "penalty_only"
        modern_run(
            group="geo_ablation",
            name=stem,
            homos=homos,
            association=association_block(
                t_min_m=10.0,
                t_distant_m=25.0,
                temporal_mode=mode,
                temporal_gap=gap,
                temporal_mid_penalty=0.0 if strict else mid_pen,
            ),
        )

    prune_stale(out_dir, keep)


def generate_baseline_trackers() -> None:
    out_dir = OUT_ROOT / "baseline_trackers"
    keep: set[str] = set()

    no_reid = (
        ("sort", "sort", {"det_thresh": 0.3, "max_age": 30, "min_hits": 5, "iou_threshold": 0.3}),
        ("botsort", "botsort", {"det_thresh": 0.3, "max_age": 30, "min_hits": 5, "iou_threshold": 0.3}),
    )
    for name, tr_type, extra in no_reid:
        keep.add(f"{name}.yaml")
        legacy_run(
            group="baseline_trackers",
            name=name,
            tracker_type=tr_type,
            extra_tracker=extra,
        )

    keep.add("ocsort.yaml")
    legacy_run(
        group="baseline_trackers",
        name="ocsort",
        tracker={
            "type": "deepocsort",
            "use_embeddings": False,
            "device": 0,
            "half": False,
            "det_thresh": 0.3,
            "max_age": 30,
            "min_hits": 5,
            "iou_threshold": 0.3,
        },
        header="# CityFlow S02: DeepOcSort without ReID (OC-SORT mode).\n\n",
    )

    for name, extra in (
        ("deepocsort", {}),
        ("deepocsort_byte", {"use_byte": True}),
    ):
        keep.add(f"{name}.yaml")
        tr = copy.deepcopy(TRACKER_LEGACY)
        tr["appearance_update"] = "aaf"
        tr["reid_accum_conf_thresh"] = 0.5
        tr.update(extra)
        legacy_run(group="baseline_trackers", name=name, tracker=tr)

    prune_stale(out_dir, keep)


def generate_byte_ablation() -> None:
    out_dir = OUT_ROOT / "byte_ablation"
    keep: set[str] = set()
    specs = {
        "byte_off": {"use_byte": False, "det_thresh": 0.3},
        "byte_on_det02": {"use_byte": True, "min_conf": 0.1, "det_thresh": 0.2},
        "byte_on_det03": {"use_byte": True, "min_conf": 0.1, "det_thresh": 0.3},
        "byte_on_narrow": {"use_byte": True, "min_conf": 0.15, "det_thresh": 0.25},
        "byte_smoke": {"use_byte": True, "min_conf": 0.1, "det_thresh": 0.2},
    }
    for name, extra in specs.items():
        keep.add(f"{name}.yaml")
        tr = copy.deepcopy(TRACKER_LEGACY)
        tr.update(extra)
        legacy_run(group="byte_ablation", name=name, tracker=tr)
    prune_stale(out_dir, keep)


def generate_conf_ablation() -> None:
    out_dir = OUT_ROOT / "conf_ablation"
    keep: set[str] = set()
    for conf in CONF_VALUES:
        tag = _conf_tag(conf)
        name = f"conf_{tag}"
        keep.add(f"{name}.yaml")
        det = copy.deepcopy(DETECTOR_YOLO26L_640)
        det["conf_thres"] = conf
        tr = copy.deepcopy(TRACKER_LEGACY)
        tr["det_thresh"] = conf
        tr["reid_accum_conf_thresh"] = conf
        tr["min_hits"] = 3
        rel = results_rel("conf_ablation", name)
        cfg = {
            "run_mode": "multi_camera",
            "detector": det,
            "tracker": tr,
            "multi_camera": {
                **multi_camera_block(homos=None, results_rel_path=rel, legacy=False, max_frames=None),
                "association": association_block(
                    t_distant_m=0.0,
                    reid_cost_threshold=0.25,
                    use_gps=False,
                ),
                "max_history_gap_frames": 30,
            },
            "output": output_block(rel),
        }
        write_yaml(
            out_dir / f"{name}.yaml",
            cfg,
            header=f"# CityFlow conf_thres={conf}\n\n",
        )
    prune_stale(out_dir, keep)


def generate_ema_vs_aaf() -> None:
    out_dir = OUT_ROOT / "ema_vs_aaf"
    keep: set[str] = set()
    for name, mode, thresh in (
        ("ema", "ema", 0.5),
        ("aaf", "aaf", 0.5),
        ("aaf_strict", "aaf", 0.6),
    ):
        keep.add(f"{name}.yaml")
        tr = copy.deepcopy(TRACKER_LEGACY)
        tr["appearance_update"] = mode
        tr["reid_accum_conf_thresh"] = thresh
        legacy_run(group="ema_vs_aaf", name=name, tracker=tr)
    prune_stale(out_dir, keep)


def generate_latency_ablation() -> None:
    out_dir = OUT_ROOT / "latency_ablation"
    keep: set[str] = set()
    specs = {
        "batch_640": {"detector": {"batch_inference": True}, "tracker": {"share_reid_model": True, "batch_reid": False}},
        "batch_640_reid": {"detector": {"batch_inference": True}, "tracker": {"share_reid_model": True, "batch_reid": True}},
        "batch_960": {"detector": {"batch_inference": True, "imgsz": 960}, "tracker": {"share_reid_model": True, "batch_reid": False}},
        "seq_960": {"detector": {"imgsz": 960}, "tracker": {}},
    }
    for name, patches in specs.items():
        keep.add(f"{name}.yaml")
        det = copy.deepcopy(DETECTOR_YOLO26L_640)
        det.update(patches.get("detector", {}))
        tr = copy.deepcopy(TRACKER_LEGACY)
        tr.update(patches.get("tracker", {}))
        legacy_run(group="latency_ablation", name=name, detector=det, tracker=tr)
    prune_stale(out_dir, keep)


def generate_zone_tracklet() -> None:
    rel = "configs_cityflow/zone_tracklet"
    tr = copy.deepcopy(TRACKER_LEGACY)
    tr["min_hits"] = 3
    cfg = {
        "run_mode": "multi_camera",
        "detector": copy.deepcopy(DETECTOR_YOLO26L_640),
        "tracker": tr,
        "multi_camera": {
            **multi_camera_block(homos=None, results_rel_path=rel),
            "association": association_block(
                t_distant_m=0.0,
                zone_transition=True,
                zones_path=ZONES_PATH,
                reid_cost_threshold=0.25,
                use_gps=False,
            ),
            "max_history_gap_frames": 30,
        },
        "output": output_block(rel),
    }
    header = "# CityFlow S02 standalone zone tracklet run.\n\n"
    write_yaml(OUT_ROOT / "zone_tracklet.yaml", cfg, header=header)


def generate_sort_yolo26l() -> None:
    out_dir = OUT_ROOT / "sort"
    out_dir.mkdir(parents=True, exist_ok=True)
    legacy_run(
        group="sort",
        name="yolo26l",
        tracker_type="sort",
        extra_tracker={"det_thresh": 0.3, "max_age": 30, "min_hits": 5, "iou_threshold": 0.3},
        header="# CityFlow SORT + YOLO26l\n\n",
    )


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    generate_baseline()
    generate_assoc_ablation()
    generate_reid_ablation()
    generate_temporal_ablation()
    generate_kinematic_ablation()
    generate_trajectory_ablation()
    generate_geo_ablation()
    generate_baseline_trackers()
    generate_byte_ablation()
    generate_conf_ablation()
    generate_ema_vs_aaf()
    generate_latency_ablation()
    generate_zone_tracklet()
    generate_sort_yolo26l()

    n = sum(1 for _ in OUT_ROOT.rglob("*.yaml"))
    print(f"Wrote {n} configs under {OUT_ROOT.relative_to(ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
