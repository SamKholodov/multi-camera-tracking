"""Generate GTA geometry ablation configs from temporal_off baseline.

Variants:
  Anchor comparison (baseline thresholds 14/38 m):
    contact_point_world_bottom_center — world_anchor=bottom_center
    contact_point_world_contact       — world_anchor=contact_point (learned)
  Threshold sweep (bottom_center anchor, plane metric):
    geo_reid_only, geo_tight, geo_baseline, geo_loose, geo_very_loose

All: temporal off, UF/speed/trajectory off, Wild epoch-120, max_frames=null, fps=30.

Usage:
    python scripts/generate_geo_ablation_configs.py
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs_gta" / "temporal_ablation" / "temporal_off.yaml"
OUT = ROOT / "configs_gta" / "geo_ablation"

BASELINE_T_MIN = 14.0
BASELINE_T_DISTANT = 38.0

CONTACT_POINT_BLOCK = {
    "enabled": True,
    "weights": "runs/contact_point/mobilenetv3_small_gta_points/best.pth",
    "conf_thresh": 0.2,
    "batch_size": 256,
    "device": "cuda:0",
}

# (filename stem, t_min_m, t_distant_m) — bottom_center anchor only
THRESHOLD_SWEEPS: list[tuple[str, float, float]] = [
    ("geo_tight", 10.0, 25.0),
    ("geo_baseline", BASELINE_T_MIN, BASELINE_T_DISTANT),
    ("geo_loose", BASELINE_T_MIN, 55.0),
]


def _base_config() -> dict:
    cfg = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    mc = cfg["multi_camera"]
    assoc = mc.setdefault("association", {})
    assoc.setdefault("gates", {})["temporal"] = False
    assoc["temporal_mode"] = "off"
    assoc["same_frame_linking"] = False
    assoc["speed_limit_enabled"] = False
    assoc["trajectory_enabled"] = False
    assoc.setdefault("global_delete_after_frames", 600)
    mc["max_frames"] = None
    mc.pop("results_dir", None)
    cfg.pop("contact_point", None)
    out = cfg.setdefault("output", {})
    out.pop("save_video_dir", None)
    out.pop("output_path", None)
    return cfg


def _apply_geo(cfg: dict, t_min_m: float, t_distant_m: float) -> dict:
    assoc = cfg["multi_camera"]["association"]
    assoc["geometry_t_min_m"] = float(t_min_m)
    assoc["geometry_t_distant_m"] = float(t_distant_m)
    return cfg


def _anchor_bottom_center(t_min_m: float, t_distant_m: float) -> dict:
    cfg = _base_config()
    cfg["world_anchor"] = "bottom_center"
    return _apply_geo(cfg, t_min_m, t_distant_m)


def _anchor_contact_point(t_min_m: float, t_distant_m: float) -> dict:
    cfg = _anchor_bottom_center(t_min_m, t_distant_m)
    cfg["world_anchor"] = "contact_point"
    cfg["contact_point"] = dict(CONTACT_POINT_BLOCK)
    return cfg


def _write(cfg: dict, out_path: Path) -> None:
    rel = out_path.relative_to(ROOT).with_suffix("").as_posix()
    mc = cfg["multi_camera"]
    mc["results_dir"] = f"outputs/{rel}"
    out = cfg.setdefault("output", {})
    out["save_video_dir"] = f"outputs/{rel}/videos"
    out["output_path"] = f"outputs/{rel}/multicam.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    written: list[str] = []

    _write(
        _anchor_bottom_center(BASELINE_T_MIN, BASELINE_T_DISTANT),
        OUT / "contact_point_world_bottom_center.yaml",
    )
    written.append("contact_point_world_bottom_center")

    _write(
        _anchor_contact_point(BASELINE_T_MIN, BASELINE_T_DISTANT),
        OUT / "contact_point_world_contact.yaml",
    )
    written.append("contact_point_world_contact")

    for name, t_min, t_distant in THRESHOLD_SWEEPS:
        _write(_anchor_bottom_center(t_min, t_distant), OUT / f"{name}.yaml")
        written.append(name)

    keep = {f"{n}.yaml" for n in written}
    for stale in OUT.glob("*.yaml"):
        if stale.name not in keep:
            stale.unlink(missing_ok=True)

    print(f"Wrote {len(written)} configs to {OUT.relative_to(ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
