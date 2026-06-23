"""Generate GTA temporal/kinematic ablation configs (full ~10k frames, Wild epoch 120, UF off)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs_gta" / "temporal_ablation" / "temporal_off.yaml"
# GTA MCMT synced JPG sequences are exported at game capture rate (~30 fps).
# S02/CityFlow configs use 10 fps; do not reuse that default here.
GTA_VIDEO_FPS = 30.0
# null = full dataset (~9999 sync frames). Use 1000 for quick smoke.
GTA_MAX_FRAMES = None
OUT_TEMPORAL = ROOT / "configs_gta" / "temporal_ablation"
OUT_KINEMATIC = ROOT / "configs_gta" / "kinematic_ablation"
OUT_TRAJECTORY = ROOT / "configs_gta" / "trajectory_ablation"
# Reduced overnight sweep (fps=30): ~2s / ~5s / ~10s cross-cam gaps.
GAPS = (60, 150, 300)
SPEEDS = (25, 35)
TRAJECTORY_KS = (1, 3)
TRAJECTORY_THRESHOLDS = {
    1: 0.75,
    3: 0.07,
}


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
    assoc["video_fps"] = GTA_VIDEO_FPS
    mc["video_fps"] = GTA_VIDEO_FPS
    mc["max_frames"] = GTA_MAX_FRAMES
    mc.pop("results_dir", None)
    out = cfg.setdefault("output", {})
    out["video_fps"] = GTA_VIDEO_FPS
    out.pop("save_video_dir", None)
    out.pop("output_path", None)
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


def _temporal_config(mode: str, gap: int | None = None) -> dict:
    cfg = _base_config()
    assoc = cfg["multi_camera"]["association"]
    assoc["temporal_mode"] = mode
    assoc["gates"]["temporal"] = mode != "off"
    if gap is not None:
        assoc["max_cross_cam_gap_frames"] = int(gap)
    assoc["global_delete_after_frames"] = 600
    return cfg


def _prune_stale(out_dir: Path, keep: set[str]) -> None:
    for stale in out_dir.glob("*.yaml"):
        if stale.name not in keep:
            stale.unlink(missing_ok=True)


def main() -> None:
    temporal_keep: set[str] = {"temporal_off.yaml"}
    _write(_temporal_config("off"), OUT_TEMPORAL / "temporal_off.yaml")
    for gap in GAPS:
        name = f"temporal_penalty_N{gap}.yaml"
        temporal_keep.add(name)
        _write(
            _temporal_config("penalty_only", gap),
            OUT_TEMPORAL / name,
        )
    _prune_stale(OUT_TEMPORAL, temporal_keep)

    kinematic_keep: set[str] = set()
    for speed in SPEEDS:
        name = f"speed_penalty_v{speed}.yaml"
        kinematic_keep.add(name)
        cfg = _temporal_config("off")
        assoc = cfg["multi_camera"]["association"]
        assoc["speed_limit_enabled"] = True
        assoc["speed_limit_mode"] = "penalty"
        assoc["speed_v_max_mps"] = float(speed)
        _write(cfg, OUT_KINEMATIC / name)
    _prune_stale(OUT_KINEMATIC, kinematic_keep)

    trajectory_keep: set[str] = set()
    for k in TRAJECTORY_KS:
        name = f"traj_linear_K{k}.yaml"
        trajectory_keep.add(name)
        cfg = _temporal_config("off")
        assoc = cfg["multi_camera"]["association"]
        assoc["trajectory_enabled"] = True
        assoc["trajectory_mode"] = "linear"
        assoc["trajectory_history_k"] = int(k)
        assoc["trajectory_threshold_m"] = TRAJECTORY_THRESHOLDS[k]
        _write(cfg, OUT_TRAJECTORY / name)
    _prune_stale(OUT_TRAJECTORY, trajectory_keep)


if __name__ == "__main__":
    main()
