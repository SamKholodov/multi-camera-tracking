"""Generate geo_tight + temporal tuning configs from GT stats.

Usage:
    python scripts/generate_geo_tight_temporal_configs.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs_gta" / "geo_ablation" / "geo_tight.yaml"
STATS = ROOT / "outputs" / "configs_gta" / "temporal_gt_stats.json"
OUT = ROOT / "configs_gta" / "geo_ablation"


def _load_stats() -> dict:
    if STATS.is_file():
        return json.loads(STATS.read_text(encoding="utf-8"))
    return {"suggested": {"max_cross_cam_gap_frames": {"penalty_only_p90": 44, "strict_p95": 104}}}


def _apply_temporal(cfg: dict, *, mode: str, gap: int, penalty: float, stem: str) -> dict:
    cfg = copy.deepcopy(cfg)
    assoc = cfg["multi_camera"]["association"]
    assoc["gates"]["temporal"] = mode != "off"
    assoc["temporal_mode"] = mode
    assoc["max_cross_cam_gap_frames"] = int(gap)
    assoc["temporal_mid_penalty"] = float(penalty)
    rel = f"configs_gta/geo_ablation/{stem}"
    cfg["multi_camera"]["results_dir"] = f"outputs/{rel}"
    out = cfg.setdefault("output", {})
    out["save_video_dir"] = f"outputs/{rel}/videos"
    out["output_path"] = f"outputs/{rel}/multicam.mp4"
    return cfg


def main() -> None:
    stats = _load_stats()
    sug = stats.get("suggested", {})
    gaps = sug.get("max_cross_cam_gap_frames", {})
    n_penalty = int(gaps.get("penalty_only_p90", 44))
    n_strict = int(gaps.get("strict_p95", 104))
    penalties = sug.get("temporal_mid_penalty_recommended", [0.15, 0.25, 0.35])

    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    written: list[str] = []

    for pen in penalties:
        tag = str(pen).replace(".", "")
        stem = f"geo_tight_temporal_N{n_penalty}_p{tag}"
        cfg = _apply_temporal(
            base,
            mode="penalty_only",
            gap=n_penalty,
            penalty=pen,
            stem=stem,
        )
        path = OUT / f"{stem}.yaml"
        header = (
            f"# geo_tight + temporal penalty_only (GT p90 N={n_penalty}, penalty={pen}).\n"
            f"# Stats: outputs/configs_gta/temporal_gt_stats.json\n\n"
        )
        path.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        written.append(path.name)

    strict_stem = f"geo_tight_temporal_strict_N{n_strict}"
    strict_cfg = _apply_temporal(
        base,
        mode="strict",
        gap=n_strict,
        penalty=0.0,
        stem=strict_stem,
    )
    strict_path = OUT / f"{strict_stem}.yaml"
    strict_path.write_text(
        "# geo_tight + temporal strict (GT p95 N={n}). Run after penalty sweep.\n\n".format(
            n=n_strict
        )
        + yaml.safe_dump(strict_cfg, sort_keys=False),
        encoding="utf-8",
    )
    written.append(strict_path.name)

    print("Wrote:", ", ".join(written))
    print(f"  penalty N={n_penalty} (~{n_penalty / 30:.1f}s), penalties={penalties}")
    print(f"  strict  N={n_strict} (~{n_strict / 30:.1f}s)")


if __name__ == "__main__":
    main()
