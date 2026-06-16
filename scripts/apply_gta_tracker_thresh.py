"""Apply tracker det_thresh and reid_accum_conf_thresh across GTA ablation configs."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = _ROOT / "configs_gta"

SKIP_PARTS = {"conf_ablation", "detector_ablation", "detectors"}


def _should_skip(path: Path) -> bool:
    return bool(SKIP_PARTS.intersection(path.relative_to(CONFIG_ROOT).parts))


def _collect_configs() -> list[Path]:
    files: list[Path] = []
    for path in sorted(CONFIG_ROOT.rglob("*")):
        if path.suffix not in {".yaml", ".yml"}:
            continue
        if _should_skip(path):
            continue
        files.append(path)
    return files


def _patch_config(cfg: dict, det_thresh: float, reid_accum: float) -> tuple[dict, list[str]]:
    cfg = copy.deepcopy(cfg)
    changes: list[str] = []
    tracker = cfg.get("tracker")
    if not isinstance(tracker, dict):
        return cfg, changes
    if "det_thresh" in tracker:
        old = tracker.get("det_thresh")
        tracker["det_thresh"] = det_thresh
        changes.append(f"tracker.det_thresh: {old} -> {det_thresh}")
    if "reid_accum_conf_thresh" in tracker:
        old = tracker.get("reid_accum_conf_thresh")
        tracker["reid_accum_conf_thresh"] = reid_accum
        changes.append(f"tracker.reid_accum_conf_thresh: {old} -> {reid_accum}")
    return cfg, changes


def _write_config(path: Path, raw: str, cfg: dict) -> None:
    header_lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith("#"):
            header_lines.append(line)
        else:
            break
    header = "\n".join(header_lines)
    if header:
        header += "\n\n"
    path.write_text(header + yaml.dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det-thresh", type=float, default=0.3)
    ap.add_argument("--reid-accum-conf", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    updated = 0
    for path in _collect_configs():
        raw = path.read_text(encoding="utf-8")
        cfg = yaml.safe_load(raw) or {}
        patched, changes = _patch_config(cfg, args.det_thresh, args.reid_accum_conf)
        if not changes:
            continue
        rel = path.relative_to(_ROOT).as_posix()
        print(rel)
        for line in changes:
            print(f"  {line}")
        if not args.dry_run:
            _write_config(path, raw, patched)
        updated += 1

    action = "Would update" if args.dry_run else "Updated"
    print(
        f"\n{action} {updated} config(s): "
        f"det_thresh={args.det_thresh}, reid_accum_conf_thresh={args.reid_accum_conf}"
    )


if __name__ == "__main__":
    main()
