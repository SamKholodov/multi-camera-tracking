"""Apply a chosen detector confidence threshold across GTA ablation configs.

Updates detector.conf_thres and aligns tracker.det_thresh / reid_accum_conf_thresh
when those fields are present.

Usage:
    python scripts/apply_gta_conf_thres.py --conf 0.3
    python scripts/apply_gta_conf_thres.py --conf 0.3 --dry-run
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = _ROOT / "configs_gta"

SKIP_DIR_PARTS = {
    "conf_ablation",
    "detector_ablation",
    "detectors",
}


def _should_skip(path: Path) -> bool:
    rel = path.relative_to(CONFIG_ROOT)
    parts = set(rel.parts)
    if "conf_ablation" in parts:
        return True
    if "detectors" in parts:
        return True
    return False


def _collect_configs() -> list[Path]:
    files: list[Path] = []
    for path in sorted(CONFIG_ROOT.rglob("*")):
        if path.suffix not in {".yaml", ".yml"}:
            continue
        if _should_skip(path):
            continue
        files.append(path)
    return files


def _patch_config(cfg: dict, conf: float) -> tuple[dict, list[str]]:
    cfg = copy.deepcopy(cfg)
    changes: list[str] = []

    detector = cfg.get("detector")
    if isinstance(detector, dict) and "conf_thres" in detector:
        old = detector.get("conf_thres")
        detector["conf_thres"] = conf
        changes.append(f"detector.conf_thres: {old} -> {conf}")

    tracker = cfg.get("tracker")
    if isinstance(tracker, dict):
        if "det_thresh" in tracker:
            old = tracker.get("det_thresh")
            tracker["det_thresh"] = conf
            changes.append(f"tracker.det_thresh: {old} -> {conf}")
        if "reid_accum_conf_thresh" in tracker:
            old = tracker.get("reid_accum_conf_thresh")
            tracker["reid_accum_conf_thresh"] = conf
            changes.append(f"tracker.reid_accum_conf_thresh: {old} -> {conf}")

    return cfg, changes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conf", type=float, required=True, help="Detector confidence threshold")
    ap.add_argument("--dry-run", action="store_true", help="Print changes without writing files")
    args = ap.parse_args()

    if not (0.0 < args.conf <= 1.0):
        raise SystemExit("--conf must be in (0, 1].")

    updated = 0
    for path in _collect_configs():
        raw = path.read_text(encoding="utf-8")
        cfg = yaml.safe_load(raw) or {}
        patched, changes = _patch_config(cfg, args.conf)
        if not changes:
            continue

        rel = path.relative_to(_ROOT).as_posix()
        print(f"{rel}")
        for line in changes:
            print(f"  {line}")

        if not args.dry_run:
            header_lines: list[str] = []
            for line in raw.splitlines():
                if line.startswith("#"):
                    header_lines.append(line)
                else:
                    break
            header = "\n".join(header_lines)
            if header:
                header += "\n\n"
            body = yaml.dump(patched, sort_keys=False, allow_unicode=True)
            path.write_text(header + body, encoding="utf-8")
        updated += 1

    action = "Would update" if args.dry_run else "Updated"
    print(f"\n{action} {updated} config(s) with conf_thres={args.conf:.2f}")
    if not args.dry_run and updated:
        print("Next: .\\scripts\\run_gta_ablations.ps1")


if __name__ == "__main__":
    main()
