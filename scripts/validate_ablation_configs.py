"""Check ablation YAML configs and referenced paths exist."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

CONFIGS: list[str] = [
    "config/reid_ablation/osnet_ibn_msmt17.yaml",
    "config/reid_ablation/vehicle_osnet_veri_vric.yaml",
    "config/reid_ablation/vehicle_osnet_view_finetune.yaml",
    "config/baseline_trackers/sort.yaml",
    "config/baseline_trackers/ocsort.yaml",
    "config/baseline_trackers/deepocsort.yaml",
    "config/baseline_trackers/botsort.yaml",
    "config/ema_vs_aaf/ema.yaml",
    "config/ema_vs_aaf/aaf.yaml",
    "config/s02_baseline.yaml",
    *[f"config/s02_baseline/deepocsort/detectors/{m}.yml"
      for m in ("yolov8s", "yolov8m", "yolov8l", "yolov8x",
                "yolo26m", "yolo26l", "yolo26x", "rtdetr_l")],
    *[f"config/s02_baseline/ocsort/detectors/{m}.yml"
      for m in ("yolov8s", "yolov8m", "yolov8l", "yolov8x",
                "yolo26m", "yolo26l", "yolo26x", "rtdetr_l")],
    *[f"config/geo_ablation/vehicle_view_geo_{d}.yaml" for d in (3, 5, 9)],
]

missing_cfg = [c for c in CONFIGS if not (ROOT / c).exists()]
if missing_cfg:
    print("MISSING CONFIGS:")
    for c in missing_cfg:
        print(f"  {c}")
else:
    print(f"All {len(CONFIGS)} config files present.")

refs: set[str] = set()
for c in CONFIGS:
    path = ROOT / c
    if not path.exists():
        continue
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    det = cfg.get("detector") or {}
    tr = cfg.get("tracker") or {}
    if det.get("model"):
        refs.add(str(det["model"]))
    if tr.get("reid_weights"):
        refs.add(str(tr["reid_weights"]))
    mc = cfg.get("multi_camera") or {}
    for src in mc.get("sources") or []:
        refs.add(str(src))
    for homo in mc.get("homos") or []:
        refs.add(str(homo))

print("\nReferenced assets:")
missing_assets: list[str] = []
for r in sorted(refs):
    ok = (ROOT / r).exists()
    tag = "OK" if ok else "MISSING"
    print(f"  [{tag}] {r}")
    if not ok:
        missing_assets.append(r)

if missing_assets:
    print(f"\n{len(missing_assets)} asset(s) missing — ablations will fail until resolved.")
    raise SystemExit(1)

print("\nValidation passed.")
