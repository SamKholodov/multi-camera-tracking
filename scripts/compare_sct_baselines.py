"""Compare user per_cam_local SCT vs dataset MTSC baselines (eval_s02 + ROI)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

import motmetrics as mm

from core.io.roi import ROIFilter
from scripts.eval_s02 import _accumulate, _load_mot

GT_ROOT = Path("datasets/validation/S02")
CAMS = [6, 7, 8, 9]


def _eval_paths(get_pr_path) -> tuple[mm.MOTAccumulator, dict, "pd.Series"]:
    accs = {}
    for cam in CAMS:
        gt = _load_mot(GT_ROOT / f"c{cam:03d}/gt/gt.txt")
        pr = _load_mot(get_pr_path(cam))
        roi = GT_ROOT / f"c{cam:03d}/roi.jpg"
        if roi.exists():
            f = ROIFilter.from_path(roi)
            gt, pr = f.filter_mot(gt), f.filter_mot(pr)
        accs[f"c{cam:03d}"] = _accumulate(gt, pr)
    mh = mm.metrics.create()
    summ = mh.compute_many(
        list(accs.values()),
        names=list(accs.keys()),
        metrics=["idf1", "idp", "idr", "mota"],
        generate_overall=True,
    )
    return summ.loc["OVERALL"], {n: summ.loc[n] for n in accs}


def main():
    rows = []

    for label, pdir in [
        ("YOU: YOLOv8l+DeepOCSORT", _ROOT / "outputs/s02_baseline/per_cam_local"),
        ("YOU: MaskRCNN det+DeepOCSORT", _ROOT / "outputs/s02_maskrcnn_det/per_cam_local"),
    ]:
        if not pdir.is_dir():
            continue
        ov, per = _eval_paths(lambda cam, d=pdir: d / f"c{cam:03d}.txt")
        rows.append((label, ov, per))

    mtsc_dir = GT_ROOT / "c006" / "mtsc"
    for p in sorted(mtsc_dir.glob("mtsc_*.txt")):
        name = "MTSC: " + p.stem.replace("mtsc_", "")
        ov, per = _eval_paths(
            lambda cam, bn=p.name: GT_ROOT / f"c{cam:03d}/mtsc" / bn
        )
        rows.append((name, ov, per))

    rows.sort(key=lambda x: -float(x[1]["idf1"]))

    print("=== Single-camera SCT (eval_s02.py, ROI, cameras 6-9) ===\n")
    hdr = f"{'method':34} {'IDF1':>6} {'IDP':>6} {'IDR':>6} {'MOTA':>7}  | per-cam IDF1"
    print(hdr)
    print("-" * len(hdr))
    for name, ov, per in rows:
        cam_idf1 = " ".join(
            f"c{c:03d}={float(per[f'c{c:03d}']['idf1']) * 100:4.0f}%"
            for c in CAMS
        )
        print(
            f"{name:34} {float(ov['idf1']) * 100:6.1f} {float(ov['idp']) * 100:6.1f} "
            f"{float(ov['idr']) * 100:6.1f} {float(ov['mota']) * 100:7.1f}  | {cam_idf1}"
        )


if __name__ == "__main__":
    main()
