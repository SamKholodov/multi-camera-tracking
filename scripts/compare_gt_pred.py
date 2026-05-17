"""Quick GT vs prediction comparison for S02."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.io.roi import ROIFilter


def load_mot(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 10))
    data = np.loadtxt(str(path), delimiter=",")
    if data.size == 0:
        return np.empty((0, 10))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def stats(data: np.ndarray, name: str) -> dict:
    if len(data) == 0:
        return {"name": name, "rows": 0}
    frames = data[:, 0].astype(int)
    ids = data[:, 1].astype(int)
    keys = list(zip(frames, ids))
    uframes = np.unique(frames)
    per_frame = [np.sum(frames == f) for f in uframes]
    return {
        "name": name,
        "rows": len(data),
        "unique_ids": len(set(ids.tolist())),
        "id_min": int(ids.min()),
        "id_max": int(ids.max()),
        "frames": len(uframes),
        "fmin": int(frames.min()),
        "fmax": int(frames.max()),
        "bpf_mean": float(np.mean(per_frame)),
        "bpf_max": int(np.max(per_frame)),
        "dups": len(keys) - len(set(keys)),
    }


def main():
    root = Path("datasets/validation/S02")
    pred_g = Path("outputs/s02_baseline/per_cam")
    pred_l = Path("outputs/s02_baseline/per_cam_local")

    print("=" * 78)
    print("GT (dataset) vs predictions")
    print("=" * 78)

    for cam in [6, 7, 8, 9]:
        gt_path = root / f"c{cam:03d}" / "gt" / "gt.txt"
        gt = load_mot(gt_path)
        pr = load_mot(pred_g / f"c{cam:03d}.txt")
        pl = load_mot(pred_l / f"c{cam:03d}.txt")

        roi_path = root / f"c{cam:03d}" / "roi.jpg"
        roi = ROIFilter.from_path(roi_path)
        gt_roi = roi.filter_mot(gt)
        pr_roi = roi.filter_mot(pr)

        print(f"\n--- c{cam:03d} ---")
        for s in [
            stats(gt, "GT all"),
            stats(gt_roi, "GT + ROI"),
            stats(pr, "PR global"),
            stats(pr_roi, "PR global + ROI"),
            stats(pl, "PR local"),
        ]:
            if s["rows"] == 0:
                print(f"  {s['name']}: empty")
                continue
            print(
                f"  {s['name']:18} rows={s['rows']:6}  ids={s['unique_ids']:4}  "
                f"id=[{s['id_min']}..{s['id_max']}]  frames={s['frames']:4} "
                f"({s['fmin']}-{s['fmax']})  box/frame={s['bpf_mean']:.2f} "
                f"max={s['bpf_max']}  dups={s['dups']}"
            )

        gt_n = len(gt_roi)
        pr_n = len(pr_roi)
        print(
            f"  => PR/GT rows (ROI): {pr_n}/{gt_n} = {pr_n / max(gt_n, 1):.2f}x  "
            f"(eval GT column ~unique objects, not rows)"
        )

    print("\n" + "=" * 78)
    print("Your terminal metrics (per_cam + --apply-roi) — reference")
    print("=" * 78)
    ref = {
        6: {"idf1": 9.9, "fp": 20391, "fn": 1884, "gt_obj": 123},
        7: {"idf1": 33.3, "fp": 1084, "fn": 277, "gt_obj": 90},
        8: {"idf1": 6.5, "fp": 14836, "fn": 2087, "gt_obj": 98},
        9: {"idf1": 16.1, "fp": 22083, "fn": 4747, "gt_obj": 136},
    }
    for cam, m in ref.items():
        prec = m["fp"] / max(m["fp"] + m["gt_obj"], 1)  # rough
        print(
            f"  c{cam:03d}: IDF1={m['idf1']}%  FP={m['fp']}  FN={m['fn']}  "
            f"GT_objects={m['gt_obj']}"
        )


if __name__ == "__main__":
    main()
