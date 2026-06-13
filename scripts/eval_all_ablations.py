"""Batch CityFlow / MOT re-evaluation for all ablation output folders."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eval_s02 import BATCH_METRICS, evaluate_s02

GT_ROOT = Path("datasets/validation/S02")
CAMS = [6, 7, 8, 9]

TRACKERS = ["sort", "ocsort", "deepocsort", "botsort"]
EMA_AAF = ["ema", "aaf"]


def _has_predictions(pred_dir: Path) -> bool:
    return (pred_dir / "c006.txt").is_file()


def _scalar(value) -> float:
    if hasattr(value, "iloc"):
        return float(value.iloc[0])
    return float(value)


def _metrics_row(result: dict, stream: str) -> dict[str, float]:
    if stream == "mcmt":
        row = result["mcmt"]
        if row is None:
            return {m: float("nan") for m in BATCH_METRICS}
        return {m: _scalar(row[m]) for m in BATCH_METRICS}
    return {m: _scalar(result["per_cam"].loc["OVERALL", m]) for m in BATCH_METRICS}


def _eval_run(
    group: str,
    name: str,
    pred_subdir: str,
    pred_root: Path,
    cityflow_protocol: bool,
    id_mode: str,
) -> dict | None:
    pred_dir = pred_root / pred_subdir
    if not _has_predictions(pred_dir):
        print(f"[SKIP] {group}/{name}/{pred_subdir}: missing c006.txt")
        return None
    try:
        result = evaluate_s02(
            GT_ROOT,
            pred_dir,
            cameras=CAMS,
            cityflow_protocol=cityflow_protocol,
            pred_id_mode=id_mode,  # type: ignore[arg-type]
        )
    except SystemExit as exc:
        print(f"[SKIP] {group}/{name}/{pred_subdir}: {exc}")
        return None

    stream = "mcmt" if pred_subdir == "per_cam" else "sct"
    metrics = _metrics_row(result, stream)
    return {
        "group": group,
        "name": name,
        "stream": stream,
        "pred_dir": str(pred_dir),
        **{m: metrics[m] for m in BATCH_METRICS},
    }


def _collect_detector_runs(base: Path, group: str) -> list[dict]:
    rows: list[dict] = []
    if not base.is_dir():
        print(f"[SKIP] detector base missing: {base}")
        return rows
    for det_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for sub, mode in [("per_cam_local", "local"), ("per_cam", "global")]:
            row = _eval_run(group, det_dir.name, sub, det_dir, True, mode)
            if row:
                rows.append(row)
    return rows


def _collect_fixed_runs(
    group: str,
    names: list[str],
    root: Path,
    cityflow_protocol: bool,
) -> list[dict]:
    rows: list[dict] = []
    for name in names:
        run_root = root / name
        for sub, mode in [("per_cam_local", "local"), ("per_cam", "global")]:
            row = _eval_run(group, name, sub, run_root, cityflow_protocol, mode)
            if row:
                rows.append(row)
    return rows


def _print_group_table(title: str, rows: list[dict], stream: str):
    subset = [r for r in rows if r["stream"] == stream]
    if not subset:
        return
    print(f"\n=== {title} ({stream.upper()}) ===")
    print(f"{'name':16s} {'IDF1':>7s} {'IDP':>7s} {'IDR':>7s} {'MOTA':>8s} {'FP':>7s} {'FN':>7s}")
    for r in sorted(subset, key=lambda x: -(x["idf1"] if x["idf1"] == x["idf1"] else -1)):
        fp = r["num_false_positives"]
        fn = r["num_misses"]
        fp_s = f"{int(fp):7d}" if fp == fp else "    nan"
        fn_s = f"{int(fn):7d}" if fn == fn else "    nan"
        print(
            f"{r['name']:16s} {r['idf1']*100:6.1f}% {r['idp']*100:6.1f}% {r['idr']*100:6.1f}% "
            f"{r['mota']*100:7.1f}% {fp_s} {fn_s}"
        )


def _save_results(rows: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"summary_{stamp}.csv"
    json_path = out_dir / f"summary_{stamp}.json"

    fieldnames = ["group", "name", "stream", "pred_dir", *BATCH_METRICS]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cityflow-protocol",
        action="store_true",
        default=True,
        help="CityFlow-aligned evaluation (default: on)",
    )
    ap.add_argument(
        "--full-mot",
        action="store_true",
        help="Standard MOT eval instead of CityFlow protocol",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/eval_cityflow"),
    )
    args = ap.parse_args()
    cityflow = args.cityflow_protocol and not args.full_mot
    proto_label = "CityFlow" if cityflow else "full MOT"

    all_rows: list[dict] = []

    tracker_rows = _collect_fixed_runs(
        "baseline_trackers",
        TRACKERS,
        Path("outputs/baseline_trackers"),
        cityflow,
    )
    all_rows.extend(tracker_rows)
    _print_group_table(f"baseline_trackers ({proto_label})", tracker_rows, "sct")
    _print_group_table(f"baseline_trackers ({proto_label})", tracker_rows, "mcmt")

    ema_rows = _collect_fixed_runs(
        "ema_vs_aaf",
        EMA_AAF,
        Path("outputs/ema_vs_aaf"),
        cityflow,
    )
    all_rows.extend(ema_rows)
    _print_group_table(f"ema_vs_aaf ({proto_label})", ema_rows, "sct")
    _print_group_table(f"ema_vs_aaf ({proto_label})", ema_rows, "mcmt")

    deep_rows = _collect_detector_runs(
        Path("outputs/s02_baseline/deepocsort/detectors"),
        "deepocsort_detectors",
    )
    all_rows.extend(deep_rows)
    _print_group_table(f"deepocsort detectors ({proto_label})", deep_rows, "sct")
    _print_group_table(f"deepocsort detectors ({proto_label})", deep_rows, "mcmt")

    ocsort_rows = _collect_detector_runs(
        Path("outputs/s02_baseline/ocsort/detectors"),
        "ocsort_detectors",
    )
    all_rows.extend(ocsort_rows)
    _print_group_table(f"ocsort detectors ({proto_label})", ocsort_rows, "sct")
    _print_group_table(f"ocsort detectors ({proto_label})", ocsort_rows, "mcmt")

    if all_rows:
        _save_results(all_rows, args.out_dir)
    else:
        print("\nNo runs evaluated (all skipped).")


if __name__ == "__main__":
    main()
