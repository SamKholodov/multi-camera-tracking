"""Aggregate latency ablation FPS reports into CSV and markdown."""
from __future__ import annotations

import csv
import json
from pathlib import Path


BASE = Path("outputs/latency_ablation")
VARIANTS = ["seq_960", "batch_960", "batch_640", "batch_640_reid"]
FIELDS = [
    "variant",
    "pipeline_fps",
    "ms_per_sync_frame",
    "speedup_vs_seq",
    "det_ms_sync",
    "track_ms_cam",
    "reid_ms_sync",
    "mcmt_ms",
]


def _read_report(variant: str) -> dict | None:
    path = BASE / variant / "fps.json"
    if not path.is_file():
        print(f"[SKIP] {variant}: missing {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _round(value, ndigits=3):
    if value is None:
        return ""
    return round(float(value), ndigits)


def _row(variant: str, report: dict, baseline_ms: float | None) -> dict:
    ms = float(report.get("ms_per_sync_frame", 0.0))
    speedup = (baseline_ms / ms) if baseline_ms and ms > 0 else None
    return {
        "variant": variant,
        "pipeline_fps": _round(report.get("pipeline_fps")),
        "ms_per_sync_frame": _round(ms, 2),
        "speedup_vs_seq": _round(speedup),
        "det_ms_sync": _round(report.get("avg_det_ms_per_sync_frame"), 2),
        "track_ms_cam": _round(report.get("avg_track_ms_per_cam"), 2),
        "reid_ms_sync": _round(report.get("avg_reid_ms_per_sync_frame"), 2),
        "mcmt_ms": _round(report.get("avg_mcmt_ms"), 2),
    }


def _print_markdown(rows: list[dict]) -> None:
    print("\n| variant | pipeline_fps | ms_per_sync_frame | speedup_vs_seq | det_ms_sync | track_ms_cam | reid_ms_sync | mcmt_ms |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {variant} | {pipeline_fps} | {ms_per_sync_frame} | {speedup_vs_seq} | "
            "{det_ms_sync} | {track_ms_cam} | {reid_ms_sync} | {mcmt_ms} |".format(
                **row
            )
        )


def main() -> None:
    reports = {variant: _read_report(variant) for variant in VARIANTS}
    baseline = reports.get("seq_960")
    baseline_ms = (
        float(baseline["ms_per_sync_frame"])
        if baseline and "ms_per_sync_frame" in baseline
        else None
    )

    rows = [
        _row(variant, report, baseline_ms)
        for variant, report in reports.items()
        if report is not None
    ]
    if not rows:
        raise SystemExit("No latency fps.json files found")

    BASE.mkdir(parents=True, exist_ok=True)
    out_csv = BASE / "summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    _print_markdown(rows)
    print(f"\n[OK] Wrote {out_csv}")


if __name__ == "__main__":
    main()
