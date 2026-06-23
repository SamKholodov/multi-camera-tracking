"""Bootstrap GTA diploma_metrics.csv from temporal summary + targeted evals."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.build_diploma_metrics import _eval_run, _write_csv  # noqa: E402
from scripts.eval_s02 import BATCH_METRICS  # noqa: E402

GT_ROOT = _ROOT / "outputs" / "configs_gta"
OUT = GT_ROOT / "diploma_metrics.csv"
TEMPORAL = GT_ROOT / "temporal_ablation_summary.csv"

MCMT_ONLY_GROUPS = {
    "geo_ablation",
    "temporal_ablation",
    "kinematic_ablation",
    "trajectory_ablation",
}

FAST_GROUPS: set[str] = set()  # pass --with-conf / --with-byte to enable


def load_mcmt_from_summary() -> list[dict]:
    rows: list[dict] = []
    if not TEMPORAL.is_file():
        return rows
    with TEMPORAL.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["stream"] != "mcmt":
                continue
            rows.append(
                {
                    "group": row["group"],
                    "name": row["name"],
                    "stream": "mcmt",
                    **{k: row[k] for k in BATCH_METRICS},
                }
            )
    return rows


def eval_groups(groups: set[str]) -> list[dict]:
    rows: list[dict] = []
    root = GT_ROOT
    for group_dir in sorted(root.iterdir()):
        if not group_dir.is_dir() or group_dir.name not in groups:
            continue
        for run_dir in sorted(group_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            metrics = _eval_run(run_dir, dataset="gta")
            for stream, m in metrics.items():
                rows.append(
                    {
                        "group": group_dir.name,
                        "name": run_dir.name,
                        "stream": stream,
                        **{k: m[k] for k in BATCH_METRICS},
                    }
                )
            print(f"[OK] {group_dir.name}/{run_dir.name}: {list(metrics)}", flush=True)
    return rows


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--with-conf", action="store_true")
    ap.add_argument("--with-byte", action="store_true")
    args = ap.parse_args()
    groups = set()
    if args.with_conf:
        groups.add("conf_ablation")
    if args.with_byte:
        groups.add("byte_ablation")

    rows = load_mcmt_from_summary()
    print(f"Loaded {len(rows)} MCMT rows from {TEMPORAL.name}", flush=True)
    if groups:
        rows.extend(eval_groups(groups))
    _write_csv(OUT, rows)
    print(f"Wrote {len(rows)} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
