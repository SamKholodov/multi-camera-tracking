"""Build SCT (per_cam_local) + MCMT (concatenated) metrics for DIPLOMA_TECHNICAL.md."""
from __future__ import annotations

import argparse
import atexit
import csv
import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cityflow_ablation_common import GT_ROOT as CF_GT  # noqa: E402
from scripts.eval_gta_mcmt import evaluate_gta_mcmt, resolve_eval_max_frame  # noqa: E402
from scripts.eval_s02 import BATCH_METRICS, evaluate_s02  # noqa: E402

CF_CAMS = [6, 7, 8, 9]


def _scalar(value) -> float:
    if hasattr(value, "iloc"):
        return float(value.iloc[0])
    return float(value)


def _metrics_from_result(result: dict, stream: str) -> dict[str, float]:
    if stream == "mcmt":
        row = result["mcmt"]
        if row is None:
            return {m: float("nan") for m in BATCH_METRICS}
        return {m: _scalar(row[m]) for m in BATCH_METRICS}
    return {m: _scalar(result["per_cam"].loc["OVERALL", m]) for m in BATCH_METRICS}


def _eval_run(run_dir: Path, *, dataset: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if dataset == "cityflow":
        if not (run_dir / "per_cam_local" / "c006.txt").is_file():
            return out
        sct = evaluate_s02(
            CF_GT,
            run_dir / "per_cam_local",
            cameras=CF_CAMS,
            cityflow_protocol=True,
            max_iou_dist=0.5,
        )
        out["sct"] = _metrics_from_result(sct, "per_cam")
        if (run_dir / "per_cam" / "c006.txt").is_file():
            mcmt = evaluate_s02(
                CF_GT,
                run_dir / "per_cam",
                cameras=CF_CAMS,
                cityflow_protocol=True,
                max_iou_dist=0.5,
            )
            out["mcmt"] = _metrics_from_result(mcmt, "mcmt")
        return out

    if not (run_dir / "per_cam_local" / "c000.txt").is_file():
        return out
    gt = Path("datasets/gta_mcmt")
    eval_cap, _, _ = resolve_eval_max_frame(gt, run_dir / "per_cam_local")
    sct = evaluate_gta_mcmt(
        gt, run_dir / "per_cam_local", max_iou_dist=0.7, apply_roi=True, max_frame=eval_cap
    )
    out["sct"] = _metrics_from_result(sct, "per_cam")
    if (run_dir / "per_cam" / "c000.txt").is_file():
        mcmt = evaluate_gta_mcmt(
            gt, run_dir / "per_cam", max_iou_dist=0.7, apply_roi=True, max_frame=eval_cap
        )
        out["mcmt"] = _metrics_from_result(mcmt, "mcmt")
    return out


def _iter_runs(root: Path) -> list[tuple[str, str, Path]]:
    runs: list[tuple[str, str, Path]] = []
    singles = {"baseline": "baseline", "zone_tracklet": "zone_tracklet"}
    for group, name in singles.items():
        run = root / name
        if run.is_dir():
            runs.append((group, name, run))
    for group_dir in sorted(root.iterdir()):
        if not group_dir.is_dir():
            continue
        if group_dir.name in singles:
            continue
        for run_dir in sorted(group_dir.iterdir()):
            if run_dir.is_dir():
                runs.append((group_dir.name, run_dir.name, run_dir))
    return runs


def _row_key(group: str, name: str, stream: str) -> tuple[str, str, str]:
    return (group, name, stream)


def _load_existing(path: Path) -> dict[tuple[str, str, str], dict]:
    if not path.is_file():
        return {}
    out: dict[tuple[str, str, str], dict] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[_row_key(row["group"], row["name"], row["stream"])] = row
    return out


def collect(dataset: str, *, output: Path | None = None, resume: bool = False) -> list[dict]:
    root = _ROOT / "outputs" / f"configs_{dataset}"
    existing = _load_existing(output) if resume and output else {}
    rows: list[dict] = list(existing.values()) if existing else []
    seen = set(existing)
    for group, name, run_dir in _iter_runs(root):
        pending = [s for s in ("sct", "mcmt") if _row_key(group, name, s) not in seen]
        if not pending:
            print(f"[SKIP] {group}/{name}: already in {output}", flush=True)
            continue
        metrics = _eval_run(run_dir, dataset=dataset)
        for stream, m in metrics.items():
            key = _row_key(group, name, stream)
            if key in seen:
                continue
            row = {
                "group": group,
                "name": name,
                "stream": stream,
                **{k: m[k] for k in BATCH_METRICS},
            }
            rows.append(row)
            seen.add(key)
            if output is not None:
                _write_csv(output, rows)
        print(f"[OK] {group}/{name}: streams={list(metrics)}", flush=True)
    return rows


def _lock_path(dataset: str) -> Path:
    return _ROOT / "outputs" / f".diploma_metrics_{dataset}.lock"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_lock(dataset: str) -> None:
    path = _lock_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            holder = int(path.read_text(encoding="utf-8").strip())
        except ValueError:
            holder = -1
        if _pid_alive(holder):
            raise SystemExit(
                f"[LOCK] {dataset} metrics build already running (pid {holder}, lock={path})"
            )
        path.unlink(missing_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"[LOCK] {dataset} metrics build already running (lock={path})")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)

    def _release() -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    atexit.register(_release)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["cityflow", "gta", "both"], default="both")
    ap.add_argument("--output-dir", type=Path, default=_ROOT / "outputs")
    ap.add_argument("--resume", action="store_true", help="Skip rows already present in output CSV")
    args = ap.parse_args()

    if args.dataset in ("cityflow", "both"):
        out = args.output_dir / "configs_cityflow" / "diploma_metrics.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = collect("cityflow", output=out, resume=args.resume)
        _write_csv(out, rows)
        print(f"Wrote {len(rows)} rows -> {out}", flush=True)

    if args.dataset in ("gta", "both"):
        _acquire_lock("gta")
        out = args.output_dir / "configs_gta" / "diploma_metrics.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = collect("gta", output=out, resume=args.resume)
        _write_csv(out, rows)
        print(f"Wrote {len(rows)} rows -> {out}", flush=True)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["group", "name", "stream", *BATCH_METRICS]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
