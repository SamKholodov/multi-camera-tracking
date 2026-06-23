"""Sweep reid_cost_threshold for CityFlow MCMT with fine-tuned ReID."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_s02 import evaluate_s02

BASE_CONFIG = ROOT / "best_configs" / "cityflow_mcmt_best_reid_cf_ft.local.yaml"
GT_ROOT = ROOT / "datasets" / "AICity22_Track1_MTMC_Tracking" / "validation" / "S02"
SWEEP_ROOT = ROOT / "outputs" / "sweeps"


def _load_base_config() -> dict:
    with BASE_CONFIG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_config(threshold: float, out_dir: Path) -> dict:
    cfg = copy.deepcopy(_load_base_config())
    cfg["multi_camera"]["results_dir"] = str(out_dir).replace("\\", "/")
    assoc = cfg["multi_camera"].setdefault("association", {})
    assoc["reid_cost_threshold"] = float(threshold)
    cfg["multi_camera"]["association_cost_threshold"] = float(threshold)
    return cfg


CAMERAS = [6, 7, 8, 9]


def _predictions_ready(out_dir: Path) -> bool:
    per_cam = out_dir / "per_cam"
    if not per_cam.is_dir():
        return False
    for cam in CAMERAS:
        path = per_cam / f"c{cam:03d}.txt"
        if not path.exists() or path.stat().st_size == 0:
            return False
    return True


def _run_mcmt(config_path: Path) -> None:
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(ROOT)
    cmd = [sys.executable, str(ROOT / "run.py"), "--config", str(config_path)]
    print(f"[run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def _series_float(value) -> float:
    try:
        return float(value.iloc[0])
    except AttributeError:
        return float(value)


def _run_eval(pred_dir: Path) -> dict:
    result = evaluate_s02(
        GT_ROOT,
        pred_dir,
        cityflow_protocol=True,
    )
    mcmt = result["mcmt"]
    if mcmt is None:
        raise RuntimeError(f"No MCMT metrics for {pred_dir}")
    return {
        "idf1": _series_float(mcmt["idf1"]) * 100.0,
        "idp": _series_float(mcmt["idp"]) * 100.0,
        "idr": _series_float(mcmt["idr"]) * 100.0,
    }


def _threshold_tag(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thresholds",
        default="0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80",
        help="Comma-separated reid_cost_threshold values",
    )
    parser.add_argument(
        "--results-json",
        default=str(SWEEP_ROOT / "cityflow_reid_cf_threshold_sweep.json"),
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only eval existing outputs (skip run.py)",
    )
    args = parser.parse_args()
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    results_path = Path(args.results_json)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)

    rows = list(existing.get("results", []))
    done = {float(r["reid_cost_threshold"]) for r in rows if "mcmt_idf1" in r}

    for threshold in thresholds:
        if threshold in done:
            print(f"[skip] threshold {threshold} already in results", flush=True)
            continue
        tag = _threshold_tag(threshold)
        out_dir = SWEEP_ROOT / f"cityflow_reid_cf_thresh_{tag}"
        config_path = SWEEP_ROOT / f"cityflow_reid_cf_thresh_{tag}.yaml"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = _make_config(threshold, out_dir)
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        t0 = time.time()
        if not args.skip_run and not _predictions_ready(out_dir):
            try:
                _run_mcmt(config_path)
            except subprocess.CalledProcessError as exc:
                if _predictions_ready(out_dir):
                    print(
                        f"[warn] run.py exited {exc.returncode} but predictions exist; continuing eval",
                        flush=True,
                    )
                else:
                    raise
        elif _predictions_ready(out_dir):
            print(f"[skip-run] predictions already exist for threshold {threshold}", flush=True)
        pred_dir = out_dir / "per_cam"
        metrics = _run_eval(pred_dir)
        elapsed_min = (time.time() - t0) / 60.0
        row = {
            "reid_cost_threshold": threshold,
            "results_dir": str(out_dir).replace("\\", "/"),
            "mcmt_idf1": metrics.get("idf1"),
            "mcmt_idp": metrics.get("idp"),
            "mcmt_idr": metrics.get("idr"),
            "elapsed_min": round(elapsed_min, 2),
        }
        rows = [r for r in rows if float(r["reid_cost_threshold"]) != threshold]
        rows.append(row)
        rows.sort(key=lambda r: float(r["reid_cost_threshold"]))

        best = max((r for r in rows if r.get("mcmt_idf1") is not None), key=lambda r: r["mcmt_idf1"])
        prev_thresholds = [float(x) for x in existing.get("thresholds", [])]
        all_thresholds = sorted(set(prev_thresholds + thresholds))
        payload = {
            "base_config": str(BASE_CONFIG).replace("\\", "/"),
            "reid_weights": cfg["tracker"]["reid_weights"],
            "note": "reid_strong_reject_threshold unused (same_frame_linking=false)",
            "thresholds": all_thresholds,
            "results": rows,
            "best": {
                "reid_cost_threshold": best["reid_cost_threshold"],
                "mcmt_idf1": best["mcmt_idf1"],
                "mcmt_idp": best["mcmt_idp"],
                "mcmt_idr": best["mcmt_idr"],
            },
        }
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(
            f"[done] threshold={threshold} IDF1={row['mcmt_idf1']:.1f}% ({elapsed_min:.1f} min)",
            flush=True,
        )


if __name__ == "__main__":
    main()
