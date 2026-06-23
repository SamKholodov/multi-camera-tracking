"""Run best_configs (GTA + CityFlow) with yolo26l real fine-tune, then eval."""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs" / "best_configs" / "yolo26l_real_run.log"
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(label: str, args: list[str]) -> None:
    log(label)
    cmd = [str(PY), "-u", *args]
    with LOG.open("a", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    run("=== RUN GTA best_configs + yolo26l real ===", [
        "run.py", "--config", "best_configs/gta_mcmt_best_yolo26l_real.yaml",
    ])
    gta_out = ROOT / "outputs/best_configs/gta_mcmt_best_yolo26l_real"
    run("=== EVAL GTA SCT ===", [
        "scripts/eval_gta_mcmt.py",
        "--gt-root", "datasets/gta_mcmt",
        "--pred-dir", str(gta_out / "per_cam_local"),
        "--cameras", "0", "1", "2", "3",
        "--max-iou-dist", "0.7", "--apply-roi", "--align-pred-frames",
    ])
    run("=== EVAL GTA MCMT ===", [
        "scripts/eval_gta_mcmt.py",
        "--gt-root", "datasets/gta_mcmt",
        "--pred-dir", str(gta_out / "per_cam"),
        "--cameras", "0", "1", "2", "3",
        "--max-iou-dist", "0.7", "--apply-roi", "--align-pred-frames",
    ])

    run("=== RUN CityFlow best_configs + yolo26l real ===", [
        "run.py", "--config", "best_configs/cityflow_mcmt_best_yolo26l_real.yaml",
    ])
    cf_out = ROOT / "outputs/best_configs/cityflow_mcmt_best_yolo26l_real"
    gt = "datasets/AICity22_Track1_MTMC_Tracking/validation/S02"
    run("=== EVAL CityFlow SCT ===", [
        "scripts/eval_s02.py", "--gt-root", gt,
        "--pred-dir", str(cf_out / "per_cam_local"), "--cityflow-protocol",
    ])
    run("=== EVAL CityFlow MCMT ===", [
        "scripts/eval_s02.py", "--gt-root", gt,
        "--pred-dir", str(cf_out / "per_cam"), "--cityflow-protocol",
    ])
    log("=== DONE ===")


if __name__ == "__main__":
    main()
