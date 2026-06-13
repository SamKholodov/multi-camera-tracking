"""Batch SCT + MCMT eval for detector ablation outputs."""

from __future__ import annotations



import argparse

import sys

from pathlib import Path



_ROOT = Path(__file__).resolve().parents[1]

if str(_ROOT) not in sys.path:

    sys.path.insert(0, str(_ROOT))



from scripts.eval_s02 import BATCH_METRICS, evaluate_s02



GT_ROOT = Path("datasets/validation/S02")

CAMS = [6, 7, 8, 9]





def _scalar(value) -> float:

    if hasattr(value, "iloc"):

        return float(value.iloc[0])

    return float(value)





def _row_from_eval(result: dict, stream: str) -> dict:

    if stream == "mcmt":

        row = result["mcmt"]

        if row is None:

            return {m: float("nan") for m in BATCH_METRICS}

        return {m: _scalar(row[m]) for m in BATCH_METRICS}

    return {m: _scalar(result["per_cam"].loc["OVERALL", m]) for m in BATCH_METRICS}





def eval_detector(base: Path, det: str, cityflow_protocol: bool) -> tuple[dict, dict]:

    sct = evaluate_s02(

        GT_ROOT,

        base / det / "per_cam_local",

        cameras=CAMS,

        cityflow_protocol=cityflow_protocol,

        pred_id_mode="local",

    )

    mcmt = evaluate_s02(

        GT_ROOT,

        base / det / "per_cam",

        cameras=CAMS,

        cityflow_protocol=cityflow_protocol,

        pred_id_mode="global",

    )

    return _row_from_eval(sct, "sct"), _row_from_eval(mcmt, "mcmt")





def _print_table(title: str, rows: list[tuple[str, dict]]):

    print(f"\n=== {title} ===")

    print(f"{'detector':10s} {'IDF1':>7s} {'IDP':>7s} {'IDR':>7s} {'MOTA':>8s} {'FP':>7s} {'FN':>7s}")

    for det, s in sorted(rows, key=lambda x: -x[1]["idf1"]):

        print(

            f"{det:10s} {s['idf1']*100:6.1f}% {s['idp']*100:6.1f}% {s['idr']*100:6.1f}% "

            f"{s['mota']*100:7.1f}% {int(s['num_false_positives']):7d} {int(s['num_misses']):7d}"

        )





def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(

        "--base",

        type=Path,

        default=Path("outputs/s02_baseline/deepocsort/detectors"),

    )

    ap.add_argument(

        "--cityflow-protocol",

        action="store_true",

        help="CityFlow-aligned evaluation (ROI + cross-camera objects)",

    )

    args = ap.parse_args()

    base = args.base

    if not base.is_dir():

        raise SystemExit(f"Base directory missing: {base}")



    detectors = sorted(

        p.name for p in base.iterdir() if p.is_dir() and (p / "per_cam" / "c006.txt").exists()

    )

    if not detectors:

        raise SystemExit(f"No detector runs found under {base}")



    proto = "CityFlow" if args.cityflow_protocol else "full MOT"

    sct_rows: list[tuple[str, dict]] = []

    mcmt_rows: list[tuple[str, dict]] = []

    for det in detectors:

        sct, mcmt = eval_detector(base, det, args.cityflow_protocol)

        sct_rows.append((det, sct))

        mcmt_rows.append((det, mcmt))



    _print_table(f"SCT (per_cam_local, {proto})", sct_rows)

    _print_table(f"MCMT (per_cam, {proto})", mcmt_rows)





if __name__ == "__main__":

    main()

