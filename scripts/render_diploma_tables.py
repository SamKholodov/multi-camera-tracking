"""Render markdown tables from diploma_metrics.csv for DIPLOMA_TECHNICAL.md."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def pct(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v):
        return "—"
    return f"{v * 100:.1f}%"


def num(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v):
        return "—"
    return str(int(v))


def load_csv(path: Path) -> dict[tuple[str, str, str], dict]:
    data: dict[tuple[str, str, str], dict] = {}
    if not path.is_file():
        return data
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["group"], row["name"], row["stream"])
            data[key] = row
    return data


def row_line(group: str, name: str, data: dict, stream: str) -> str:
    r = data.get((group, name, stream))
    if not r:
        return f"| {group} | {name} | — | — | — | — | — | — |"
    return (
        f"| {group} | {name} | {pct(r['idf1'])} | {pct(r['idp'])} | {pct(r['idr'])} | "
        f"{pct(r['mota'])} | {num(r['num_false_positives'])} | {num(r['num_misses'])} |"
    )


def combined_row(group: str, name: str, data: dict) -> str:
    sct = data.get((group, name, "sct"))
    mcmt = data.get((group, name, "mcmt"))
    if not sct and not mcmt:
        return f"| {name} | — | — | — | — | — | — |"
    s_id = pct(sct["idf1"]) if sct else "—"
    s_mo = pct(sct["mota"]) if sct else "—"
    m_id = pct(mcmt["idf1"]) if mcmt else "—"
    m_mo = pct(mcmt["mota"]) if mcmt else "—"
    delta = "—"
    if sct and mcmt:
        try:
            sv = float(sct["idf1"])
            mv = float(mcmt["idf1"])
            if not (math.isnan(sv) or math.isnan(mv)):
                delta = f"{(mv - sv) * 100:+.1f}pp"
        except (TypeError, ValueError):
            pass
    return f"| {name} | {s_id} | {s_mo} | {m_id} | {m_mo} | {delta} |"


def table_sct(data: dict, groups: list[tuple[str, list[str]]]) -> str:
    lines = [
        "| Абляция | Вариант | IDF1 | IDP | IDR | MOTA | FP | FN |",
        "|---------|---------|------|-----|-----|------|----|----|",
    ]
    for group, names in groups:
        for name in names:
            lines.append(row_line(group, name, data, "sct"))
    return "\n".join(lines)


def table_mcmt(data: dict, groups: list[tuple[str, list[str]]]) -> str:
    lines = [
        "| Абляция | Вариант | IDF1 | IDP | IDR | MOTA | FP | FN |",
        "|---------|---------|------|-----|-----|------|----|----|",
    ]
    for group, names in groups:
        for name in names:
            lines.append(row_line(group, name, data, "mcmt"))
    return "\n".join(lines)


def table_combined(data: dict, group: str, names: list[str], header: str) -> str:
    lines = [
        header,
        "|---------|----------|----------|-----------|-----------|--------|",
    ]
    for name in names:
        lines.append(combined_row(group, name, data))
    return "\n".join(lines)


def best_rows(data: dict, label: str) -> tuple[str, str]:
    best_sct = ("", -1.0)
    best_mcmt = ("", -1.0)
    seen = set()
    for (group, name, stream), row in data.items():
        if (group, name) in seen:
            continue
        seen.add((group, name))
        for stream_name, best in (("sct", best_sct), ("mcmt", best_mcmt)):
            r = data.get((group, name, stream_name))
            if not r:
                continue
            try:
                v = float(r["idf1"])
            except (TypeError, ValueError):
                continue
            if math.isnan(v):
                continue
            if v > best[1]:
                best = (f"{group}/{name}", v)
                if stream_name == "sct":
                    best_sct = best
                else:
                    best_mcmt = best
    return (
        f"| {label} | {best_sct[0] or '—'} | {pct(best_sct[1]) if best_sct[0] else '—'} | — | |",
        f"| {label} | {best_mcmt[0] or '—'} | {pct(best_mcmt[1]) if best_mcmt[0] else '—'} | — | — | |",
    )


CF_GROUPS = [
    ("baseline_trackers", ["sort", "ocsort", "deepocsort", "botsort", "deepocsort_byte"]),
    ("conf_ablation", [f"conf_0_{x}" for x in (10, 20, 30, 40, 50, 60)]),
    ("reid_ablation", [
        "osnet_ibn_msmt17",
        "vehicle_osnet_veri_vric",
        "vehicle_osnet_view_finetune",
        "vehicle_osnet_veri_vric_wild_epoch120",
        "vehicle_osnet_veri_vric_wild_epoch120_asso070",
    ]),
    ("byte_ablation", ["byte_off", "byte_on_det02", "byte_on_det03", "byte_on_narrow", "byte_smoke"]),
    ("ema_vs_aaf", ["ema", "aaf", "aaf_strict"]),
    ("sort", ["yolo26l"]),
    ("baseline", ["baseline"]),
    ("zone_tracklet", ["zone_tracklet"]),
    ("assoc_ablation", ["reid_only", "+zone_tracklet", "no_different_cam_geo_tiers"]),
    ("geo_ablation", [
        "geo_tight", "geo_baseline", "geo_loose",
        "contact_point_world_bottom_center",
        "geo_tight_temporal_N15_p015", "geo_tight_temporal_N15_p025",
        "geo_tight_temporal_N15_p035", "geo_tight_temporal_N50",
        "geo_tight_temporal_strict_N35",
    ]),
    ("temporal_ablation", ["temporal_off", "temporal_penalty_N20", "temporal_penalty_N50", "temporal_penalty_N100"]),
    ("kinematic_ablation", ["speed_penalty_v25", "speed_penalty_v35"]),
    ("trajectory_ablation", ["traj_linear_K1", "traj_linear_K3"]),
]

GTA_GROUPS = [
    ("baseline_trackers", ["sort", "ocsort", "deepocsort", "botsort"]),
    ("conf_ablation", [f"conf_0_{x}" for x in (10, 20, 30, 40, 50, 60)]),
    ("reid_ablation", [
        "osnet_ibn_msmt17",
        "vehicle_osnet_veri_vric",
        "vehicle_osnet_view_finetune",
        "vehicle_osnet_veri_vric_wild_epoch120",
    ]),
    ("byte_ablation", ["byte_off", "byte_on_det02", "byte_on_det03", "byte_on_narrow"]),
    ("assoc_ablation", ["reid_only", "+zone_tracklet", "no_different_cam_geo_tiers"]),
    ("geo_ablation", [
        "geo_tight", "geo_baseline", "geo_loose",
        "contact_point_world_bottom_center", "contact_point_world_contact",
    ]),
    ("temporal_ablation", ["temporal_off", "temporal_penalty_N60", "temporal_penalty_N150", "temporal_penalty_N300"]),
    ("kinematic_ablation", ["speed_penalty_v25", "speed_penalty_v35"]),
    ("trajectory_ablation", ["traj_linear_K1", "traj_linear_K3"]),
]


def render_section(title: str, data: dict, groups: list) -> str:
    sct_groups = [g for g in groups if g[0] in {
        "baseline_trackers", "conf_ablation", "reid_ablation", "byte_ablation",
        "ema_vs_aaf", "sort", "baseline", "latency_ablation",
    }]
    mcmt_groups = groups
    parts = [f"### {title} — SCT (`per_cam_local`, OVERALL)", "", table_sct(data, sct_groups), ""]
    parts += [f"### {title} — MCMT (concatenated stream, CityFlow protocol)", "", table_mcmt(data, mcmt_groups)]
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf-csv", type=Path, default=Path("outputs/configs_cityflow/diploma_metrics.csv"))
    ap.add_argument("--gta-csv", type=Path, default=Path("outputs/configs_gta/diploma_metrics.csv"))
    ap.add_argument("--output", type=Path, default=Path("outputs/diploma_tables.md"))
    args = ap.parse_args()

    cf = load_csv(args.cf_csv)
    gta = load_csv(args.gta_csv)

    out = ["## Сводные таблицы результатов (фактические прогоны)", ""]
    out.append("Протокол: **CityFlow** — ROI + cross-camera GT objects, IoU 0.5, sync 1920 кадров. **GTA** — ROI, IoU 0.7, ~10k кадров.")
    out.append("**SCT** = `per_cam_local` (локальные ID). **MCMT** = concatenated stream с глобальными ID.")
    out.append("")
    out.append(render_section("CityFlow S02", cf, CF_GROUPS))
    out.append("")
    if gta:
        out.append(render_section("GTA MCMT", gta, GTA_GROUPS))
    else:
        out.append("### GTA MCMT — ожидается `diploma_metrics.csv`")
    out.append("")
    if cf:
        s, m = best_rows(cf, "CityFlow")
        out += ["### Итог — лучшие конфигурации", "", "**SCT:**", "| Датасет | Вариант | IDF1 | MOTA | Комментарий |", "|---------|---------|------|------|-------------|", s, "", "**MCMT:**", "| Датасет | Вариант | IDF1 | MOTA | Δ от SCT | Комментарий |", "|---------|---------|------|------|----------|-------------|", m]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
