"""Inject factual metrics from diploma_metrics.csv into DIPLOMA_TECHNICAL.md."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.render_diploma_tables import (
    CF_GROUPS,
    GTA_GROUPS,
    best_rows,
    load_csv,
    row_line,
    table_mcmt,
    table_sct,
)

DIPLOMA = _ROOT / "DIPLOMA_TECHNICAL.md"

REID_NAMES = {
    "msmt17": "osnet_ibn_msmt17",
    "veri_vric": "vehicle_osnet_veri_vric",
    "view_finetune": "vehicle_osnet_view_finetune",
    "wild_epoch120": "vehicle_osnet_veri_vric_wild_epoch120",
    "wild_asso070": "vehicle_osnet_veri_vric_wild_epoch120_asso070",
}

CONF_ROWS = [(f"0.{d}", f"conf_0_{d}0") for d in range(1, 7)]

CF_TEMPORAL = {
    "temporal_off": "temporal_off",
    "temporal_N20": "temporal_penalty_N20",
    "temporal_N50": "temporal_penalty_N50",
    "temporal_N100": "temporal_penalty_N100",
}
GTA_TEMPORAL = {
    "temporal_off": "temporal_off",
    "temporal_N60": "temporal_penalty_N60",
    "temporal_N150": "temporal_penalty_N150",
    "temporal_N300": "temporal_penalty_N300",
}

GEO_CF = {
    "geo_tight": "geo_tight",
    "geo_baseline": "geo_baseline",
    "geo_loose": "geo_loose",
    "contact_bottom_center": "contact_point_world_bottom_center",
    "contact_point": "contact_point_world_bottom_center",
}
GEO_GTA = {
    "geo_tight": "geo_tight",
    "geo_baseline": "geo_baseline",
    "geo_loose": "geo_loose",
    "contact_bottom_center": "contact_point_world_bottom_center",
    "contact_point": "contact_point_world_contact",
}


def pct(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v):
        return "—"
    return f"{v * 100:.1f}%"


def delta_idf1(data: dict, group: str, name: str) -> str:
    sct = data.get((group, name, "sct"))
    mcmt = data.get((group, name, "mcmt"))
    if not sct or not mcmt:
        return "—"
    try:
        sv = float(sct["idf1"])
        mv = float(mcmt["idf1"])
        if math.isnan(sv) or math.isnan(mv):
            return "—"
        return f"{(mv - sv) * 100:+.1f}pp"
    except (TypeError, ValueError):
        return "—"


def tracker_row(data: dict, group: str, name: str) -> str:
    sct = data.get((group, name, "sct"))
    mcmt = data.get((group, name, "mcmt"))
    return (
        f"| {name} | {pct(sct['idf1']) if sct else '—'} | {pct(sct['mota']) if sct else '—'} | "
        f"{pct(mcmt['idf1']) if mcmt else '—'} | {pct(mcmt['mota']) if mcmt else '—'} | "
        f"{delta_idf1(data, group, name)} |"
    )


def tracker_table(data: dict, group: str, names: list[str], title: str) -> str:
    lines = [f"**{title}**", "", "| Вариант | SCT IDF1 | SCT MOTA | MCMT IDF1 | MCMT MOTA | Δ IDF1 |",
             "|---------|----------|----------|-----------|-----------|--------|"]
    for name in names:
        lines.append(tracker_row(data, group, name))
    return "\n".join(lines)


def mcmt_only_row(data: dict, group: str, name: str, label: str, extra: str = "") -> str:
    mcmt = data.get((group, name, "mcmt"))
    return (
        f"| {label} | {extra} | {pct(mcmt['idf1']) if mcmt else '—'} | "
        f"{pct(mcmt['mota']) if mcmt else '—'} |"
    )


def conf_table(data: dict, title: str) -> str:
    lines = [
        f"**{title}**",
        "",
        "| conf_thres | SCT IDF1 | SCT MOTA | MCMT IDF1 | MCMT MOTA | Δ IDF1 |",
        "|------------|----------|----------|-----------|-----------|--------|",
    ]
    for conf, name in CONF_ROWS:
        lines.append(
            f"| {conf} | "
            + " | ".join(
                [
                    pct(data.get(("conf_ablation", name, "sct"), {}).get("idf1")),
                    pct(data.get(("conf_ablation", name, "sct"), {}).get("mota")),
                    pct(data.get(("conf_ablation", name, "mcmt"), {}).get("idf1")),
                    pct(data.get(("conf_ablation", name, "mcmt"), {}).get("mota")),
                    delta_idf1(data, "conf_ablation", name),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def reid_table(data: dict, title: str) -> str:
    lines = [
        f"**{title}**",
        "",
        "| Модель | SCT IDF1 | SCT MOTA | MCMT IDF1 | MCMT MOTA | Δ IDF1 |",
        "|--------|----------|----------|-----------|-----------|--------|",
    ]
    for label, name in REID_NAMES.items():
        lines.append(
            f"| {label} | "
            + " | ".join(
                [
                    pct(data.get(("reid_ablation", name, "sct"), {}).get("idf1")),
                    pct(data.get(("reid_ablation", name, "sct"), {}).get("mota")),
                    pct(data.get(("reid_ablation", name, "mcmt"), {}).get("idf1")),
                    pct(data.get(("reid_ablation", name, "mcmt"), {}).get("mota")),
                    delta_idf1(data, "reid_ablation", name),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def byte_table(data: dict, title: str) -> str:
    names = ["byte_off", "byte_on_det02", "byte_on_det03", "byte_on_narrow"]
    meta = {
        "byte_off": ("нет", "0.3", "—"),
        "byte_on_det02": ("да", "0.2", "0.1"),
        "byte_on_det03": ("да", "0.3", "0.1"),
        "byte_on_narrow": ("да", "0.25", "0.15"),
    }
    lines = [
        f"**{title}**",
        "",
        "| Вариант | use_byte | det_thresh | min_conf | SCT IDF1 | MCMT IDF1 | Δ IDF1 |",
        "|---------|----------|------------|----------|----------|-----------|--------|",
    ]
    for name in names:
        ub, dt, mc = meta[name]
        lines.append(
            f"| {name} | {ub} | {dt} | {mc} | "
            f"{pct(data.get(('byte_ablation', name, 'sct'), {}).get('idf1'))} | "
            f"{pct(data.get(('byte_ablation', name, 'mcmt'), {}).get('idf1'))} | "
            f"{delta_idf1(data, 'byte_ablation', name)} |"
        )
    return "\n".join(lines)


def ema_table(data: dict, title: str, names: list[str]) -> str:
    labels = {"ema": "ema", "aaf": "aaf", "aaf_strict": "aaf (strict assoc)"}
    lines = [
        f"**{title}**",
        "",
        "| Вариант | appearance_update | SCT IDF1 | MCMT IDF1 | Δ IDF1 |",
        "|---------|-------------------|----------|-----------|--------|",
    ]
    for name in names:
        lines.append(
            f"| {name} | {labels.get(name, name)} | "
            f"{pct(data.get(('ema_vs_aaf', name, 'sct'), {}).get('idf1'))} | "
            f"{pct(data.get(('ema_vs_aaf', name, 'mcmt'), {}).get('idf1'))} | "
            f"{delta_idf1(data, 'ema_vs_aaf', name)} |"
        )
    return "\n".join(lines)


def assoc_table(data: dict, title: str) -> str:
    rows = [
        ("reid_only", "Только ReID, все gates off"),
        ("+zone_tracklet", "+ zone-transition graph"),
        ("no_different_cam_geo_tiers", "+ tiered plane geo 14/38 m"),
    ]
    lines = [
        f"**{title}**",
        "",
        "| Вариант | Описание | MCMT IDF1 | MCMT MOTA |",
        "|---------|----------|-----------|-----------|",
    ]
    for name, desc in rows:
        mcmt = data.get(("assoc_ablation", name, "mcmt"))
        lines.append(
            f"| `{name}` | {desc} | {pct(mcmt['idf1']) if mcmt else '—'} | "
            f"{pct(mcmt['mota']) if mcmt else '—'} |"
        )
    return "\n".join(lines)


def geo_table(data: dict, mapping: dict, title: str) -> str:
    specs = [
        ("geo_tight", "10", "25", "bottom_center"),
        ("geo_baseline", "14", "38", "bottom_center"),
        ("geo_loose", "14", "55", "bottom_center"),
        ("contact_bottom_center", "14", "38", "bottom_center"),
        ("contact_point", "14", "38", "contact_point"),
    ]
    lines = [
        f"**{title}**",
        "",
        "| Вариант | t_min (m) | t_distant (m) | anchor | MCMT IDF1 | MCMT MOTA |",
        "|---------|-----------|---------------|--------|-----------|-----------|",
    ]
    for key, tmin, td, anchor in specs:
        name = mapping.get(key, key)
        mcmt = data.get(("geo_ablation", name, "mcmt"))
        lines.append(
            f"| {key} | {tmin} | {td} | {anchor} | "
            f"{pct(mcmt['idf1']) if mcmt else '—'} | {pct(mcmt['mota']) if mcmt else '—'} |"
        )
    return "\n".join(lines)


def temporal_table(data: dict, mapping: dict, title: str) -> str:
    specs = list(mapping.items())
    lines = [
        f"**{title}**",
        "",
        "| Вариант | mode | max_gap (frames) | MCMT IDF1 | MCMT MOTA |",
        "|---------|------|------------------|-----------|-----------|",
    ]
    gaps = {
        "temporal_off": "off / 300",
        "temporal_N60": "penalty / 60",
        "temporal_N150": "penalty / 150",
        "temporal_N300": "penalty / 300",
        "temporal_N20": "penalty / 20",
        "temporal_N50": "penalty / 50",
        "temporal_N100": "penalty / 100",
    }
    for label, name in specs:
        mcmt = data.get(("temporal_ablation", name, "mcmt"))
        mode = "off" if label == "temporal_off" else "penalty_only"
        lines.append(
            f"| {label} | {mode} | {gaps.get(label, '—')} | "
            f"{pct(mcmt['idf1']) if mcmt else '—'} | {pct(mcmt['mota']) if mcmt else '—'} |"
        )
    return "\n".join(lines)


def kinematic_table(data: dict, title: str) -> str:
    lines = [
        f"**{title}**",
        "",
        "| Вариант | v_max (m/s) | MCMT IDF1 | MCMT MOTA |",
        "|---------|-------------|-----------|-----------|",
    ]
    for name, vmax in [("speed_penalty_v25", "25"), ("speed_penalty_v35", "35")]:
        mcmt = data.get(("kinematic_ablation", name, "mcmt"))
        lines.append(
            f"| {name} | {vmax} | {pct(mcmt['idf1']) if mcmt else '—'} | "
            f"{pct(mcmt['mota']) if mcmt else '—'} |"
        )
    return "\n".join(lines)


def trajectory_table(data: dict, title: str) -> str:
    lines = [
        f"**{title}**",
        "",
        "| Вариант | history K | threshold (m) | MCMT IDF1 | MCMT MOTA |",
        "|---------|-----------|---------------|-----------|-----------|",
    ]
    for name, k, thr in [("traj_linear_K1", "1", "0.75"), ("traj_linear_K3", "3", "0.07")]:
        mcmt = data.get(("trajectory_ablation", name, "mcmt"))
        lines.append(
            f"| {name} | {k} | {thr} | {pct(mcmt['idf1']) if mcmt else '—'} | "
            f"{pct(mcmt['mota']) if mcmt else '—'} |"
        )
    return "\n".join(lines)


def latency_table() -> str:
    variants = [
        ("seq_960", 960, "нет", "нет"),
        ("batch_960", 960, "да", "нет"),
        ("batch_640", 640, "да", "нет"),
        ("batch_640_reid", 640, "да", "да"),
    ]
    lines = [
        "| Вариант | FPS | ms/frame | imgsz | batch det | batch ReID |",
        "|---------|-----|----------|-------|-----------|------------|",
    ]
    root = _ROOT / "outputs" / "configs_cityflow" / "latency_ablation"
    for name, imgsz, bdet, breid in variants:
        fps_path = root / name / "fps.json"
        if fps_path.is_file():
            fps = json.loads(fps_path.read_text(encoding="utf-8"))
            lines.append(
                f"| {name} | {fps['pipeline_fps']:.2f} | {fps['ms_per_sync_frame']:.1f} | "
                f"{imgsz} | {bdet} | {breid} |"
            )
        else:
            lines.append(f"| {name} | — | — | {imgsz} | {bdet} | {breid} |")
    return "\n".join(lines)


def best_summary(cf: dict, gta: dict) -> str:
    sct_lines = [
        "| Датасет | Лучший вариант | IDF1 | MOTA | Комментарий |",
        "|---------|----------------|------|------|-------------|",
    ]
    mcmt_lines = [
        "| Датасет | Лучший вариант | IDF1 | MOTA | Δ от SCT | Комментарий |",
        "|---------|----------------|------|------|----------|-------------|",
    ]
    if gta:
        gs, gm = best_rows(gta, "GTA")
        sct_lines.append(gs.replace("| GTA |", "| GTA |").replace("— | |", "— | — |"))
        mcmt_lines.append(gm)
    if cf:
        cs, cm = best_rows(cf, "CityFlow")
        sct_lines.append(cs)
        mcmt_lines.append(cm)
    return (
        "**SCT — лучшие конфигурации:**\n\n"
        + "\n".join(sct_lines)
        + "\n\n**MCMT — лучшие конфигурации:**\n\n"
        + "\n".join(mcmt_lines)
    )


def groups_from_csv(data: dict, stream: str) -> list[tuple[str, list[str]]]:
    by_group: dict[str, list[str]] = {}
    for group, name, st in data:
        if st != stream:
            continue
        if name not in by_group.setdefault(group, []):
            by_group[group].append(name)
    return [(group, sorted(names)) for group, names in sorted(by_group.items())]


def conclusion_text(cf: dict, gta: dict) -> str:
    parts = ["Численные значения метрик приведены в таблицах §5–6."]
    if cf:
        cs, cm = best_rows(cf, "CityFlow")
        cf_sct = cs.split("|")[2].strip()
        cf_mcmt = cm.split("|")[2].strip()
        parts.append(f"На **CityFlow** лучший SCT — `{cf_sct}`; лучший MCMT — `{cf_mcmt}`.")
    if gta:
        gs, gm = best_rows(gta, "GTA")
        gta_sct = gs.split("|")[2].strip()
        gta_mcmt = gm.split("|")[2].strip()
        parts.append(f"На **GTA** лучший SCT — `{gta_sct}`; лучший MCMT — `{gta_mcmt}`.")
    parts.append(
        "Межкамерные gates (geo tiers + temporal penalty) дают основной прирост MCMT "
        "относительно pure ReID. View loss на ReID не дал значимого улучшения; "
        "в финальном стеке — VeRI-Wild `epoch_120`."
    )
    return " ".join(parts)


def replace_block(text: str, start: str, end: str, body: str) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"Block not found: {start!r} .. {end!r}")
    return pattern.sub(start + "\n\n" + body + "\n\n" + end, text, count=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cf-csv", type=Path, default=_ROOT / "outputs/configs_cityflow/diploma_metrics.csv")
    ap.add_argument("--gta-csv", type=Path, default=_ROOT / "outputs/configs_gta/diploma_metrics.csv")
    ap.add_argument("--diploma", type=Path, default=DIPLOMA)
    args = ap.parse_args()

    cf = load_csv(args.cf_csv)
    gta = load_csv(args.gta_csv)
    text = args.diploma.read_text(encoding="utf-8")

    trackers = ["sort", "ocsort", "deepocsort", "botsort", "deepocsort_byte"]

    text = replace_block(
        text,
        "**Результаты (GTA / CityFlow):**",
        "---\n\n### 5.2. Детекторы",
        "\n\n".join(
            [
                tracker_table(gta, "baseline_trackers", trackers, "GTA MCMT"),
                tracker_table(cf, "baseline_trackers", trackers, "CityFlow S02"),
            ]
        ),
    )

    text = replace_block(
        text,
        "#### 5.2.1. Conf threshold sweep (GTA, 2k кадров)",
        "#### 5.2.2. Detector backbone sweep (8 моделей)",
        "\n\n".join(
            [
                "Предварительный подбор `detector.conf_thres` перед основными абляциями. "
                "Пороги: `0.1 … 0.6`. Протокол: **2k кадров**, ROI, IoU 0.7.",
                conf_table(gta, "GTA MCMT"),
                conf_table(cf, "CityFlow S02 (полный прогон 1920 sync-кадров)"),
            ]
        ),
    )

    text = replace_block(
        text,
        "**Фиксировано:** DeepOcSort + MSMT17, shared ReID model.\n\n| Вариант | FPS | ms/frame |",
        "**Ожидаемый вывод:** батчевая детекция",
        latency_table(),
    )

    text = replace_block(
        text,
        "**Результаты:**",
        "---\n\n### 5.5. Межкамерная ассоциация",
        "\n\n".join([reid_table(gta, "GTA MCMT"), reid_table(cf, "CityFlow S02")]),
    )

    text = replace_block(
        text,
        "#### 5.5.1. Association ablation",
        "#### 5.5.2. Geometry ablation",
        "\n\n".join([assoc_table(gta, "GTA MCMT"), assoc_table(cf, "CityFlow S02")]),
    )

    text = replace_block(
        text,
        "#### 5.5.2. Geometry ablation",
        "#### 5.5.3. Temporal ablation",
        "\n\n".join(
            [
                geo_table(gta, GEO_GTA, "GTA MCMT"),
                geo_table(cf, GEO_CF, "CityFlow S02"),
            ]
        ),
    )

    text = replace_block(
        text,
        "#### 5.5.3. Temporal ablation",
        "#### 5.5.4. Kinematic ablation",
        "\n\n".join(
            [
                temporal_table(gta, GTA_TEMPORAL, "GTA MCMT"),
                temporal_table(cf, CF_TEMPORAL, "CityFlow S02"),
            ]
        ),
    )

    text = replace_block(
        text,
        "#### 5.5.4. Kinematic ablation",
        "#### 5.5.5. Trajectory ablation",
        "\n\n".join(
            [
                kinematic_table(gta, "GTA MCMT"),
                kinematic_table(cf, "CityFlow S02"),
            ]
        ),
    )

    text = replace_block(
        text,
        "#### 5.5.5. Trajectory ablation",
        "---\n\n### 5.6. Вспомогательные абляции",
        "\n\n".join(
            [
                trajectory_table(gta, "GTA MCMT"),
                trajectory_table(cf, "CityFlow S02"),
            ]
        ),
    )

    text = replace_block(
        text,
        "#### BYTE\n\n",
        "#### EMA vs AAF",
        "\n\n".join([byte_table(gta, "GTA MCMT"), byte_table(cf, "CityFlow S02")]),
    )

    text = replace_block(
        text,
        "#### EMA vs AAF\n\n",
        "---\n\n## 6. Сводные таблицы результатов",
        "\n\n".join(
            [
                ema_table(gta, "GTA MCMT", ["ema", "aaf", "aaf_strict"]),
                ema_table(cf, "CityFlow S02", ["ema", "aaf", "aaf_strict"]),
            ]
        ),
    )

    sct_gta = groups_from_csv(gta, "sct") if gta else []
    mcmt_gta = groups_from_csv(gta, "mcmt") if gta else GTA_GROUPS
    sct_cf = groups_from_csv(cf, "sct") if cf else []
    mcmt_cf = groups_from_csv(cf, "mcmt") if cf else CF_GROUPS

    section6 = "\n".join(
        [
            "Протокол: **CityFlow** — ROI + cross-camera GT objects, IoU 0.5, sync 1920 кадров. "
            "**GTA** — ROI, IoU 0.7, полный прогон (~10k кадров; conf sweep — 2k).",
            "**SCT** = `per_cam_local` (локальные ID). **MCMT** = concatenated stream с глобальными ID.",
            "Для SORT/OcSort на CityFlow MCMT-IDF1 = `—`: нет cross-camera global ID (протокол CityFlow "
            "оставляет только объекты с ≥2 камерами).",
            "",
            "### 6.1. GTA — SCT (`per_cam_local`, OVERALL)",
            "",
            table_sct(gta, sct_gta) if sct_gta else "_нет данных_",
            "",
            "### 6.2. GTA — MCMT (`per_cam`, concatenated stream)",
            "",
            table_mcmt(gta, mcmt_gta) if mcmt_gta else "_нет данных_",
            "",
            "### 6.3. CityFlow S02 — SCT",
            "",
            table_sct(cf, sct_cf),
            "",
            "### 6.4. CityFlow S02 — MCMT",
            "",
            table_mcmt(cf, mcmt_cf),
            "",
            "### 6.5. Latency (CityFlow S02)",
            "",
            latency_table(),
            "",
            "### 6.6. Итоговая сводка",
            "",
            best_summary(cf, gta),
            "",
            "### 6.7. Воспроизведение метрик",
            "",
            "Метрики извлекаются скриптом `scripts/build_diploma_metrics.py` и вставляются в этот документ "
            "через `scripts/inject_diploma_metrics.py`.",
            "",
            "- **SCT:** `outputs/<run>/per_cam_local/` (локальные ID, OVERALL по камерам).",
            "- **MCMT:** `outputs/<run>/per_cam/` (глобальные ID, concatenated multi-camera stream).",
        ]
    )

    text = replace_block(text, "## 6. Сводные таблицы результатов", "## 7. Заключение", section6)

    text = replace_block(
        text,
        "Численные значения метрик приведены в таблицах §5–6.",
        "",
        conclusion_text(cf, gta),
    )

    args.diploma.write_text(text, encoding="utf-8")
    print(f"Updated {args.diploma} (CF rows={len(cf)//2}, GTA rows={len(gta)//2})")


if __name__ == "__main__":
    main()
