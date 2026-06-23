"""Fix configs duplicated by apply_gta_conf_thres header bug."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = _ROOT / "configs_gta"


def _repair_file(path: Path) -> bool:
    raw = path.read_text(encoding="utf-8")
    docs = list(yaml.safe_load_all(raw))
    docs = [d for d in docs if isinstance(d, dict) and d]
    if len(docs) <= 1:
        return False
    cfg = docs[-1]
    header_lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith("#"):
            header_lines.append(line)
        else:
            break
    header = "\n".join(header_lines)
    if header:
        header += "\n\n"
    path.write_text(header + yaml.dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return True


def main() -> None:
    fixed = 0
    for path in sorted(CONFIG_ROOT.rglob("*")):
        if path.suffix not in {".yaml", ".yml"}:
            continue
        if _repair_file(path):
            print(path.relative_to(_ROOT).as_posix())
            fixed += 1
    print(f"Repaired {fixed} file(s)")


if __name__ == "__main__":
    main()
