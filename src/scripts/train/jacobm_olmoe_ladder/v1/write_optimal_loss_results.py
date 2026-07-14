#!/usr/bin/env python
"""Extract compact optimal-loss tables from PLOTTED_RESULTS.md."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

LADDER_DIR = Path(__file__).parent
RESULTS_DIR = LADDER_DIR / "results"


def parse_tables(text: str) -> list[dict]:
    lines = text.splitlines()
    section = ""
    tables = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            section = line[3:].strip()
        if line.startswith("| ") and i + 1 < len(lines) and lines[i + 1].startswith("| ---"):
            headers = [cell.strip() for cell in line.strip("| ").split(" | ")]
            rows = []
            i += 2
            while i < len(lines) and lines[i].startswith("| "):
                rows.append([cell.strip() for cell in lines[i].strip("| ").split(" | ")])
                i += 1
            tables.append({"section": section, "headers": headers, "rows": rows})
            continue
        i += 1
    return tables


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=LADDER_DIR / "PLOTTED_RESULTS.md")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "optimal_losses.md")
    parser.add_argument("--json-output", type=Path, default=RESULTS_DIR / "optimal_losses.json")
    args = parser.parse_args()

    tables = parse_tables(args.source.read_text())
    records = []
    for table in tables:
        headers = table["headers"]
        if "best observed LR" not in headers or "best avg250M" not in headers:
            continue
        for row in table["rows"]:
            item = dict(zip(headers, row, strict=False))
            item["section"] = table["section"]
            records.append(item)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open("w") as f:
        json.dump({"generated_at_utc": datetime.now(UTC).isoformat(), "source": str(args.source), "records": records}, f, indent=2, sort_keys=True)

    lines = [
        "# Optimal Loss Values",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"Source: `{args.source}`. Values mirror the completed-run plotting policy: final-window training CE averaged over the last 250M tokens, running jobs excluded.",
        "",
    ]
    by_section = {}
    for record in records:
        by_section.setdefault(record["section"], []).append(record)
    for section, section_records in by_section.items():
        keys = [key for key in ["model", "Cx", "variant", "family", "batch", "best observed LR", "best avg250M", "fit LR", "fit avg250M", "points"] if any(key in r for r in section_records)]
        lines.extend([f"## {section}", ""])
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
        for record in section_records:
            lines.append("| " + " | ".join(record.get(key, "") for key in keys) + " |")
        lines.append("")

    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output}")
    print(f"wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
