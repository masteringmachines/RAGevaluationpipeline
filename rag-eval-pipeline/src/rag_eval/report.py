"""Report Generator: CSV per query, JSON summary, console comparison table (FR8)."""
import csv
import json
from pathlib import Path
from typing import Dict, List


def write_csv(rows: List[Dict], path) -> None:
    """One row per query, including all per-query metrics (FR8)."""
    if not rows:
        return
    fieldnames = [k for k in rows[0] if k != "retrieved"] + ["retrieved"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "retrieved": "|".join(row["retrieved"])})


def write_summary(summaries: List[Dict], path) -> None:
    """Aggregated numbers per retriever, as JSON (FR8)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


def print_table(summaries: List[Dict]) -> None:
    """Render a simple aligned comparison table to stdout."""
    if not summaries:
        print("No results.")
        return
    cols = list(summaries[0].keys())
    widths = {c: max(len(c), *(len(_fmt(s[c])) for s in summaries)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(" | ".join(_fmt(s[c]).ljust(widths[c]) for c in cols))


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)
