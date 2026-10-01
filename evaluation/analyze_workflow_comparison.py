"""Validate and summarize recorded manual/basic-Agent/GeoScope workflow trials."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


WORKFLOWS = {"manual", "basic_agent", "geoscope"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    recorded = [row for row in rows if row.get("evidence_path", "").strip()]
    if not recorded:
        raise ValueError("no evidence-backed workflow records; do not publish placeholder rows")
    unknown = sorted({row["workflow"] for row in recorded} - WORKFLOWS)
    if unknown:
        raise ValueError(f"unknown workflows: {unknown}")
    grouped = defaultdict(list)
    for row in recorded:
        if not row.get("total_seconds", "").strip() or not row.get("operation_steps", "").strip():
            raise ValueError(f"missing time or steps for {row['task_id']} / {row['workflow']}")
        grouped[row["workflow"]].append(row)
    missing = WORKFLOWS - set(grouped)
    if missing:
        raise ValueError(f"missing completed workflow groups: {sorted(missing)}")
    summary = {}
    for name, items in grouped.items():
        times = [float(row["total_seconds"]) for row in items]
        is_complete = lambda row: row["completed"].lower() in {"true", "1", "yes"}
        summary[name] = {
            "trials": len(items), "mean_seconds": mean(times), "median_seconds": median(times),
            "mean_steps": mean(int(row["operation_steps"]) for row in items),
            "completion_rate": sum(is_complete(row) for row in items) / len(items),
            "correct_rate": sum(is_complete(row) and row["correct_target"].lower() in {"true", "1", "yes"} and row["correct_scope"].lower() in {"true", "1", "yes"} for row in items) / len(items),
            "export_complete_rate": sum(is_complete(row) and row["export_complete"].lower() in {"true", "1", "yes"} for row in items) / len(items),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"verified_real_rows": len(recorded), "by_workflow": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
