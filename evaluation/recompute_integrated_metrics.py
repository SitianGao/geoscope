"""Recompute the committed RemoteAgent and RemoteSAM headline metrics.

This script uses only versioned CSV evidence. It does not contact either model
service and therefore lets a reviewer check every aggregate number offline.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def _rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _same(actual: float, expected: float, tolerance: float = 1e-6) -> bool:
    return math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance)


def verify_agent() -> dict:
    rows = _rows("agent_full_results.csv")
    source = json.loads((RESULTS / "agent_summary.json").read_text(encoding="utf-8"))
    attempts = [r for r in rows if r["agent_status"] in {"tool_call", "rejected_tool"}]
    segmented = [r for r in rows if r["outcome"] == "segment"]
    with_target = [r for r in segmented if r["expected_target"]]
    with_side = [r for r in segmented if r["expected_side"]]
    clarify = [r for r in rows if r["expected_task_type"] == "clarify"]
    latency = sorted(float(r["decision_latency_ms"]) for r in rows if float(r["decision_latency_ms"]) > 0)
    calculated = {
        "total_tasks": len(rows),
        "task_routing_accuracy": sum(r["routing_correct"].lower() == "true" for r in rows) / len(rows),
        "legal_tool_call_rate": sum(r["agent_status"] == "tool_call" for r in attempts) / len(attempts),
        "target_category_accuracy": sum(r["resolved_target"] == r["expected_target"] for r in with_target) / len(with_target),
        "spatial_condition_accuracy": sum(r["resolved_side"] == r["expected_side"] for r in with_side) / len(with_side),
        "clarification_recognition_rate": sum(r["outcome"] != "segment" for r in clarify) / len(clarify),
        "avg_decision_latency_ms": mean(latency),
        "median_decision_latency_ms": median(latency),
        "outcome_counts": dict(Counter(r["outcome"] for r in rows)),
        "agent_status_counts": dict(Counter(r["agent_status"] for r in rows)),
    }
    expected = source["metrics"]
    checks = {
        "total_tasks": calculated["total_tasks"] == source["total_tasks"],
        **{name: _same(calculated[name], expected[name], 5e-5)
           for name in ("task_routing_accuracy", "legal_tool_call_rate",
                        "target_category_accuracy", "spatial_condition_accuracy",
                        "clarification_recognition_rate")},
        "avg_decision_latency_ms": _same(calculated["avg_decision_latency_ms"], expected["avg_decision_latency_ms"], .05),
        "median_decision_latency_ms": _same(calculated["median_decision_latency_ms"], expected["median_decision_latency_ms"], .05),
        "outcome_counts": calculated["outcome_counts"] == source["outcome_counts"],
        "agent_status_counts": calculated["agent_status_counts"] == source["agent_status_counts"],
    }
    if not all(checks.values()):
        raise ValueError(f"RemoteAgent aggregate mismatch: {checks}")
    return {"verified": True, "checks": checks, "metrics": calculated}


def verify_sam() -> dict:
    rows = _rows("remotesam_full_results.csv")
    source = json.loads((RESULTS / "remotesam_summary.json").read_text(encoding="utf-8"))
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    row_checks = []
    for row in rows:
        tp, fp, fn = (int(row[key]) for key in ("tp", "fp", "fn"))
        iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
        dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
        valid = (_same(iou, float(row["iou"]), 1e-12)
                 and _same(dice, float(row["dice"]), 1e-12)
                 and int(row["output_width"]) == int(row["width"])
                 and int(row["output_height"]) == int(row["height"])
                 and row["binary_valid"].lower() == "true")
        row_checks.append(valid)
        groups[row["class_name"]].append(row)

    def aggregate(items: list[dict[str, str]]) -> dict:
        successful = [r for r in items if r["status"] == "success"]
        return {
            "images": len(items),
            "successes": len(successful),
            "mean_iou": mean(float(r["iou"]) if r["status"] == "success" else 0 for r in items),
            "mean_dice": mean(float(r["dice"]) if r["status"] == "success" else 0 for r in items),
            "mean_service_inference_ms": mean(float(r["service_inference_ms"]) for r in successful),
            "median_service_inference_ms": median(float(r["service_inference_ms"]) for r in successful),
        }

    overall = aggregate(rows)
    by_class = {name: aggregate(items) for name, items in groups.items()}
    checks = {
        "row_formulas_and_contract": all(row_checks),
        "images": overall["images"] == source["overall"]["images"],
        "successes": overall["successes"] == source["overall"]["successes"],
        "mean_iou": _same(overall["mean_iou"], source["overall"]["mean_iou_all_failure_as_zero"]),
        "mean_dice": _same(overall["mean_dice"], source["overall"]["mean_dice_all_failure_as_zero"]),
    }
    for name, values in by_class.items():
        saved = source["by_class"][name]
        checks[f"{name}_images"] = values["images"] == saved["images"]
        checks[f"{name}_mean_iou"] = _same(values["mean_iou"], saved["mean_iou_all_failure_as_zero"])
        checks[f"{name}_mean_dice"] = _same(values["mean_dice"], saved["mean_dice_all_failure_as_zero"])
    if not all(checks.values()):
        raise ValueError(f"RemoteSAM aggregate mismatch: {checks}")
    return {"verified": True, "checks": checks, "metrics": overall, "by_class": by_class,
            "protocol_version": source["protocol_version"],
            "dataset_manifest_sha256": source["dataset_manifest_sha256"],
            "checkpoint_sha256": source["checkpoint_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline verification of integrated competition metrics")
    parser.add_argument("--write", action="store_true", help="write evaluation/results/captain_integrated_summary.json")
    args = parser.parse_args()
    result = {
        "verification_scope": "offline aggregate recomputation from committed per-task/per-image CSV evidence",
        "remoteagent": verify_agent(),
        "remotesam": verify_sam(),
        "evidence_boundaries": {
            "end_to_end_precision": "not measured by these separate evaluations",
            "user_study": "not completed",
            "workflow_comparison": "not completed",
        },
    }
    if args.write:
        output = RESULTS / "captain_integrated_summary.json"
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"verified and wrote {output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
