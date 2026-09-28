"""Batch RemoteAgent decision evaluation for the GeoScope competition (v0.5.0).

This is 队员 A's evaluation entry. It replays a natural-language task set
through the *same live pipeline the workbench uses* — ``server.py``'s
task-routing + ``agent_bridge``'s ``AgentTurn`` / ``resolve_scope`` /
``segmentation_plan`` — against the real RemoteAgent model, and records, per
task, the decision the model produced and whether the Harness accepted it.

It is deliberately honest about the model/Harness boundary:

* ``requires_dense_result`` / ``resolve_scope`` / the "那个/这片" reference
  rule are deterministic Harness pre-checks. They can fail a task closed
  *before* the model is called; those rows are marked ``deterministic_clarify``.
* ``AgentTurn.plan(...)`` returns the raw model decision (answer / legal tool /
  illegal tool / unparsed). It never executes a tool.
* ``segmentation_plan(...)`` is the real Harness validation that binds the
  current image and resolves target / scope. "passed Harness" == produced a
  ``segment`` plan.

Usage (from the repository root)::

    python evaluation/agent_eval.py --limit 10 --out agent_smoke_results.csv
    python evaluation/agent_eval.py --out agent_full_results.csv

Environment (the real vLLM endpoint must include ``/v1`` because the current
``WorkbenchAgent`` appends ``/chat/completions`` directly)::

    GEO_AGENT_BASE_URL=http://127.0.0.1:18000/v1
    GEO_AGENT_MODEL=RemoteAgent
    GEO_SERVICE_PROXY_MODE=direct
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import agent_bridge  # noqa: E402
from agent_bridge import (  # noqa: E402
    ALLOWED_TOOLS,
    AgentTurn,
    Clarification,
    decision_record,
    resolve_scope,
    segmentation_plan,
)

# Reuse the workbench's own routing helpers (module import is side-effect free
# apart from creating the already-present experiments/ directory).
import server as _server  # noqa: E402
from server import requires_dense_result, resolve_invert  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
IMAGES_DIR = EVAL_DIR / "images"
MANIFEST_PATH = EVAL_DIR / "agent_dataset_manifest.csv"
TASK_SET_PATH = EVAL_DIR / "task_set.csv"
RESULTS_DIR = EVAL_DIR / "results"

RESULT_FIELDS = [
    "task_id",
    "category",
    "image_id",
    "request",
    "expected_task_type",
    "expected_target",
    "expected_side",
    "task_route",
    "agent_status",
    "tool_name",
    "outcome",
    "resolved_target",
    "resolved_side",
    "decision_latency_ms",
    "routing_correct",
    "failure_reason",
]

AMBIGUOUS_REF_RE = re.compile(r"那个|那片|那块|这块")


def load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def prior_context(task: Dict[str, str]) -> Dict[str, Any]:
    """Simulate the session context a follow-up modification would inherit."""
    if task["category"] == "modify":
        return {"target": "building", "side": "right", "invert": False}
    return {}


def evaluate_task(task: Dict[str, str], image_path: Path) -> Dict[str, Any]:
    query = task["request"]
    context = prior_context(task)

    row: Dict[str, Any] = {
        "task_id": task["task_id"],
        "category": task["category"],
        "image_id": task["image_id"],
        "request": query,
        "expected_task_type": task["expected_task_type"],
        "expected_target": task["expected_target"],
        "expected_side": task["expected_side"],
    }

    # --- deterministic Harness pre-checks (may fail closed before the model) ---
    dense_task = requires_dense_result(query, context)
    invert = resolve_invert(query, context) if dense_task else False
    row["task_route"] = "dense" if dense_task else "internal"

    try:
        side, scope_source = resolve_scope(query, context, None)
    except Clarification as exc:
        return finish_row(row, outcome="deterministic_clarify", agent_status="n/a",
                          tool_name="", latency=0.0, reason=str(exc))
    except ValueError as exc:
        return finish_row(row, outcome="deterministic_clarify", agent_status="n/a",
                          tool_name="", latency=0.0, reason=str(exc))

    if AMBIGUOUS_REF_RE.search(query):
        return finish_row(row, outcome="deterministic_clarify", agent_status="n/a",
                          tool_name="", latency=0.0,
                          reason="模糊指代（那个/这片）：请明确目标类别与像素范围")

    # --- real model decision ---
    t0 = time.perf_counter()
    turn = AgentTurn(query, image_path, context, side, scope_source,
                     task_route="dense" if dense_task else "internal", invert=invert)
    decision = turn.decision
    latency_ms = (time.perf_counter() - t0) * 1000.0

    tool_name = decision.tool_call.name if decision.tool_call is not None else ""
    raw = decision.raw_response or ""

    # --- Harness validation (mirrors server.py _run_task) ---
    if decision.status == "completed":
        if dense_task:
            row = finish_row(row, outcome="dense_answered_text", agent_status="completed",
                             tool_name="", latency=latency_ms,
                             reason="需要像素结果但 RemoteAgent 只返回了文字回答")
        else:
            row = finish_row(row, outcome="scene", agent_status="completed",
                             tool_name="", latency=latency_ms, reason="")
    elif decision.status == "tool_call":
        if not dense_task:
            row = finish_row(row, outcome="internal_called_tool", agent_status="tool_call",
                             tool_name=tool_name, latency=latency_ms,
                             reason="看图问答任务不应调用像素分割工具")
        else:
            try:
                plan = segmentation_plan(decision, image_path, side, scope_source, invert=invert)
                row = finish_row(row, outcome="segment", agent_status="tool_call",
                                 tool_name=tool_name, latency=latency_ms, reason="",
                                 resolved_target=plan["target"], resolved_side=plan["side"])
            except Clarification as exc:
                row = finish_row(row, outcome="reject_args", agent_status="tool_call",
                                 tool_name=tool_name, latency=latency_ms, reason=str(exc))
            except ValueError as exc:
                row = finish_row(row, outcome="reject_args", agent_status="tool_call",
                                 tool_name=tool_name, latency=latency_ms, reason=str(exc))
    else:  # rejected_tool / unparsed
        row = finish_row(row, outcome=decision.status, agent_status=decision.status,
                         tool_name=tool_name, latency=latency_ms,
                         reason=f"模型决策未通过校验（{decision.status}）")

    row["raw_response"] = raw
    row["decision_record"] = decision_record(decision)
    return row


def finish_row(row: Dict[str, Any], *, outcome: str, agent_status: str, tool_name: str,
               latency: float, reason: str, resolved_target: str = "",
               resolved_side: str = "") -> Dict[str, Any]:
    row["outcome"] = outcome
    row["agent_status"] = agent_status
    row["tool_name"] = tool_name
    row["decision_latency_ms"] = round(latency, 1)
    row["resolved_target"] = resolved_target
    row["resolved_side"] = resolved_side
    row["routing_correct"] = routing_correct(row["expected_task_type"], outcome)
    row["failure_reason"] = reason if not row["routing_correct"] else ""
    return row


def routing_correct(expected_type: str, outcome: str) -> bool:
    if expected_type == "answer":
        return outcome == "scene"
    if expected_type == "tool_call":
        return outcome == "segment"
    if expected_type == "clarify":
        return outcome != "segment"  # fail-closed: no executable segmentation
    return False


def compute_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)

    def frac(cond) -> Optional[float]:
        if not total:
            return None
        return round(sum(1 for r in rows if cond(r)) / total, 6)

    tool_attempt = [r for r in rows if r["agent_status"] in ("tool_call", "rejected_tool")]
    legal_tool = [r for r in tool_attempt if r["agent_status"] == "tool_call"]
    segment = [r for r in rows if r["outcome"] == "segment"]
    segment_target = [r for r in segment if r["expected_target"]]
    segment_side = [r for r in segment if r["expected_side"]]
    clarify = [r for r in rows if r["expected_task_type"] == "clarify"]
    latencies = sorted(r["decision_latency_ms"] for r in rows if r["decision_latency_ms"])

    outcome_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    reason_counts: Dict[str, int] = {}
    for r in rows:
        outcome_counts[r["outcome"]] = outcome_counts.get(r["outcome"], 0) + 1
        status_counts[r["agent_status"]] = status_counts.get(r["agent_status"], 0) + 1
        if r["failure_reason"]:
            reason_counts[r["failure_reason"]] = reason_counts.get(r["failure_reason"], 0) + 1

    return {
        "config": {
            "model": "RemoteAgent (AIMGroup/RemoteAgent, Qwen2.5-VL, BF16, vLLM 0.15.1)",
            "endpoint": os.environ.get("GEO_AGENT_BASE_URL", "http://127.0.0.1:18000/v1"),
            "allowed_tools": sorted(ALLOWED_TOOLS),
            "proxy_mode": os.environ.get("GEO_SERVICE_PROXY_MODE", "direct"),
            "note": "路由与校验复用工作台 server.py / agent_bridge 的真实代码",
        },
        "total_tasks": total,
        "metrics": {
            "task_routing_accuracy": frac(lambda r: r["routing_correct"]),
            "legal_tool_call_rate": round(len(legal_tool) / len(tool_attempt), 6) if tool_attempt else None,
            "target_category_accuracy": round(
                sum(1 for r in segment_target if r["resolved_target"] == r["expected_target"])
                / len(segment_target), 6
            ) if segment_target else None,
            "spatial_condition_accuracy": round(
                sum(1 for r in segment_side if r["resolved_side"] == r["expected_side"])
                / len(segment_side), 6
            ) if segment_side else None,
            "clarification_recognition_rate": round(
                sum(1 for r in clarify if r["outcome"] != "segment") / len(clarify), 6
            ) if clarify else None,
            "avg_decision_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "median_decision_latency_ms": round(median(latencies), 1) if latencies else None,
        },
        "outcome_counts": outcome_counts,
        "agent_status_counts": status_counts,
        "failure_reason_counts": reason_counts,
        "baseline_agent_vs_harness": baseline_vs_harness(rows),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def baseline_vs_harness(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Raw model decisions (baseline) vs what the Harness actually did with them."""
    baseline = {
        "direct_answers": sum(1 for r in rows if r["agent_status"] == "completed"),
        "legal_tool_calls": sum(1 for r in rows if r["agent_status"] == "tool_call"),
        "illegal_tool_calls": sum(1 for r in rows if r["agent_status"] == "rejected_tool"),
        "unparsed": sum(1 for r in rows if r["agent_status"] == "unparsed"),
        "not_called_deterministic": sum(1 for r in rows if r["agent_status"] == "n/a"),
    }
    harness = {
        "executed_segment": sum(1 for r in rows if r["outcome"] == "segment"),
        "scene_answer": sum(1 for r in rows if r["outcome"] == "scene"),
        "blocked_illegal_tool": baseline["illegal_tool_calls"],
        "blocked_unparsed": baseline["unparsed"],
        "rejected_args": sum(1 for r in rows if r["outcome"] == "reject_args"),
        "dense_answered_text": sum(1 for r in rows if r["outcome"] == "dense_answered_text"),
        "internal_called_tool": sum(1 for r in rows if r["outcome"] == "internal_called_tool"),
        "deterministic_clarify": sum(1 for r in rows if r["outcome"] == "deterministic_clarify"),
    }
    return {"baseline_agent": baseline, "harness": harness}


def write_results(rows: List[Dict[str, Any]], csv_path: Path, raw_path: Path, summary: Dict[str, Any],
                  summary_path: Optional[Path] = None, comparison_path: Optional[Path] = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in RESULT_FIELDS})
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary_path = summary_path or csv_path.with_name(csv_path.stem + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # baseline Agent raw action vs Harness outcome, one row per task
    if comparison_path is None:
        comparison_path = csv_path.with_name(csv_path.stem + "_baseline_vs_harness.csv")
    with open(comparison_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task_id", "category", "agent_status", "tool_name", "outcome",
            "harness_blocked", "harness_block_reason"])
        writer.writeheader()
        for r in rows:
            blocked = r["agent_status"] in ("rejected_tool", "unparsed") or \
                r["outcome"] in ("reject_args", "dense_answered_text", "internal_called_tool",
                                 "deterministic_clarify", "rejected_tool", "unparsed")
            writer.writerow({
                "task_id": r.get("task_id"), "category": r.get("category"),
                "agent_status": r.get("agent_status"), "tool_name": r.get("tool_name"),
                "outcome": r.get("outcome"),
                "harness_blocked": bool(blocked),
                "harness_block_reason": r.get("failure_reason", ""),
            })
    print(f"Wrote {len(rows)} rows -> {csv_path}")
    print(f"Wrote raw trace -> {raw_path}")
    print(f"Wrote summary -> {summary_path}")
    print(f"Wrote baseline-vs-harness -> {comparison_path}")


def recompute_from_raw(raw_path: Path, out_csv: Path, summary_name: str) -> None:
    """Rebuild CSV + summary + baseline-vs-harness from an already-recorded raw
    trace, without calling the model. This is the re-computation path used by the
    captain to verify that every aggregate figure is derivable from the raw rows."""
    rows: List[Dict[str, Any]] = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    summary = compute_summary(rows)
    out_raw = out_csv.with_name(out_csv.stem + "_raw.jsonl")
    write_results(rows, out_csv, out_raw, summary,
                  summary_path=out_csv.parent / summary_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch RemoteAgent decision evaluation")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N tasks (0 = all)")
    parser.add_argument("--out", type=str, default="agent_full_results.csv", help="output CSV basename")
    parser.add_argument("--summary", type=str, default=None,
                        help="summary JSON basename (default: <out stem>_summary.json)")
    parser.add_argument("--from-raw", type=str, default=None,
                        help="recompute CSV/summary/comparison from an existing raw jsonl (no model calls)")
    parser.add_argument("--seed-tasks", type=str, default=None,
                        help="comma-separated task_ids to run instead (ad-hoc probes)")
    args = parser.parse_args()

    if args.from_raw:
        raw_path = RESULTS_DIR / args.from_raw
        if not raw_path.is_file():
            raise SystemExit(f"raw trace not found: {raw_path}")
        out_csv = RESULTS_DIR / args.out
        summary_name = args.summary or (out_csv.stem + "_summary.json")
        recompute_from_raw(raw_path, out_csv, summary_name)
        return

    manifest = {r["image_id"]: r for r in load_csv(MANIFEST_PATH)}
    tasks = load_csv(TASK_SET_PATH)
    if args.seed_tasks:
        wanted = {t.strip() for t in args.seed_tasks.split(",") if t.strip()}
        tasks = [t for t in tasks if t["task_id"] in wanted]
    elif args.limit > 0:
        tasks = tasks[: args.limit]

    rows: List[Dict[str, Any]] = []
    for i, task in enumerate(tasks, 1):
        image_id = task["image_id"]
        if image_id not in manifest:
            raise SystemExit(f"task {task['task_id']} references unknown image_id {image_id}")
        image_path = IMAGES_DIR / manifest[image_id]["filename"]
        if not image_path.is_file():
            raise SystemExit(f"image file missing: {image_path} (run prepare_images.py)")
        print(f"[{i}/{len(tasks)}] {task['task_id']} {task['category']}: {task['request'][:36]}", flush=True)
        try:
            row = evaluate_task(task, image_path)
        except Exception as exc:  # keep the run going; record the failure honestly
            row = {
                "task_id": task["task_id"], "category": task["category"], "image_id": task["image_id"],
                "request": task["request"], "expected_task_type": task["expected_task_type"],
                "expected_target": task["expected_target"], "expected_side": task["expected_side"],
                "task_route": "?", "agent_status": "error", "tool_name": "", "outcome": "error",
                "resolved_target": "", "resolved_side": "", "decision_latency_ms": 0.0,
                "routing_correct": False, "failure_reason": f"eval_error: {type(exc).__name__}: {exc}",
                "raw_response": "",
            }
        rows.append(row)

    summary = compute_summary(rows)
    out_csv = RESULTS_DIR / args.out
    out_raw = out_csv.with_name(out_csv.stem + "_raw.jsonl")
    summary_path = RESULTS_DIR / args.summary if args.summary else None
    write_results(rows, out_csv, out_raw, summary, summary_path=summary_path)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
