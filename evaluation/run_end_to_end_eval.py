"""Run one frozen batch through RemoteAgent -> Harness -> RemoteSAM -> metrics.

This command requires both real services and the licensed local evaluation assets.
It never substitutes demo masks. Failed tasks remain failed and receive zero IoU.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import shutil
from pathlib import Path
from statistics import mean

from PIL import Image

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import server  # noqa: E402


FIELDS = [
    "task_id", "image_id", "class_name", "query", "status", "agent_status",
    "tool_name", "service_prompt", "harness_target", "harness_side", "width",
    "height", "tp", "fp", "fn", "iou", "dice", "duration_ms", "failure_reason",
    "artifact_dir",
]


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def confusion(prediction: Image.Image, truth: Image.Image) -> tuple[int, int, int, float, float]:
    pred = prediction.convert("L").point(lambda value: 1 if value else 0)
    gt = truth.convert("L").point(lambda value: 1 if value else 0)
    if pred.size != gt.size:
        raise ValueError(f"prediction/ground-truth size mismatch: {pred.size} != {gt.size}")
    pairs = zip(pred.getdata(), gt.getdata())
    tp = fp = fn = 0
    for p, g in pairs:
        tp += bool(p and g)
        fp += bool(p and not g)
        fn += bool(g and not p)
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return tp, fp, fn, iou, dice


def query_for(row: dict[str, str]) -> str:
    return "提取整幅影像中的所有建筑。" if row["project_target"] == "building" else "提取整幅影像中的所有飞机。"


def select_cases(rows: list[dict[str, object]], artifact_root: Path, case_root: Path) -> list[dict[str, object]]:
    successes = sorted((row for row in rows if row["status"] in {"completed", "needs_review"}), key=lambda row: float(row["iou"]))
    failures = [row for row in rows if row["status"] not in {"completed", "needs_review"}]
    if not successes and not failures:
        return []
    picks = []
    if failures:
        picks.append(("failure", failures[0]))
    elif successes:
        picks.append(("low_quality", successes[0]))
    if successes:
        picks.extend((("medium", successes[len(successes) // 2]), ("success", successes[-1])))
    case_root.mkdir(parents=True, exist_ok=True)
    selected = []
    for label, row in picks:
        source = artifact_root / str(row["artifact_dir"])
        destination = case_root / f"{label}_{row['task_id']}_{row['image_id']}"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("original.png", "ground_truth.png", "mask.png", "overlay.png", "result.json", "run_log.json"):
            candidate = source / name
            if candidate.exists():
                shutil.copy2(candidate, destination / name)
        selected.append({"level": label, "task_id": row["task_id"], "image_id": row["image_id"],
                         "iou": row["iou"], "dice": row["dice"], "path": str(destination)})
    return selected


def run(args: argparse.Namespace) -> dict[str, object]:
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if args.limit:
        manifest = manifest[: args.limit]
    if not manifest:
        raise ValueError("manifest is empty")

    artifact_root = args.artifact_dir.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    server.DATA = artifact_root
    server.SESSIONS.clear()
    rows: list[dict[str, object]] = []
    raw_path = args.output.with_suffix(".jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with raw_path.open("w", encoding="utf-8") as raw:
        for index, item in enumerate(manifest, 1):
            image_path = (args.data_root / item["image_path"]).resolve()
            truth_path = (args.data_root / item["ground_truth_path"]).resolve()
            if not image_path.is_file() or not truth_path.is_file():
                raise FileNotFoundError(f"missing licensed asset for {item['image_id']}: {image_path} or {truth_path}")
            session = server.new_session(uploaded=image_data_url(image_path), name=item["image_id"])
            live = server.SESSIONS[session["id"]]
            query = query_for(item)
            result = server.run_task(live, {"query": query, "mode": "live", "force_perception": True})
            run_folder = artifact_root / session["id"] / result["id"]
            shutil.copy2(truth_path, run_folder / "ground_truth.png")
            tp = fp = fn = 0
            iou = dice = 0.0
            if result.get("mask_url") and (run_folder / "mask.png").is_file():
                tp, fp, fn, iou, dice = confusion(Image.open(run_folder / "mask.png"), Image.open(truth_path))
            decision = result.get("agent_decision") or {}
            call = decision.get("tool_call") or {}
            request = (result.get("task") or {}).get("service_request") or {}
            row = {
                "task_id": f"E2E{index:02d}", "image_id": item["image_id"],
                "class_name": item["project_target"], "query": query, "status": result["status"],
                "agent_status": decision.get("status", ""), "tool_name": call.get("name", ""),
                "service_prompt": request.get("text") or ",".join(request.get("classes", [])),
                "harness_target": (result.get("task") or {}).get("target", ""),
                "harness_side": (result.get("task") or {}).get("side", ""),
                "width": session["width"], "height": session["height"], "tp": tp, "fp": fp, "fn": fn,
                "iou": iou, "dice": dice, "duration_ms": result.get("duration_ms", 0),
                "failure_reason": "" if result["status"] in {"completed", "needs_review"} else result.get("message", ""),
                "artifact_dir": f"{session['id']}/{result['id']}",
            }
            rows.append(row)
            raw.write(json.dumps({"manifest": item, "result": result, "row": row}, ensure_ascii=False) + "\n")

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    completed = [row for row in rows if row["status"] in {"completed", "needs_review"}]
    selected = select_cases(rows, artifact_root, args.case_dir.resolve())
    summary = {
        "protocol": "geoscope-end-to-end-v1.0",
        "total_tasks": len(rows),
        "completed_tasks": len(completed),
        "completion_rate": len(completed) / len(rows),
        "mean_iou_failure_as_zero": mean(float(row["iou"]) for row in rows),
        "mean_dice_failure_as_zero": mean(float(row["dice"]) for row in rows),
        "aircraft_service_prompts": sorted({str(row["service_prompt"]) for row in rows if row["class_name"] == "aircraft"}),
        "selected_cases": selected,
        "evidence_boundary": "Only a completed run of this script with both real services and local ground truth is an end-to-end result.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="root containing manifest image_path and ground_truth_path")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("dataset_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results") / "end_to_end_results.csv")
    parser.add_argument("--summary", type=Path, default=Path(__file__).with_name("results") / "end_to_end_summary.json")
    parser.add_argument("--artifact-dir", type=Path, default=Path(__file__).with_name("results") / "end_to_end_artifacts")
    parser.add_argument("--case-dir", type=Path, default=Path(__file__).with_name("results") / "end_to_end_cases")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
