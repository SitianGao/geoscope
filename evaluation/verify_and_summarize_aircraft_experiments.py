"""Verify aircraft experiment masks, combine rows, and create the final comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from run_remotesam_aircraft_experiment import (
    CHECKPOINT_SHA256,
    FIELDS,
    PROTOCOL_VERSION,
    SERVICE_VERSION,
    summarize,
)
from run_remotesam_eval import MANIFEST_SHA256, file_sha256
from verify_dataset_manifest import verify as verify_dataset_manifest


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10)


def verify_row(
    row: dict[str, str], source: dict[str, str], prediction_path: Path, truth: np.ndarray
) -> None:
    image_id = row["image_id"]
    if row["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError(f"Protocol mismatch: {image_id}")
    if row["service_version"] != SERVICE_VERSION or row["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError(f"Model metadata mismatch: {image_id}")
    if row["image_sha256"] != source["image_sha256"] or row["ground_truth_sha256"] != source["ground_truth_sha256"]:
        raise ValueError(f"Image/GT metadata mismatch: {image_id}")
    if row["status"] == "failure":
        if not close(float(row["iou"]), 0.0) or not close(float(row["dice"]), 0.0):
            raise ValueError(f"Failure not scored zero: {image_id}")
        return
    if row["status"] != "success":
        raise ValueError(f"Unknown status: {image_id}")
    with Image.open(prediction_path) as image:
        if image.mode != "L" or image.size != (int(source["width"]), int(source["height"])):
            raise ValueError(f"Prediction geometry mismatch: {image_id}")
        pixels = np.asarray(image)
    if not np.isin(pixels, (0, 255)).all():
        raise ValueError(f"Non-binary prediction: {image_id}")
    prediction = pixels > 0
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    expected = {
        "prediction_foreground_pixels": int(np.count_nonzero(prediction)),
        "ground_truth_foreground_pixels": int(np.count_nonzero(truth)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
    for key, value in expected.items():
        if int(row[key]) != value:
            raise ValueError(f"{key} mismatch: {row['experiment_id']} / {image_id}")
    iou = tp / (tp + fp + fn)
    dice = 2 * tp / (2 * tp + fp + fn)
    if not close(float(row["iou"]), iou) or not close(float(row["dice"]), dice):
        raise ValueError(f"Metric mismatch: {row['experiment_id']} / {image_id}")


def tint(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    result = image.astype(np.float32).copy()
    color_array = np.asarray(color, dtype=np.float32)
    result[mask] = result[mask] * 0.45 + color_array * 0.55
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")


def error_overlay(image: np.ndarray, prediction: np.ndarray, truth: np.ndarray) -> Image.Image:
    result = image.astype(np.float32).copy()
    categories = (
        (prediction & truth, (0, 255, 0)),
        (prediction & ~truth, (255, 0, 0)),
        (~prediction & truth, (0, 128, 255)),
    )
    for mask, color in categories:
        color_array = np.asarray(color, dtype=np.float32)
        result[mask] = result[mask] * 0.35 + color_array * 0.65
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")


def panel_cell(image: Image.Image, label: str, size: tuple[int, int] = (800, 600)) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    copy = image.copy()
    copy.thumbnail((size[0], size[1] - 32), Image.Resampling.LANCZOS)
    x = (size[0] - copy.width) // 2
    y = 32 + (size[1] - 32 - copy.height) // 2
    canvas.paste(copy, (x, y))
    ImageDraw.Draw(canvas).text((10, 8), label, fill="black")
    return canvas


def save_case_panel(
    output_path: Path,
    source: dict[str, str],
    row: dict[str, str],
    image_path: Path,
    truth_path: Path,
    prediction_path: Path,
    case_label: str,
) -> None:
    with Image.open(image_path) as opened:
        original = opened.convert("RGB")
    with Image.open(truth_path) as opened:
        truth = np.asarray(opened) > 0
    with Image.open(prediction_path) as opened:
        prediction = np.asarray(opened) > 0
    array = np.asarray(original)
    cells = [
        panel_cell(original, "Original (visualization resize only)"),
        panel_cell(tint(array, truth, (0, 128, 255)), "Ground truth (blue)"),
        panel_cell(tint(array, prediction, (0, 255, 0)), "Prediction (green)"),
        panel_cell(
            error_overlay(array, prediction, truth),
            "Error: TP green / FP red / FN blue",
        ),
    ]
    panel = Image.new("RGB", (1600, 1260), "white")
    panel.paste(cells[0], (0, 60))
    panel.paste(cells[1], (800, 60))
    panel.paste(cells[2], (0, 660))
    panel.paste(cells[3], (800, 660))
    title = (
        f"{case_label}: {source['image_id']} | IoU={float(row['iou']):.4f} | "
        f"Dice={float(row['dice']):.4f} | {row['experiment_id']}"
    )
    ImageDraw.Draw(panel).text((12, 20), title, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--input", action="append", nargs=2, metavar=("RESULTS_CSV", "PREDICTIONS_DIR"), required=True
    )
    parser.add_argument("--baseline-results", required=True, type=Path)
    parser.add_argument("--combined-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--case-panels-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.combined_csv.exists() or args.summary_json.exists():
        raise FileExistsError("Refusing to overwrite combined outputs")
    if args.case_panels_dir.exists() and any(args.case_panels_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty case-panel directory")

    verify_dataset_manifest(args.manifest, args.data_root, MANIFEST_SHA256)
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        sources = {
            row["image_id"]: row
            for row in csv.DictReader(stream)
            if row["class_name"] == "plane"
        }
    truths: dict[str, np.ndarray] = {}
    for image_id, source in sources.items():
        with Image.open(args.data_root / source["ground_truth_path"]) as image:
            truths[image_id] = np.asarray(image) > 0

    rows: list[dict[str, str]] = []
    prediction_by_key: dict[tuple[str, str], Path] = {}
    for results_text, predictions_text in args.input:
        results_path = Path(results_text)
        predictions_dir = Path(predictions_text)
        with results_path.open(newline="", encoding="utf-8") as stream:
            input_rows = list(csv.DictReader(stream))
        for row in input_rows:
            key = row["experiment_id"], row["image_id"]
            if key in prediction_by_key:
                raise ValueError(f"Duplicate experiment/image row: {key}")
            if row["image_id"] not in sources:
                raise ValueError(f"Unknown image ID: {row['image_id']}")
            prediction_path = predictions_dir / f"{row['experiment_id']}__{row['image_id']}.png"
            verify_row(row, sources[row["image_id"]], prediction_path, truths[row["image_id"]])
            rows.append(row)
            prediction_by_key[key] = prediction_path

    experiment_ids = sorted({row["experiment_id"] for row in rows})
    for experiment_id in experiment_ids:
        image_ids = [row["image_id"] for row in rows if row["experiment_id"] == experiment_id]
        if len(image_ids) != 10 or set(image_ids) != set(sources):
            raise ValueError(f"Experiment does not cover frozen 10-image set: {experiment_id}")

    with args.baseline_results.open(newline="", encoding="utf-8") as stream:
        baseline_rows = [
            row for row in csv.DictReader(stream) if row["class_name"] == "plane"
        ]
    if len(baseline_rows) != 10:
        raise ValueError("Baseline does not contain the frozen 10 plane rows")
    baseline_iou = sum(float(row["iou"]) for row in baseline_rows) / 10
    baseline_dice = sum(float(row["dice"]) for row in baseline_rows) / 10

    args.combined_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.combined_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    run_meta = {
        "mode": "combined-verified-comparison",
        "manifest_sha256": file_sha256(args.manifest),
        "inputs": [Path(pair[0]).name for pair in args.input],
        "selection_note": (
            "The best configuration was selected on these same 10 diagnostic images; "
            "the improvement is not an independent generalization estimate."
        ),
    }
    summary = summarize(rows, run_meta["manifest_sha256"], run_meta)
    best_id = max(
        summary["experiments"],
        key=lambda name: summary["experiments"][name]["mean_iou_failure_as_zero"],
    )
    best_metric = summary["experiments"][best_id]
    best_rows = sorted(
        [row for row in rows if row["experiment_id"] == best_id],
        key=lambda row: float(row["iou"]),
    )
    representatives = {
        "worst": best_rows[0],
        "middle": best_rows[len(best_rows) // 2],
        "best": best_rows[-1],
    }
    summary["baseline"] = {
        "experiment_id": "sam-eval-v1.0/direct/all airplanes",
        "mean_iou": baseline_iou,
        "mean_dice": baseline_dice,
    }
    summary["best_experiment"] = {
        "experiment_id": best_id,
        **best_metric,
        "absolute_iou_gain": best_metric["mean_iou_failure_as_zero"] - baseline_iou,
        "relative_iou_gain": best_metric["mean_iou_failure_as_zero"] / baseline_iou - 1,
        "target_iou": 0.25,
        "target_met": best_metric["mean_iou_failure_as_zero"] >= 0.25,
    }
    summary["representative_cases"] = {
        label: {
            "image_id": row["image_id"],
            "iou": float(row["iou"]),
            "dice": float(row["dice"]),
            "tp": int(row["tp"]),
            "fp": int(row["fp"]),
            "fn": int(row["fn"]),
        }
        for label, row in representatives.items()
    }
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for label, row in representatives.items():
        source = sources[row["image_id"]]
        save_case_panel(
            args.case_panels_dir / f"{label}__{row['image_id']}.png",
            source,
            row,
            args.data_root / source["image_path"],
            args.data_root / source["ground_truth_path"],
            prediction_by_key[(best_id, row["image_id"])],
            label,
        )
    print(
        f"Verified {len(rows)} rows across {len(experiment_ids)} experiments. "
        f"Best={best_id}, mean IoU={best_metric['mean_iou_failure_as_zero']:.6f}"
    )


if __name__ == "__main__":
    main()
