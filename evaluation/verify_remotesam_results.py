"""Recompute RemoteSAM per-image metrics from saved prediction masks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
from PIL import Image


SMOKE_IDS = {
    "WHU_SAT1_18_0",
    "WHU_SAT1_25_1",
    "WHU_SAT1_6_2",
    "ISAID_VAL_P0257",
    "ISAID_VAL_P1492",
}
REPEAT_IDS = {"WHU_SAT1_18_0", "ISAID_VAL_P0257", "ISAID_VAL_P1143"}


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(
    manifest_path: Path,
    data_root: Path,
    results_path: Path,
    summary_path: Path,
    artifacts_dir: Path,
    run: str,
) -> None:
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        manifest = {row["image_id"]: row for row in csv.DictReader(stream)}
    with results_path.open(newline="", encoding="utf-8") as stream:
        results = list(csv.DictReader(stream))
    expected_ids = {"smoke": SMOKE_IDS, "repeat": REPEAT_IDS}.get(run, set(manifest))
    ids = [row["image_id"] for row in results]
    if len(ids) != len(expected_ids) or set(ids) != expected_ids:
        raise ValueError("Result IDs do not match the frozen run set")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["dataset_manifest_sha256"] != file_sha256(manifest_path):
        raise ValueError("Summary manifest hash mismatch")
    by_class: dict[str, list[dict]] = {"building": [], "plane": []}

    for row in results:
        image_id = row["image_id"]
        source = manifest[image_id]
        if row["class_name"] != source["class_name"]:
            raise ValueError(f"Class mismatch: {image_id}")
        if row["image_sha256"] != source["image_sha256"] or row["ground_truth_sha256"] != source["ground_truth_sha256"]:
            raise ValueError(f"Image/GT hash field mismatch: {image_id}")
        if row["service_version"] != summary["service_version"] or row["checkpoint_sha256"] != summary["checkpoint_sha256"]:
            raise ValueError(f"Model metadata mismatch: {image_id}")
        by_class[row["class_name"]].append(row)
        if row["status"] == "failure":
            if not close(float(row["iou"]), 0.0) or not close(float(row["dice"]), 0.0):
                raise ValueError(f"Failure not scored zero: {image_id}")
            continue
        if row["status"] != "success":
            raise ValueError(f"Unknown status: {image_id}")
        prediction_path = artifacts_dir / "predictions" / f"{image_id}.png"
        truth_path = data_root / source["ground_truth_path"]
        with Image.open(prediction_path) as prediction_image, Image.open(truth_path) as truth_image:
            if prediction_image.size != truth_image.size:
                raise ValueError(f"Prediction/GT size mismatch: {image_id}")
            if prediction_image.size != (int(source["width"]), int(source["height"])):
                raise ValueError(f"Prediction/source size mismatch: {image_id}")
            if prediction_image.mode != "L":
                raise ValueError(f"Prediction not grayscale: {image_id}")
            prediction_pixels = np.asarray(prediction_image)
            truth_pixels = np.asarray(truth_image)
        if not np.isin(prediction_pixels, (0, 255)).all():
            raise ValueError(f"Non-binary prediction: {image_id}")
        prediction = prediction_pixels > 0
        truth = truth_pixels > 0
        tp = int(np.count_nonzero(prediction & truth))
        fp = int(np.count_nonzero(prediction & ~truth))
        fn = int(np.count_nonzero(~prediction & truth))
        expected = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "prediction_foreground_pixels": int(np.count_nonzero(prediction)),
            "ground_truth_foreground_pixels": int(np.count_nonzero(truth)),
        }
        for key, value in expected.items():
            if int(row[key]) != value:
                raise ValueError(f"{key} mismatch: {image_id}")
        if row["binary_valid"].lower() != "true" or int(row["output_width"]) != int(source["width"]) or int(row["output_height"]) != int(source["height"]):
            raise ValueError(f"Output validity fields mismatch: {image_id}")
        if (row["empty_mask"].lower() == "true") != (expected["prediction_foreground_pixels"] == 0):
            raise ValueError(f"Empty-mask field mismatch: {image_id}")
        iou = tp / (tp + fp + fn)
        dice = 2 * tp / (2 * tp + fp + fn)
        if not close(float(row["iou"]), iou) or not close(float(row["dice"]), dice):
            raise ValueError(f"IoU/Dice mismatch: {image_id}")

    def verify_group(items: list[dict], recorded: dict) -> None:
        if not items:
            return
        successes = [item for item in items if item["status"] == "success"]
        timings = [float(item["service_inference_ms"]) for item in successes if item["service_inference_ms"]]
        expected_numbers = {
            "images": len(items),
            "successes": len(successes),
            "failures": len(items) - len(successes),
            "empty_masks": sum(item["empty_mask"].lower() == "true" for item in successes),
            "failure_rate": (len(items) - len(successes)) / len(items),
            "mean_iou_all_failure_as_zero": statistics.mean(float(item["iou"]) for item in items),
            "mean_dice_all_failure_as_zero": statistics.mean(float(item["dice"]) for item in items),
            "mean_iou_success_only": statistics.mean(float(item["iou"]) for item in successes) if successes else None,
            "mean_dice_success_only": statistics.mean(float(item["dice"]) for item in successes) if successes else None,
            "mean_service_inference_ms": statistics.mean(timings) if timings else None,
            "median_service_inference_ms": statistics.median(timings) if timings else None,
        }
        for key, value in expected_numbers.items():
            if value is None:
                if recorded[key] is not None:
                    raise ValueError(f"Summary {key} mismatch")
            elif not close(float(recorded[key]), float(value)):
                raise ValueError(f"Summary {key} mismatch")

    verify_group(results, summary["overall"])
    for class_name, items in by_class.items():
        if items:
            verify_group(items, summary["by_class"][class_name])
    print(f"Recomputed {len(results)} {run} cases and matching summary metrics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--run", choices=("smoke", "full", "repeat"), required=True)
    args = parser.parse_args()
    verify(args.manifest, args.data_root, args.results_csv, args.summary_json, args.artifacts_dir, args.run)


if __name__ == "__main__":
    main()
