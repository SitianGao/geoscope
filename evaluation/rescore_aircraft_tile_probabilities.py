"""Rescore saved tiled probability maps without repeating RemoteSAM inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from run_remotesam_aircraft_experiment import (
    FIELDS,
    PROTOCOL_VERSION,
    Experiment,
    score,
    summarize,
)
from run_remotesam_eval import MANIFEST_SHA256, file_sha256
from verify_dataset_manifest import verify as verify_dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--probabilities-dir", required=True, type=Path)
    parser.add_argument("--source-results", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=float, nargs="+")
    parser.add_argument("--results-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--predictions-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.results_csv.exists() or args.summary_json.exists():
        raise FileExistsError("Refusing to overwrite an existing result or summary")
    if args.predictions_dir.exists() and any(args.predictions_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty predictions directory")
    if any(not 0 < threshold < 1 for threshold in args.thresholds):
        raise ValueError("Probability thresholds must be between 0 and 1")
    verify_dataset_manifest(args.manifest, args.data_root, MANIFEST_SHA256)

    with args.manifest.open(newline="", encoding="utf-8") as stream:
        sources = {
            row["image_id"]: row
            for row in csv.DictReader(stream)
            if row["class_name"] == "plane"
        }
    with args.source_results.open(newline="", encoding="utf-8") as stream:
        source_result_rows = list(csv.DictReader(stream))
    timing_by_image: dict[str, dict[str, object]] = {}
    for row in source_result_rows:
        timing_by_image.setdefault(
            row["image_id"],
            {
                "request_count": int(row["request_count"]),
                "client_wall_ms": float(row["client_wall_ms"]),
                "service_inference_ms": float(row["service_inference_ms"]),
                "service_total_ms": float(row["service_total_ms"]),
                "peak_vram_mb": float(row["peak_vram_mb"]),
            },
        )
    if set(timing_by_image) != set(sources):
        raise ValueError("Source results do not cover the frozen 10 aircraft images")

    rows: list[dict[str, object]] = []
    args.predictions_dir.mkdir(parents=True, exist_ok=True)
    for image_id, source in sources.items():
        with Image.open(args.probabilities_dir / f"{image_id}.png") as image:
            probability = np.asarray(image).astype(np.float32) / 255.0
        with Image.open(args.data_root / source["ground_truth_path"]) as image:
            truth = np.asarray(image) > 0
        if probability.shape != truth.shape:
            raise ValueError(f"Probability/GT shape mismatch: {image_id}")
        for threshold in sorted(set(args.thresholds)):
            experiment = Experiment(
                f"tile_1024_overlap_256_p{str(threshold).replace('.', '_')}",
                "tile",
                "all planes",
                tile_size=1024,
                overlap=256,
                threshold=threshold,
            )
            prediction = probability >= threshold
            result = score(
                source, experiment, prediction, truth, timing_by_image[image_id]
            )
            rows.append(result)
            Image.fromarray(prediction.astype(np.uint8) * 255, mode="L").save(
                args.predictions_dir / f"{experiment.experiment_id}__{image_id}.png"
            )
            print(
                f"{experiment.experiment_id} {image_id}: "
                f"IoU={result['iou']:.4f} Dice={result['dice']:.4f}",
                flush=True,
            )

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.results_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    run_meta = {
        "mode": "probability-rescore",
        "manifest_sha256": file_sha256(args.manifest),
        "source_results": args.source_results.name,
        "note": (
            "No new model calls. Metrics reuse the inference timing from the saved "
            "1024/256 tiled run and threshold its quantized probability maps."
        ),
    }
    summary = summarize(rows, run_meta["manifest_sha256"], run_meta)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["experiments"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
