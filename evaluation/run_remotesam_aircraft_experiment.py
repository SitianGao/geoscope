"""Run fixed-set RemoteSAM aircraft prompt, upscale, or tiled experiments."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from run_remotesam_eval import (
    CHECKPOINT_SHA256,
    MANIFEST_SHA256,
    SERVICE_VERSION,
    file_sha256,
    preflight,
)
from verify_dataset_manifest import verify as verify_dataset_manifest


PROTOCOL_VERSION = "sam-aircraft-v2.0"
PROMPTS = ("all airplanes", "airplanes", "all planes", "aircraft", "all aircraft")
FIELDS = (
    "protocol_version",
    "experiment_id",
    "strategy",
    "prompt",
    "scale",
    "tile_size",
    "overlap",
    "probability_threshold",
    "image_id",
    "dataset",
    "width",
    "height",
    "image_sha256",
    "ground_truth_sha256",
    "status",
    "failure_code",
    "failure_detail",
    "request_count",
    "output_width",
    "output_height",
    "binary_valid",
    "empty_mask",
    "prediction_foreground_pixels",
    "ground_truth_foreground_pixels",
    "tp",
    "fp",
    "fn",
    "iou",
    "dice",
    "client_wall_ms",
    "service_inference_ms",
    "service_total_ms",
    "peak_vram_mb",
    "service_version",
    "checkpoint_sha256",
)


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    strategy: str
    prompt: str
    scale: float = 1.0
    tile_size: int = 0
    overlap: int = 0
    threshold: float = 0.5


def encode_png(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def post_predict(
    url: str, image_bytes: bytes, prompt: str, timeout: int, *, probability: bool
) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(
            {
                "task": "referring_seg",
                "image": base64.b64encode(image_bytes).decode("ascii"),
                "text": prompt,
                "return_probability": probability,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"detail": body}


def decode_gray_png(encoded: str, size: tuple[int, int], label: str) -> np.ndarray:
    if not encoded:
        raise ValueError(f"RemoteSAM returned no {label}")
    payload = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(payload)) as image:
        if image.format != "PNG" or image.mode != "L" or image.size != size:
            raise ValueError(
                f"Invalid {label}: format={image.format}, mode={image.mode}, size={image.size}"
            )
        return np.asarray(image).copy()


def validate_response(response: dict, expected_size: tuple[int, int]) -> None:
    if response.get("status") != "success":
        raise RuntimeError(
            f"RemoteSAM status={response.get('status')}: {response.get('message', '')}"
        )
    model = response.get("model") or {}
    if (
        model.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or model.get("service_version") != SERVICE_VERSION
    ):
        raise ValueError("Response model version differs from frozen configuration")
    if response.get("original_size") != list(expected_size):
        raise ValueError("Response input size differs from the submitted image")
    if response.get("output_size") != list(expected_size):
        raise ValueError("Response mask size differs from the submitted image")


def positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    result = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if result[-1] != last:
        result.append(last)
    return result


def blend_weight(tile_size: int) -> np.ndarray:
    axis = np.hanning(tile_size).astype(np.float32)
    weight = np.outer(axis, axis)
    # A positive floor keeps image-border pixels valid while still suppressing seams.
    return np.maximum(weight, 0.05)


def padded_tile(image: np.ndarray, x: int, y: int, tile_size: int) -> tuple[Image.Image, int, int]:
    height, width = image.shape[:2]
    crop = image[y : min(y + tile_size, height), x : min(x + tile_size, width)]
    crop_height, crop_width = crop.shape[:2]
    if crop_height != tile_size or crop_width != tile_size:
        crop = np.pad(
            crop,
            ((0, tile_size - crop_height), (0, tile_size - crop_width), (0, 0)),
            mode="edge",
        )
    return Image.fromarray(crop, mode="RGB"), crop_width, crop_height


def infer_direct(
    image: Image.Image,
    experiment: Experiment,
    url: str,
    timeout: int,
) -> tuple[dict[float, np.ndarray], dict]:
    original_size = image.size
    submitted = image
    if experiment.scale != 1.0:
        submitted = image.resize(
            (
                max(1, round(original_size[0] * experiment.scale)),
                max(1, round(original_size[1] * experiment.scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    started = time.perf_counter()
    status, response = post_predict(
        url, encode_png(submitted), experiment.prompt, timeout, probability=False
    )
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {response.get('detail', 'request rejected')}")
    validate_response(response, submitted.size)
    mask_pixels = decode_gray_png(response.get("mask", ""), submitted.size, "mask")
    if not np.isin(mask_pixels, (0, 255)).all():
        raise ValueError("Prediction mask is not binary 0/255")
    if submitted.size != original_size:
        mask_pixels = np.asarray(
            Image.fromarray(mask_pixels, mode="L").resize(
                original_size, Image.Resampling.NEAREST
            )
        ).copy()
    timing = response.get("timing_ms") or {}
    return {experiment.threshold: mask_pixels > 0}, {
        "request_count": 1,
        "client_wall_ms": round((time.perf_counter() - started) * 1000, 2),
        "service_inference_ms": float(timing.get("inference", 0.0)),
        "service_total_ms": float(timing.get("total", 0.0)),
        "peak_vram_mb": response.get("peak_vram_mb", ""),
        "responses": [
            {
                "http_status": status,
                "original_size": response.get("original_size"),
                "output_size": response.get("output_size"),
                "timing_ms": timing,
                "quality": response.get("quality"),
                "model": response.get("model"),
            }
        ],
    }


def infer_tiled(
    image: Image.Image,
    experiment: Experiment,
    thresholds: list[float],
    url: str,
    timeout: int,
) -> tuple[dict[float, np.ndarray], dict]:
    image_array = np.asarray(image)
    height, width = image_array.shape[:2]
    xs = positions(width, experiment.tile_size, experiment.overlap)
    ys = positions(height, experiment.tile_size, experiment.overlap)
    weight = blend_weight(experiment.tile_size)
    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    responses: list[dict] = []
    inference_ms = 0.0
    service_total_ms = 0.0
    peak_vram = 0.0
    started = time.perf_counter()
    for y in ys:
        for x in xs:
            tile, crop_width, crop_height = padded_tile(
                image_array, x, y, experiment.tile_size
            )
            status, response = post_predict(
                url, encode_png(tile), experiment.prompt, timeout, probability=True
            )
            if status != 200:
                raise RuntimeError(
                    f"Tile ({x},{y}) HTTP {status}: {response.get('detail', 'request rejected')}"
                )
            validate_response(response, tile.size)
            probability = decode_gray_png(
                response.get("probability_map", ""), tile.size, "probability map"
            ).astype(np.float32) / 255.0
            tile_weight = weight[:crop_height, :crop_width]
            probability_sum[y : y + crop_height, x : x + crop_width] += (
                probability[:crop_height, :crop_width] * tile_weight
            )
            weight_sum[y : y + crop_height, x : x + crop_width] += tile_weight
            timing = response.get("timing_ms") or {}
            inference_ms += float(timing.get("inference", 0.0))
            service_total_ms += float(timing.get("total", 0.0))
            peak_vram = max(peak_vram, float(response.get("peak_vram_mb") or 0.0))
            responses.append(
                {
                    "x": x,
                    "y": y,
                    "valid_width": crop_width,
                    "valid_height": crop_height,
                    "timing_ms": timing,
                    "quality": response.get("quality"),
                }
            )
    if np.any(weight_sum <= 0):
        raise ValueError("Tiled blending left uncovered pixels")
    blended = probability_sum / weight_sum
    return {threshold: blended >= threshold for threshold in thresholds}, {
        "request_count": len(responses),
        "client_wall_ms": round((time.perf_counter() - started) * 1000, 2),
        "service_inference_ms": round(inference_ms, 2),
        "service_total_ms": round(service_total_ms, 2),
        "peak_vram_mb": peak_vram,
        "responses": responses,
        "probability": blended,
    }


def blank_result(source: dict[str, str], experiment: Experiment) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "experiment_id": experiment.experiment_id,
        "strategy": experiment.strategy,
        "prompt": experiment.prompt,
        "scale": experiment.scale,
        "tile_size": experiment.tile_size or "",
        "overlap": experiment.overlap or "",
        "probability_threshold": experiment.threshold,
        "image_id": source["image_id"],
        "dataset": source["dataset"],
        "width": source["width"],
        "height": source["height"],
        "image_sha256": source["image_sha256"],
        "ground_truth_sha256": source["ground_truth_sha256"],
        "status": "failure",
        "failure_code": "",
        "failure_detail": "",
        "request_count": 0,
        "output_width": "",
        "output_height": "",
        "binary_valid": False,
        "empty_mask": "",
        "prediction_foreground_pixels": "",
        "ground_truth_foreground_pixels": source["ground_truth_foreground_pixels"],
        "tp": "",
        "fp": "",
        "fn": "",
        "iou": 0.0,
        "dice": 0.0,
        "client_wall_ms": "",
        "service_inference_ms": "",
        "service_total_ms": "",
        "peak_vram_mb": "",
        "service_version": SERVICE_VERSION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
    }


def score(
    source: dict[str, str],
    experiment: Experiment,
    prediction: np.ndarray,
    truth: np.ndarray,
    timing: dict,
) -> dict[str, object]:
    result = blank_result(source, experiment)
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    denominator = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn
    result.update(
        {
            "status": "success",
            "request_count": timing["request_count"],
            "output_width": int(source["width"]),
            "output_height": int(source["height"]),
            "binary_valid": True,
            "empty_mask": not bool(np.any(prediction)),
            "prediction_foreground_pixels": int(np.count_nonzero(prediction)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "iou": tp / denominator if denominator else 1.0,
            "dice": 2 * tp / dice_denominator if dice_denominator else 1.0,
            "client_wall_ms": timing["client_wall_ms"],
            "service_inference_ms": timing["service_inference_ms"],
            "service_total_ms": timing["service_total_ms"],
            "peak_vram_mb": timing["peak_vram_mb"],
        }
    )
    return result


def summarize(rows: list[dict[str, object]], manifest_sha256: str, run_meta: dict) -> dict:
    def is_true(value: object) -> bool:
        return value is True or str(value).lower() == "true"

    experiments: dict[str, dict] = {}
    for experiment_id in sorted({str(row["experiment_id"]) for row in rows}):
        items = [row for row in rows if row["experiment_id"] == experiment_id]
        successes = [row for row in items if row["status"] == "success"]
        experiments[experiment_id] = {
            "configuration": {
                key: items[0][key]
                for key in (
                    "strategy",
                    "prompt",
                    "scale",
                    "tile_size",
                    "overlap",
                    "probability_threshold",
                )
            },
            "images": len(items),
            "successes": len(successes),
            "failures": len(items) - len(successes),
            "failure_rate": (len(items) - len(successes)) / len(items),
            "empty_masks": sum(is_true(row["empty_mask"]) for row in successes),
            "mean_iou_failure_as_zero": statistics.mean(float(row["iou"]) for row in items),
            "mean_dice_failure_as_zero": statistics.mean(float(row["dice"]) for row in items),
            "mean_service_inference_ms_per_image": statistics.mean(
                float(row["service_inference_ms"] or 0.0) for row in items
            ),
            "median_service_inference_ms_per_image": statistics.median(
                float(row["service_inference_ms"] or 0.0) for row in items
            ),
            "mean_client_wall_ms_per_image": statistics.mean(
                float(row["client_wall_ms"] or 0.0) for row in items
            ),
            "total_requests": sum(int(row["request_count"] or 0) for row in items),
        }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_manifest_sha256": manifest_sha256,
        "service_version": SERVICE_VERSION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "run": run_meta,
        "experiments": experiments,
    }


def experiments_for_args(args: argparse.Namespace) -> tuple[list[Experiment], list[float]]:
    if args.mode == "prompt-sweep":
        return [
            Experiment(
                "prompt_" + prompt.replace(" ", "_"), "direct", prompt
            )
            for prompt in PROMPTS
        ], [0.5]
    if args.mode == "upscale":
        return [
            Experiment(
                f"upscale_{args.scale:g}_{args.prompt.replace(' ', '_')}",
                "upscale",
                args.prompt,
                scale=args.scale,
            )
        ], [0.5]
    thresholds = sorted(set(args.thresholds))
    return [
        Experiment(
            f"tile_{args.tile_size}_overlap_{args.overlap}_p{str(threshold).replace('.', '_')}",
            "tile",
            args.prompt,
            tile_size=args.tile_size,
            overlap=args.overlap,
            threshold=threshold,
        )
        for threshold in thresholds
    ], thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--mode", choices=("prompt-sweep", "upscale", "tile"), required=True)
    parser.add_argument("--prompt", default="all airplanes")
    parser.add_argument("--scale", type=float, default=1.25)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--results-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    if args.scale <= 1.0 and args.mode == "upscale":
        raise ValueError("Upscale mode requires --scale greater than 1")
    if args.mode == "tile" and not 0 <= args.overlap < args.tile_size:
        raise ValueError("Tile overlap must be in [0, tile_size)")
    if any(not 0 < threshold < 1 for threshold in args.thresholds):
        raise ValueError("Probability thresholds must be between 0 and 1")
    if args.results_csv.exists() or args.summary_json.exists():
        raise FileExistsError("Refusing to overwrite an existing result or summary")
    if args.artifacts_dir.exists() and any(args.artifacts_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty artifacts directory")

    verify_dataset_manifest(args.manifest, args.data_root, MANIFEST_SHA256)
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        source_rows = [row for row in csv.DictReader(stream) if row["class_name"] == "plane"]
    if len(source_rows) != 10:
        raise ValueError(f"Expected the frozen 10 plane images, got {len(source_rows)}")

    base_url = args.base_url.rstrip("/")
    service_meta = preflight(base_url, args.timeout)
    experiments, thresholds = experiments_for_args(args)
    run_meta = {
        "mode": args.mode,
        "manifest_sha256": file_sha256(args.manifest),
        "plane_ids": [row["image_id"] for row in source_rows],
        "service": service_meta,
        "note": "Ground truth is used only after inference for scoring.",
    }
    for directory in ("predictions", "responses", "probabilities", "by_experiment"):
        (args.artifacts_dir / directory).mkdir(parents=True, exist_ok=True)
    (args.artifacts_dir / "run_metadata.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    results: list[dict[str, object]] = []
    tile_template = experiments[0] if args.mode == "tile" else None
    for source in source_rows:
        image_path = args.data_root / source["image_path"]
        truth_path = args.data_root / source["ground_truth_path"]
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        with Image.open(truth_path) as opened:
            truth = np.asarray(opened) > 0
        expected_size = int(source["width"]), int(source["height"])
        if image.size != expected_size or truth.shape != expected_size[::-1]:
            raise ValueError(f"Frozen size mismatch: {source['image_id']}")

        if tile_template:
            try:
                predictions, timing = infer_tiled(
                    image, tile_template, thresholds, base_url + "/predict", args.timeout
                )
                probability = timing.pop("probability")
                Image.fromarray(np.rint(probability * 255).astype(np.uint8), mode="L").save(
                    args.artifacts_dir / "probabilities" / f"{source['image_id']}.png"
                )
                for experiment in experiments:
                    prediction = predictions[experiment.threshold]
                    result = score(source, experiment, prediction, truth, timing)
                    Image.fromarray(prediction.astype(np.uint8) * 255, mode="L").save(
                        args.artifacts_dir
                        / "predictions"
                        / f"{experiment.experiment_id}__{source['image_id']}.png"
                    )
                    results.append(result)
                    print(
                        f"{experiment.experiment_id} {source['image_id']}: "
                        f"IoU={result['iou']:.4f} Dice={result['dice']:.4f}",
                        flush=True,
                    )
                (args.artifacts_dir / "responses" / f"{source['image_id']}.json").write_text(
                    json.dumps(timing["responses"], ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as exc:
                for experiment in experiments:
                    result = blank_result(source, experiment)
                    result["failure_code"] = type(exc).__name__
                    result["failure_detail"] = str(exc).replace("\r", " ").replace("\n", " ")[:240]
                    results.append(result)
                    print(
                        f"{experiment.experiment_id} {source['image_id']}: failure {exc}",
                        flush=True,
                    )
        else:
            for experiment in experiments:
                try:
                    predictions, timing = infer_direct(
                        image, experiment, base_url + "/predict", args.timeout
                    )
                    prediction = predictions[experiment.threshold]
                    result = score(source, experiment, prediction, truth, timing)
                    Image.fromarray(prediction.astype(np.uint8) * 255, mode="L").save(
                        args.artifacts_dir
                        / "predictions"
                        / f"{experiment.experiment_id}__{source['image_id']}.png"
                    )
                    (args.artifacts_dir / "responses" / f"{experiment.experiment_id}__{source['image_id']}.json").write_text(
                        json.dumps(timing["responses"], ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                except Exception as exc:
                    result = blank_result(source, experiment)
                    result["failure_code"] = type(exc).__name__
                    result["failure_detail"] = str(exc).replace("\r", " ").replace("\n", " ")[:240]
                results.append(result)
                print(
                    f"{experiment.experiment_id} {source['image_id']}: {result['status']} "
                    f"IoU={float(result['iou']):.4f} Dice={float(result['dice']):.4f}",
                    flush=True,
                )

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.results_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    for experiment in experiments:
        experiment_rows = [row for row in results if row["experiment_id"] == experiment.experiment_id]
        with (args.artifacts_dir / "by_experiment" / f"{experiment.experiment_id}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(experiment_rows)
    summary = summarize(results, run_meta["manifest_sha256"], run_meta)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["experiments"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
