"""Run the frozen RemoteSAM evaluation with per-image evidence and metrics."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from verify_dataset_manifest import verify as verify_dataset_manifest


MANIFEST_SHA256 = "5b5339814a2c83c4b87ec026fa46fe8ff67cf6b534554c24ac8d21e421456241"
CHECKPOINT_SHA256 = "f85dfa044a527f096b9e41eacfe298040d52e77fca642f46dfe1226745e137e7"
SERVICE_VERSION = "1.1.0"
SMOKE_IDS = (
    "WHU_SAT1_18_0",
    "WHU_SAT1_25_1",
    "WHU_SAT1_6_2",
    "ISAID_VAL_P0257",
    "ISAID_VAL_P1492",
)
REPEAT_IDS = (
    "WHU_SAT1_18_0",
    "ISAID_VAL_P0257",
    "ISAID_VAL_P1143",
)
FIELDS = (
    "protocol_version",
    "image_id",
    "dataset",
    "class_name",
    "prompt",
    "width",
    "height",
    "image_sha256",
    "ground_truth_sha256",
    "status",
    "failure_code",
    "failure_detail",
    "http_status",
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_json(url: str, timeout: int) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"Unexpected preflight HTTP status: {response.status}")
        return json.load(response)


def preflight(base_url: str, timeout: int) -> dict:
    ready = get_json(base_url + "/ready", timeout)
    model_info = get_json(base_url + "/model-info", timeout)
    expected = {
        "status": "ready",
        "service_version": SERVICE_VERSION,
        "device": "cuda:0",
        "fp16": True,
        "epoc": False,
        "checkpoint_sha256": CHECKPOINT_SHA256,
    }
    for key, value in expected.items():
        if ready.get(key) != value:
            raise ValueError(f"RemoteSAM preflight mismatch in {key}: {ready.get(key)!r}")
    if model_info.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("Model-info checkpoint SHA-256 differs from /ready")
    if not model_info.get("ready"):
        raise ValueError("Model-info does not report ready")
    return {
        "ready": {key: ready.get(key) for key in expected},
        "model_loaded_at": model_info.get("loaded_at"),
        "model_load_time_ms": model_info.get("load_time_ms"),
    }


def redact(message: str) -> str:
    message = re.sub(r"[A-Za-z]:\\[^\s,;]+", "<local-path>", message)
    message = re.sub(
        r"\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b",
        "<private-ip>",
        message,
    )
    return message.replace("\r", " ").replace("\n", " ")[:240]


def post_predict(url: str, image_bytes: bytes, prompt: str, timeout: int) -> tuple[int, dict]:
    payload = {
        "task": "referring_seg",
        "image": base64.b64encode(image_bytes).decode("ascii"),
        "text": prompt,
        "return_probability": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"detail": body}
        return exc.code, parsed


def decode_prediction(encoded: str, expected_size: tuple[int, int]) -> tuple[bytes, np.ndarray]:
    if not encoded:
        raise ValueError("RemoteSAM returned no mask")
    mask_bytes = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(mask_bytes)) as image:
        if image.format != "PNG" or image.mode != "L":
            raise ValueError(f"Expected grayscale PNG mask, got {image.format}/{image.mode}")
        if image.size != expected_size:
            raise ValueError(f"Mask size {image.size} differs from {expected_size}")
        pixels = np.asarray(image)
    if not np.isin(pixels, (0, 255)).all():
        raise ValueError("Prediction mask is not binary 0/255")
    return mask_bytes, pixels > 0


def blank_result(source: dict[str, str]) -> dict[str, object]:
    return {
        "protocol_version": source["protocol_version"],
        "image_id": source["image_id"],
        "dataset": source["dataset"],
        "class_name": source["class_name"],
        "prompt": source["prompt"],
        "width": source["width"],
        "height": source["height"],
        "image_sha256": source["image_sha256"],
        "ground_truth_sha256": source["ground_truth_sha256"],
        "status": "failure",
        "failure_code": "",
        "failure_detail": "",
        "http_status": "",
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


def evaluate_one(
    source: dict[str, str], data_root: Path, base_url: str, artifacts_dir: Path, timeout: int
) -> dict[str, object]:
    result = blank_result(source)
    image_path = data_root / source["image_path"]
    mask_path = data_root / source["ground_truth_path"]
    expected_size = int(source["width"]), int(source["height"])
    image_bytes = image_path.read_bytes()
    with Image.open(mask_path) as image:
        ground_truth = np.asarray(image) > 0
    started = time.perf_counter()
    response: dict = {}
    try:
        http_status, response = post_predict(
            base_url + "/predict", image_bytes, source["prompt"], timeout
        )
        result["http_status"] = http_status
        if http_status != 200:
            raise RuntimeError(f"HTTP {http_status}: {response.get('detail', 'request rejected')}")
        if response.get("status") != "success":
            raise RuntimeError(f"RemoteSAM status={response.get('status')}: {response.get('message', '')}")
        model = response.get("model") or {}
        if model.get("checkpoint_sha256") != CHECKPOINT_SHA256 or model.get("service_version") != SERVICE_VERSION:
            raise ValueError("Response model version differs from frozen preflight")
        if response.get("original_size") != list(expected_size) or response.get("output_size") != list(expected_size):
            raise ValueError("Response original/output size differs from source image")
        mask_bytes, prediction = decode_prediction(response.get("mask", ""), expected_size)
        result["output_width"], result["output_height"] = expected_size
        result["binary_valid"] = True
        result["empty_mask"] = not bool(np.any(prediction))
        result["prediction_foreground_pixels"] = int(np.count_nonzero(prediction))
        tp = int(np.count_nonzero(prediction & ground_truth))
        fp = int(np.count_nonzero(prediction & ~ground_truth))
        fn = int(np.count_nonzero(~prediction & ground_truth))
        result["tp"], result["fp"], result["fn"] = tp, fp, fn
        result["iou"] = tp / (tp + fp + fn)
        result["dice"] = 2 * tp / (2 * tp + fp + fn)
        timing = response.get("timing_ms") or {}
        result["service_inference_ms"] = timing.get("inference", "")
        result["service_total_ms"] = timing.get("total", "")
        result["peak_vram_mb"] = response.get("peak_vram_mb", "")
        result["status"] = "success"
        (artifacts_dir / "predictions" / f"{source['image_id']}.png").write_bytes(mask_bytes)
        if response.get("result_image"):
            try:
                (artifacts_dir / "overlays" / f"{source['image_id']}.png").write_bytes(
                    base64.b64decode(response["result_image"], validate=True)
                )
            except (ValueError, OSError):
                # Visualization is evidence, not the segmentation metric itself.
                pass
    except Exception as exc:
        result["failure_code"] = type(exc).__name__
        result["failure_detail"] = redact(str(exc))
    finally:
        result["client_wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
        response_without_images = {
            key: value for key, value in response.items()
            if key not in ("mask", "result_image", "probability_map", "masks")
        }
        (artifacts_dir / "responses" / f"{source['image_id']}.json").write_text(
            json.dumps(response_without_images, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


def summarize(rows: list[dict[str, object]], manifest_sha256: str) -> dict:
    def metrics(items: list[dict[str, object]]) -> dict:
        successes = [row for row in items if row["status"] == "success"]
        timings = [float(row["service_inference_ms"]) for row in successes if row["service_inference_ms"] != ""]
        return {
            "images": len(items),
            "successes": len(successes),
            "failures": len(items) - len(successes),
            "failure_rate": (len(items) - len(successes)) / len(items),
            "empty_masks": sum(row["empty_mask"] is True for row in successes),
            "mean_iou_all_failure_as_zero": statistics.mean(float(row["iou"]) for row in items),
            "mean_dice_all_failure_as_zero": statistics.mean(float(row["dice"]) for row in items),
            "mean_iou_success_only": statistics.mean(float(row["iou"]) for row in successes) if successes else None,
            "mean_dice_success_only": statistics.mean(float(row["dice"]) for row in successes) if successes else None,
            "mean_service_inference_ms": statistics.mean(timings) if timings else None,
            "median_service_inference_ms": statistics.median(timings) if timings else None,
        }

    return {
        "protocol_version": "sam-eval-v1.0",
        "dataset_manifest_sha256": manifest_sha256,
        "service_version": SERVICE_VERSION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "overall": metrics(rows),
        "by_class": {
            name: metrics([row for row in rows if row["class_name"] == name])
            for name in ("building", "plane")
            if any(row["class_name"] == name for row in rows)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--run", choices=("smoke", "full", "repeat"), required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    if args.results_csv.exists() or args.summary_json.exists():
        raise FileExistsError("Refusing to overwrite an existing result or summary")
    if args.artifacts_dir.exists() and any(args.artifacts_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite nonempty artifacts directory")
    verify_dataset_manifest(args.manifest, args.data_root, MANIFEST_SHA256)
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    if args.run in ("smoke", "repeat"):
        by_id = {row["image_id"]: row for row in all_rows}
        sample_ids = SMOKE_IDS if args.run == "smoke" else REPEAT_IDS
        source_rows = [by_id[image_id] for image_id in sample_ids]
    else:
        source_rows = all_rows

    base_url = args.base_url.rstrip("/")
    run_meta = preflight(base_url, args.timeout)
    run_meta.update(
        {
            "protocol_version": "sam-eval-v1.0",
            "run": args.run,
            "dataset_manifest_sha256": file_sha256(args.manifest),
            "sample_ids": [row["image_id"] for row in source_rows],
            "input_policy": "RGB PNG, original geometry, server cap 40 MiB",
        }
    )
    for subdir in ("predictions", "overlays", "responses"):
        (args.artifacts_dir / subdir).mkdir(parents=True)
    (args.artifacts_dir / "run_metadata.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    with args.results_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for source in source_rows:
            result = evaluate_one(source, args.data_root, base_url, args.artifacts_dir, args.timeout)
            writer.writerow(result)
            stream.flush()
            results.append(result)
            print(
                f"{result['image_id']}: {result['status']} "
                f"IoU={float(result['iou']):.4f} Dice={float(result['dice']):.4f} "
                f"{result['failure_code']}",
                flush=True,
            )
    summary = summarize(results, run_meta["dataset_manifest_sha256"])
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
