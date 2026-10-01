"""Run the GeoScope v0.7 fast/accurate RemoteSAM comparison.

The script keeps raw masks outside Git, writes compact per-image CSV/JSON
records, independently recomputes metrics and candidate-component statistics,
and never stores the service URL or base64 payloads in committed results.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


CSV_FIELDS = [
    "image_id",
    "quality_mode",
    "status",
    "error_type",
    "prompt",
    "width",
    "height",
    "candidate_min_area_pixels",
    "raw_candidate_count",
    "candidate_count",
    "filtered_out_count",
    "prediction_area_pixels",
    "ground_truth_area_pixels",
    "tp",
    "fp",
    "fn",
    "iou",
    "dice",
    "service_inference_ms",
    "service_total_ms",
    "client_round_trip_ms",
    "tile_count",
    "tile_size",
    "overlap",
    "probability_threshold",
    "service_version",
    "checkpoint_sha256",
    "mask_sha256",
    "candidate_stats_verified",
    "artifact_key",
]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any], float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return (
                response.status,
                json.load(response),
                round((time.perf_counter() - started) * 1000, 2),
            )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            result = json.loads(body)
        except json.JSONDecodeError:
            result = {"detail": body}
        return exc.code, result, round((time.perf_counter() - started) * 1000, 2)


def decode_base64_png(value: str, expected_size: tuple[int, int]) -> tuple[np.ndarray, bytes]:
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    payload = base64.b64decode(value, validate=True)
    with Image.open(io.BytesIO(payload)) as image:
        gray = image.convert("L")
        if gray.size != expected_size:
            raise ValueError(f"mask size {gray.size} != image size {expected_size}")
        pixels = np.asarray(gray, dtype=np.uint8)
    values = set(np.unique(pixels).tolist())
    if not values.issubset({0, 255}):
        raise ValueError(f"mask is not binary 0/255: {sorted(values)[:20]}")
    return pixels > 0, payload


def score(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    union = tp + fp + fn
    denominator = 2 * tp + fp + fn
    return {
        "prediction_area_pixels": int(np.count_nonzero(prediction)),
        "ground_truth_area_pixels": int(np.count_nonzero(truth)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": float(tp / union) if union else 1.0,
        "dice": float(2 * tp / denominator) if denominator else 1.0,
    }


def candidate_summary(mask: np.ndarray, min_area_pixels: int) -> dict[str, Any]:
    binary = mask.astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates = []
    for label in range(1, int(count)):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_pixels:
            continue
        candidates.append(
            {
                "candidate_id": len(candidates) + 1,
                "area_pixels": area,
                "bbox": {
                    "x": int(stats[label, cv2.CC_STAT_LEFT]),
                    "y": int(stats[label, cv2.CC_STAT_TOP]),
                    "width": int(stats[label, cv2.CC_STAT_WIDTH]),
                    "height": int(stats[label, cv2.CC_STAT_HEIGHT]),
                },
                "center": {
                    "x": round(float(centroids[label, 0]), 3),
                    "y": round(float(centroids[label, 1]), 3),
                },
            }
        )
    areas = [item["area_pixels"] for item in candidates]
    raw_count = max(0, int(count) - 1)
    return {
        "raw_candidate_count": raw_count,
        "candidate_count": len(candidates),
        "filtered_out_count": raw_count - len(candidates),
        "total_area_pixels": int(sum(areas)),
        "min_area": min(areas) if areas else 0,
        "max_area": max(areas) if areas else 0,
        "mean_area": round(float(np.mean(areas)), 3) if areas else 0.0,
        "candidates": candidates,
    }


def verify_candidate_stats(service: dict[str, Any], local: dict[str, Any]) -> None:
    for key in (
        "raw_candidate_count",
        "candidate_count",
        "filtered_out_count",
        "total_area_pixels",
        "min_area",
        "max_area",
        "mean_area",
        "candidates",
    ):
        if service.get(key) != local.get(key):
            raise ValueError(f"candidate_stats mismatch for {key}")


def compact_response(response: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in response.items()
        if key not in {"mask", "result_image", "probability_map", "masks"}
    }


def write_overlay(image_path: Path, mask: np.ndarray, output_path: Path) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    array = np.asarray(image, dtype=np.uint8).copy()
    foreground = mask.astype(bool)
    array[foreground] = np.rint(
        array[foreground].astype(np.float32) * 0.5
        + np.array([255, 0, 0], dtype=np.float32) * 0.5
    ).astype(np.uint8)
    Image.fromarray(array).save(output_path)


def load_aircraft_rows(manifest: Path, limit: int) -> list[dict[str, str]]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row.get("project_target") == "aircraft"
        ]
    if len(rows) < limit:
        raise ValueError(f"manifest contains only {len(rows)} aircraft rows; requested {limit}")
    return rows[:limit]


def blank_row(source: dict[str, str], mode: str, prompt: str, min_area: int) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        image_id=source["image_id"],
        quality_mode=mode,
        status="error",
        prompt=prompt,
        width=int(source["width"]),
        height=int(source["height"]),
        candidate_min_area_pixels=min_area,
        ground_truth_area_pixels=int(source["ground_truth_foreground_pixels"]),
        artifact_key=f'{source["image_id"]}/{mode}',
        candidate_stats_verified=False,
    )
    return row


def finite_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row["status"] == "success" and row[key] != ""]
    return float(np.mean(values)) if values else None


def summarize(
    rows: list[dict[str, Any]],
    manifest_sha256: str,
    selected_ids: list[str],
    prompt: str,
    min_area: int,
    error_probes: list[dict[str, Any]],
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode in ("fast", "accurate"):
        selected = [row for row in rows if row["quality_mode"] == mode]
        successes = [row for row in selected if row["status"] == "success"]
        modes[mode] = {
            "images": len(selected),
            "successes": len(successes),
            "failures": len(selected) - len(successes),
            "mean_iou": finite_mean(selected, "iou"),
            "mean_dice": finite_mean(selected, "dice"),
            "mean_candidate_count": finite_mean(selected, "candidate_count"),
            "mean_prediction_area_pixels": finite_mean(selected, "prediction_area_pixels"),
            "mean_service_inference_ms": finite_mean(selected, "service_inference_ms"),
            "mean_service_total_ms": finite_mean(selected, "service_total_ms"),
            "mean_client_round_trip_ms": finite_mean(selected, "client_round_trip_ms"),
        }

    by_image = {image_id: {} for image_id in selected_ids}
    for row in rows:
        by_image[row["image_id"]][row["quality_mode"]] = row
    comparisons = []
    for image_id, pair in by_image.items():
        if pair.get("fast", {}).get("status") != "success" or pair.get("accurate", {}).get("status") != "success":
            continue
        comparisons.append(
            {
                "image_id": image_id,
                "fast_iou": pair["fast"]["iou"],
                "accurate_iou": pair["accurate"]["iou"],
                "iou_delta": pair["accurate"]["iou"] - pair["fast"]["iou"],
                "fast_candidate_count": pair["fast"]["candidate_count"],
                "accurate_candidate_count": pair["accurate"]["candidate_count"],
                "fast_client_round_trip_ms": pair["fast"]["client_round_trip_ms"],
                "accurate_client_round_trip_ms": pair["accurate"]["client_round_trip_ms"],
            }
        )

    accurate_successes = sorted(
        [row for row in rows if row["quality_mode"] == "accurate" and row["status"] == "success"],
        key=lambda row: float(row["iou"]),
    )
    cases: dict[str, Any] = {}
    if accurate_successes:
        case_rows = {
            "worst": accurate_successes[0],
            "middle": accurate_successes[len(accurate_successes) // 2],
            "best": accurate_successes[-1],
        }
        cases = {
            label: {
                "image_id": row["image_id"],
                "iou": row["iou"],
                "dice": row["dice"],
                "candidate_count": row["candidate_count"],
                "artifact_key": row["artifact_key"],
            }
            for label, row in case_rows.items()
        }

    first_success = next((row for row in rows if row["status"] == "success"), {})
    return {
        "schema_version": "geoscope-v0.7-remotesam-modes-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest_sha256,
        "selected_image_ids": selected_ids,
        "prompt": prompt,
        "candidate_min_area_pixels": min_area,
        "service": {
            "service_version": first_success.get("service_version"),
            "checkpoint_sha256": first_success.get("checkpoint_sha256"),
        },
        "modes": modes,
        "per_image_comparison": comparisons,
        "mean_iou_delta_accurate_minus_fast": (
            float(np.mean([item["iou_delta"] for item in comparisons])) if comparisons else None
        ),
        "cases": cases,
        "stability": {
            "serial_requests": len(rows),
            "successes": sum(row["status"] == "success" for row in rows),
            "failures": sum(row["status"] != "success" for row in rows),
            "all_candidate_stats_independently_verified": all(
                row["candidate_stats_verified"] is True
                for row in rows
                if row["status"] == "success"
            ),
        },
        "error_probes": error_probes,
        "limitations": [
            "Connected mask regions are candidate targets, not guaranteed object instances.",
            "The accurate threshold was selected on the fixed diagnostic aircraft set and is not an independent generalization estimate.",
            "All areas are pixel areas because the PNG inputs have no georeferencing.",
        ],
    }


def run_error_probes(url: str, timeout: float) -> list[dict[str, Any]]:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (0, 0, 0)).save(buffer, format="PNG")
    valid_image = base64.b64encode(buffer.getvalue()).decode("ascii")
    probes = [
        (
            "invalid_base64",
            {"task": "referring_seg", "image": "not-base64", "text": "all planes"},
        ),
        (
            "invalid_quality_mode",
            {
                "task": "referring_seg",
                "image": valid_image,
                "text": "all planes",
                "quality_mode": "turbo",
            },
        ),
        (
            "unsupported_mask_size",
            {
                "task": "referring_seg",
                "image": valid_image,
                "text": "all planes",
                "mask_size": 256,
            },
        ),
    ]
    results = []
    for name, payload in probes:
        http_status, response, round_trip = post_json(url, payload, timeout)
        service_status = response.get("status")
        passed = (
            "mask" not in response
            and (
                (name == "unsupported_mask_size" and http_status == 400)
                or (name != "unsupported_mask_size" and service_status == "error")
            )
        )
        results.append(
            {
                "name": name,
                "passed": passed,
                "http_status": http_status,
                "service_status": service_status,
                "message": response.get("message") or response.get("detail"),
                "client_round_trip_ms": round_trip,
                "returned_mask": "mask" in response,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/dataset_manifest.csv"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:6657/predict")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--prompt", default="all planes")
    parser.add_argument("--candidate-min-area-pixels", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    if args.limit < 1 or args.limit > 10:
        raise ValueError("--limit must be between 1 and 10")
    if args.candidate_min_area_pixels < 1:
        raise ValueError("--candidate-min-area-pixels must be at least 1")

    sources = load_aircraft_rows(args.manifest, args.limit)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for source in sources:
        image_path = args.data_root / source["image_path"]
        truth_path = args.data_root / source["ground_truth_path"]
        image_bytes = image_path.read_bytes()
        with Image.open(image_path) as image:
            image_size = image.size
        with Image.open(truth_path) as truth_image:
            truth = np.asarray(truth_image.convert("L"), dtype=np.uint8) > 0
        if image_size != (int(source["width"]), int(source["height"])):
            raise ValueError(f"manifest size mismatch for {source['image_id']}")

        for mode in ("fast", "accurate"):
            row = blank_row(
                source, mode, args.prompt, args.candidate_min_area_pixels
            )
            artifact_path = args.artifact_dir / source["image_id"] / mode
            artifact_path.mkdir(parents=True, exist_ok=True)
            try:
                http_status, response, client_ms = post_json(
                    args.url,
                    {
                        "task": "referring_seg",
                        "image": base64.b64encode(image_bytes).decode("ascii"),
                        "text": args.prompt,
                        "quality_mode": mode,
                        "candidate_min_area_pixels": args.candidate_min_area_pixels,
                    },
                    args.timeout,
                )
                if http_status != 200 or response.get("status") != "success":
                    raise RuntimeError(
                        f"HTTP {http_status}: {response.get('message') or response.get('detail')}"
                    )
                if response.get("quality_mode") != mode:
                    raise ValueError(
                        f"service resolved {response.get('quality_mode')!r}, expected {mode!r}"
                    )
                prediction, mask_bytes = decode_base64_png(response["mask"], image_size)
                metrics = score(prediction, truth)
                local_candidates = candidate_summary(
                    prediction, args.candidate_min_area_pixels
                )
                service_candidates = response.get("candidate_stats") or {}
                verify_candidate_stats(service_candidates, local_candidates)
                mask_path = artifact_path / "mask.png"
                mask_path.write_bytes(mask_bytes)
                write_overlay(image_path, prediction, artifact_path / "overlay.png")
                (artifact_path / "response.json").write_text(
                    json.dumps(compact_response(response), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                timing = response.get("timing_ms") or {}
                mode_params = response.get("mode_parameters") or {}
                model = response.get("model") or {}
                row.update(
                    status="success",
                    error_type="",
                    raw_candidate_count=service_candidates["raw_candidate_count"],
                    candidate_count=service_candidates["candidate_count"],
                    filtered_out_count=service_candidates["filtered_out_count"],
                    service_inference_ms=timing.get("inference", ""),
                    service_total_ms=timing.get("total", ""),
                    client_round_trip_ms=client_ms,
                    tile_count=mode_params.get("tile_count", ""),
                    tile_size=(
                        mode_params.get("tile_size")
                        if mode_params.get("tile_size") is not None
                        else ""
                    ),
                    overlap=mode_params.get("overlap", ""),
                    probability_threshold=(
                        mode_params.get("probability_threshold")
                        if mode_params.get("probability_threshold") is not None
                        else ""
                    ),
                    service_version=model.get("service_version", ""),
                    checkpoint_sha256=model.get("checkpoint_sha256", ""),
                    mask_sha256=sha256_bytes(mask_bytes),
                    candidate_stats_verified=True,
                    **metrics,
                )
            except Exception as exc:
                row["error_type"] = type(exc).__name__
                (artifact_path / "error.json").write_text(
                    json.dumps(
                        {"error_type": type(exc).__name__, "message": str(exc)},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            rows.append(row)
            print(
                json.dumps(
                    {
                        "image_id": row["image_id"],
                        "mode": mode,
                        "status": row["status"],
                        "iou": row["iou"],
                        "candidates": row["candidate_count"],
                        "client_ms": row["client_round_trip_ms"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    error_probes = run_error_probes(args.url, args.timeout)
    with args.results_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(
        rows,
        sha256_file(args.manifest),
        [row["image_id"] for row in sources],
        args.prompt,
        args.candidate_min_area_pixels,
        error_probes,
    )
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if any(row["status"] != "success" for row in rows):
        return 1
    if not all(item["passed"] for item in error_probes):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
