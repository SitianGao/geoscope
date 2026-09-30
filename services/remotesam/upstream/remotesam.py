"""Production-oriented RemoteSAM inference service for GeoScope.

The service keeps the upstream model isolated behind a small, traceable API.
It never fabricates a mask, never silently resizes a model output, and exposes
enough metadata for GeoScope to audit each inference run.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("remotesam.service")

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = Path(
    os.environ.get("GEO_REMOTESAM_CHECKPOINT", ROOT / "pretrained_weights" / "checkpoint.pth")
).expanduser()
DEVICE = os.environ.get(
    "GEO_REMOTESAM_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"
)
HOST = os.environ.get("GEO_REMOTESAM_HOST", "127.0.0.1")
PORT = int(os.environ.get("GEO_REMOTESAM_PORT", "6657"))
USE_FP16 = os.environ.get("GEO_REMOTESAM_FP16", "1") == "1"
USE_EPOC = os.environ.get("GEO_REMOTESAM_USE_EPOC", "0") == "1"
MAX_IMAGE_BYTES = int(os.environ.get("GEO_REMOTESAM_MAX_IMAGE_MB", "20")) * 1024 * 1024
MAX_IMAGE_PIXELS = int(os.environ.get("GEO_REMOTESAM_MAX_PIXELS", "64000000"))

sam_model = None
INFERENCE_LOCK = threading.Lock()
MODEL_STATE: Dict[str, object] = {
    "ready": False,
    "error": None,
    "checkpoint_sha256": None,
    "loaded_at": None,
    "load_time_ms": None,
    "checkpoint_diagnostics": None,
}


def _env_list(name: str, default: str) -> List[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


app = FastAPI(
    title="RemoteSAM API Server",
    description="Traceable RemoteSAM inference service for GeoScope",
    version="1.2.0",
)

QUALITY_MODES = {"fast", "accurate", "auto"}
ACCURATE_TILE_SIZE = 1024
ACCURATE_TILE_OVERLAP = 256
ACCURATE_PROBABILITY_THRESHOLD = 0.88
ACCURATE_BLEND_MIN_WEIGHT = 0.05
AUTO_ACCURATE_MIN_PIXELS = 1024 * 1024

cors_origins = _env_list("GEO_REMOTESAM_CORS_ORIGINS", "http://127.0.0.1:4180,http://localhost:4180")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task: str
    image: str
    text: str = ""
    classes: List[str] = Field(default_factory=list)
    return_probability: bool = False
    quality_mode: str = "fast"
    candidate_min_area_pixels: int = Field(default=1, ge=1)
    # Retained only to return an explicit compatibility error to older clients.
    mask_size: Optional[int] = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_from_config() -> None:
    """Load the checkpoint once and retain a diagnostic state if loading fails."""
    global sam_model
    started = time.perf_counter()
    MODEL_STATE.update(
        ready=False,
        error=None,
        checkpoint_sha256=None,
        loaded_at=None,
        checkpoint_diagnostics=None,
    )
    try:
        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(
                f"RemoteSAM checkpoint not found: {CHECKPOINT_PATH}. "
                "Set GEO_REMOTESAM_CHECKPOINT to the teacher-provided weight file."
            )
        if DEVICE.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {DEVICE}")

        # Lazy import keeps health diagnostics available even when model-only
        # dependencies have not yet been installed.
        from tasks.code.model import RemoteSAM, init_demo_model

        logger.info("Loading RemoteSAM checkpoint %s on %s", CHECKPOINT_PATH, DEVICE)
        base_model = init_demo_model(str(CHECKPOINT_PATH), DEVICE)
        sam_model = RemoteSAM(base_model, DEVICE, use_EPOC=USE_EPOC)
        MODEL_STATE.update(
            ready=True,
            checkpoint_sha256=sha256_file(CHECKPOINT_PATH),
            checkpoint_diagnostics=getattr(base_model, "checkpoint_diagnostics", None),
            loaded_at=time.time(),
            load_time_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        logger.info("RemoteSAM loaded successfully in %s ms", MODEL_STATE["load_time_ms"])
    except Exception as exc:  # Keep the API alive so /ready explains the failure.
        sam_model = None
        MODEL_STATE.update(
            ready=False,
            error=f"{type(exc).__name__}: {exc}",
            load_time_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        logger.exception("RemoteSAM failed to load")


@app.on_event("startup")
async def startup_load_model() -> None:
    load_model_from_config()


def base64_to_cv2(encoded: str) -> np.ndarray:
    try:
        payload = encoded.split(",", 1)[1] if "," in encoded else encoded
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 image data") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image payload must be between 1 byte and {MAX_IMAGE_BYTES} bytes")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Decoded payload is not a supported image")
    height, width = image.shape[:2]
    if height * width > MAX_IMAGE_PIXELS:
        raise ValueError(f"Image exceeds the configured {MAX_IMAGE_PIXELS}-pixel limit")
    return image


def cv2_to_base64(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode result image")
    return base64.b64encode(buffer).decode("ascii")


def normalize_mask(mask: np.ndarray, image_hw: Tuple[int, int]) -> np.ndarray:
    if mask is None:
        raise ValueError("RemoteSAM returned no mask")
    array = np.asarray(mask).squeeze()
    if array.ndim != 2:
        raise ValueError(f"RemoteSAM returned an invalid mask shape: {array.shape}")
    if tuple(array.shape) != tuple(image_hw):
        raise ValueError(
            f"RemoteSAM mask shape {array.shape} does not match input image {image_hw}; "
            "the service will not silently resize model output"
        )
    return (array > 0).astype(np.uint8)


def normalize_probability(probability: np.ndarray, image_hw: Tuple[int, int]) -> np.ndarray:
    array = np.asarray(probability).squeeze().astype(np.float32)
    if array.ndim != 2 or tuple(array.shape) != tuple(image_hw):
        raise ValueError(f"RemoteSAM probability map shape {array.shape} does not match input image {image_hw}")
    return np.clip(array, 0.0, 1.0)


def mask_to_base64(mask: np.ndarray) -> str:
    return cv2_to_base64((mask > 0).astype(np.uint8) * 255)


def probability_to_base64(probability: np.ndarray) -> str:
    return cv2_to_base64(np.rint(np.clip(probability, 0.0, 1.0) * 255).astype(np.uint8))


def apply_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.5) -> np.ndarray:
    result = image.copy()
    foreground = mask > 0
    if foreground.any():
        overlay = np.zeros_like(image)
        overlay[foreground] = color
        result[foreground] = cv2.addWeighted(
            image[foreground], 1 - alpha, overlay[foreground], alpha, 0
        )
    return result


def quality_summary(mask: np.ndarray, probability: Optional[np.ndarray]) -> Dict[str, object]:
    foreground = mask > 0
    summary: Dict[str, object] = {
        "empty_mask": not bool(foreground.any()),
        "foreground_pixels": int(foreground.sum()),
        "foreground_ratio": float(foreground.mean()),
    }
    if probability is not None:
        summary["mean_foreground_confidence"] = (
            round(float(probability[foreground].mean()), 6) if foreground.any() else None
        )
        summary["mean_image_confidence"] = round(float(probability.mean()), 6)
    return summary


def candidate_summary(mask: np.ndarray, min_area_pixels: int = 1) -> Dict[str, object]:
    """Return deterministic connected-component statistics for a binary mask.

    These are candidate regions from a semantic mask, not guaranteed object
    instances. Eight-connectivity is fixed so the result can be reproduced.
    """
    if min_area_pixels < 1:
        raise ValueError("candidate_min_area_pixels must be at least 1")
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"Candidate statistics require a 2D mask, got {binary.shape}")

    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    raw_candidate_count = max(0, int(component_count) - 1)
    candidates: List[Dict[str, object]] = []
    for label in range(1, int(component_count)):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_pixels:
            continue
        candidates.append(
            {
                "candidate_id": len(candidates) + 1,
                "area_pixels": area,
                "bbox": {"x": x, "y": y, "width": width, "height": height},
                "center": {
                    "x": round(float(centroids[label, 0]), 3),
                    "y": round(float(centroids[label, 1]), 3),
                },
            }
        )

    areas = [int(item["area_pixels"]) for item in candidates]
    return {
        "label": "candidate_regions",
        "connectivity": 8,
        "min_area_pixels": int(min_area_pixels),
        "raw_candidate_count": raw_candidate_count,
        "candidate_count": len(candidates),
        "filtered_out_count": raw_candidate_count - len(candidates),
        "total_area_pixels": int(sum(areas)),
        "min_area": min(areas) if areas else 0,
        "max_area": max(areas) if areas else 0,
        "mean_area": round(float(np.mean(areas)), 3) if areas else 0.0,
        "candidates": candidates,
        "notice": "Connected regions are candidate targets and require human review.",
    }


def _tile_positions(length: int, tile_size: int, overlap: int) -> List[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def _blend_weight(tile_size: int) -> np.ndarray:
    axis = np.hanning(tile_size).astype(np.float32)
    return np.maximum(np.outer(axis, axis), ACCURATE_BLEND_MIN_WEIGHT)


def _padded_tile(
    image: np.ndarray, x: int, y: int, tile_size: int
) -> Tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    crop = image[y : min(y + tile_size, height), x : min(x + tile_size, width)]
    crop_height, crop_width = crop.shape[:2]
    if crop_height == tile_size and crop_width == tile_size:
        return crop, crop_width, crop_height
    padded = np.pad(
        crop,
        ((0, tile_size - crop_height), (0, tile_size - crop_width), (0, 0)),
        mode="edge",
    )
    return padded, crop_width, crop_height


def resolve_quality_mode(requested: str, image_hw: Tuple[int, int]) -> str:
    normalized = requested.strip().lower()
    if normalized not in QUALITY_MODES:
        raise ValueError(
            f"Unsupported quality_mode: {requested!r}; expected fast, accurate, or auto"
        )
    if normalized != "auto":
        return normalized
    height, width = image_hw
    return "accurate" if height * width >= AUTO_ACCURATE_MIN_PIXELS else "fast"


def mode_parameters(requested: str, resolved: str, tile_count: int) -> Dict[str, object]:
    parameters: Dict[str, object] = {
        "requested_mode": requested,
        "resolved_mode": resolved,
        "tile_count": int(tile_count),
    }
    if resolved == "accurate":
        parameters.update(
            strategy="overlapping_tiles_probability_blend",
            tile_size=ACCURATE_TILE_SIZE,
            overlap=ACCURATE_TILE_OVERLAP,
            probability_threshold=ACCURATE_PROBABILITY_THRESHOLD,
            blend="hann_center_weighted",
            blend_min_weight=ACCURATE_BLEND_MIN_WEIGHT,
        )
    else:
        parameters.update(
            strategy="whole_image_argmax",
            tile_size=None,
            overlap=0,
            probability_threshold=None,
            blend=None,
            blend_min_weight=None,
        )
    return parameters


def _accurate_referring_seg(image: np.ndarray, sentence: str) -> Tuple[np.ndarray, np.ndarray, int]:
    height, width = image.shape[:2]
    xs = _tile_positions(width, ACCURATE_TILE_SIZE, ACCURATE_TILE_OVERLAP)
    ys = _tile_positions(height, ACCURATE_TILE_SIZE, ACCURATE_TILE_OVERLAP)
    weight = _blend_weight(ACCURATE_TILE_SIZE)
    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    for y in ys:
        for x in xs:
            tile, crop_width, crop_height = _padded_tile(
                image, x, y, ACCURATE_TILE_SIZE
            )
            _, raw_probability = sam_model.referring_seg(
                image=tile, sentence=sentence, return_prob=True
            )
            probability = normalize_probability(
                raw_probability, (ACCURATE_TILE_SIZE, ACCURATE_TILE_SIZE)
            )
            tile_weight = weight[:crop_height, :crop_width]
            probability_sum[y : y + crop_height, x : x + crop_width] += (
                probability[:crop_height, :crop_width] * tile_weight
            )
            weight_sum[y : y + crop_height, x : x + crop_width] += tile_weight

    if np.any(weight_sum <= 0):
        raise ValueError("Accurate mode failed to cover the complete input image")
    blended = probability_sum / weight_sum
    mask = (blended >= ACCURATE_PROBABILITY_THRESHOLD).astype(np.uint8)
    return mask, blended, len(xs) * len(ys)


def _accurate_semantic_seg(
    image: np.ndarray, classes: List[str]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int]:
    height, width = image.shape[:2]
    xs = _tile_positions(width, ACCURATE_TILE_SIZE, ACCURATE_TILE_OVERLAP)
    ys = _tile_positions(height, ACCURATE_TILE_SIZE, ACCURATE_TILE_OVERLAP)
    weight = _blend_weight(ACCURATE_TILE_SIZE)
    probability_sums = {
        name: np.zeros((height, width), dtype=np.float32) for name in classes
    }
    weight_sum = np.zeros((height, width), dtype=np.float32)

    for y in ys:
        for x in xs:
            tile, crop_width, crop_height = _padded_tile(
                image, x, y, ACCURATE_TILE_SIZE
            )
            _, raw_probabilities = sam_model.semantic_seg(
                image=tile, classnames=classes, return_prob=True
            )
            tile_weight = weight[:crop_height, :crop_width]
            for class_name in classes:
                if class_name not in raw_probabilities:
                    raise ValueError(f"RemoteSAM returned no probability for {class_name!r}")
                probability = normalize_probability(
                    raw_probabilities[class_name],
                    (ACCURATE_TILE_SIZE, ACCURATE_TILE_SIZE),
                )
                probability_sums[class_name][
                    y : y + crop_height, x : x + crop_width
                ] += probability[:crop_height, :crop_width] * tile_weight
            weight_sum[y : y + crop_height, x : x + crop_width] += tile_weight

    if np.any(weight_sum <= 0):
        raise ValueError("Accurate mode failed to cover the complete input image")
    probabilities = {
        name: probability_sums[name] / weight_sum for name in classes
    }
    masks = {
        name: (probability >= ACCURATE_PROBABILITY_THRESHOLD).astype(np.uint8)
        for name, probability in probabilities.items()
    }
    return masks, probabilities, len(xs) * len(ys)


def runtime_metadata() -> Dict[str, object]:
    return {
        "service_version": app.version,
        "device": DEVICE,
        "fp16": bool(USE_FP16 and DEVICE.startswith("cuda")),
        "epoc": USE_EPOC,
        "checkpoint_sha256": MODEL_STATE.get("checkpoint_sha256"),
    }


def autocast_context():
    if not (USE_FP16 and DEVICE.startswith("cuda")):
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.cuda.amp.autocast(dtype=torch.float16)


@app.get("/health")
async def health() -> Dict[str, object]:
    return {"status": "ok", "model_ready": bool(MODEL_STATE["ready"]), **runtime_metadata()}


@app.get("/ready")
async def ready() -> Dict[str, object]:
    if not MODEL_STATE["ready"] or sam_model is None:
        raise HTTPException(status_code=503, detail=MODEL_STATE.get("error") or "RemoteSAM is not ready")
    return {"status": "ready", **runtime_metadata()}


@app.get("/model-info")
async def model_info() -> Dict[str, object]:
    return {**MODEL_STATE, **runtime_metadata(), "checkpoint_name": CHECKPOINT_PATH.name}


@app.post("/predict")
async def predict(request: PredictRequest) -> Dict[str, object]:
    if sam_model is None or not MODEL_STATE["ready"]:
        raise HTTPException(status_code=503, detail=MODEL_STATE.get("error") or "RemoteSAM is not ready")
    if request.mask_size is not None:
        raise HTTPException(
            status_code=400,
            detail="mask_size is not supported by the upstream RemoteSAM API; output always matches the input image",
        )

    total_started = time.perf_counter()
    try:
        image_bgr = base64_to_cv2(request.image)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_hw = image_bgr.shape[:2]
        requested_quality_mode = request.quality_mode.strip().lower()
        resolved_quality_mode = resolve_quality_mode(request.quality_mode, image_hw)
        preprocess_ms = (time.perf_counter() - total_started) * 1000

        if DEVICE.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        inference_started = time.perf_counter()
        with INFERENCE_LOCK, torch.inference_mode():
            with autocast_context():
                if request.task in ("referring_seg", "referring_segmentation"):
                    if not request.text.strip():
                        raise ValueError("Text prompt is required for referring segmentation")
                    if resolved_quality_mode == "accurate":
                        mask, probability, tile_count = _accurate_referring_seg(
                            image_rgb, request.text.strip()
                        )
                    else:
                        raw = sam_model.referring_seg(
                            image=image_rgb,
                            sentence=request.text.strip(),
                            return_prob=request.return_probability,
                        )
                        if request.return_probability:
                            raw_mask, raw_probability = raw
                            probability = normalize_probability(raw_probability, image_hw)
                        else:
                            raw_mask, probability = raw, None
                        mask = normalize_mask(raw_mask, image_hw)
                        tile_count = 1
                    per_class_masks = None

                elif request.task in ("semantic_seg", "semantic_segmentation"):
                    classes = [item.strip() for item in request.classes if item.strip()]
                    if not classes:
                        raise ValueError("Classes are required for semantic segmentation")
                    if resolved_quality_mode == "accurate":
                        masks, probabilities, tile_count = _accurate_semantic_seg(
                            image_rgb, classes
                        )
                    else:
                        raw = sam_model.semantic_seg(
                            image=image_rgb,
                            classnames=classes,
                            return_prob=request.return_probability,
                        )
                        if request.return_probability:
                            raw_masks, raw_probabilities = raw
                        else:
                            raw_masks, raw_probabilities = raw, {}
                        masks = {}
                        probabilities = {}
                        for class_name in classes:
                            if class_name not in raw_masks:
                                continue
                            masks[class_name] = normalize_mask(
                                raw_masks[class_name], image_hw
                            )
                            if request.return_probability and class_name in raw_probabilities:
                                probabilities[class_name] = normalize_probability(
                                    raw_probabilities[class_name], image_hw
                                )
                        tile_count = 1
                    if not masks:
                        raise ValueError("RemoteSAM returned no class masks")
                    mask = np.maximum.reduce(list(masks.values()))
                    probability = (
                        np.maximum.reduce(list(probabilities.values())) if probabilities else None
                    )
                    per_class_masks = {name: mask_to_base64(value) for name, value in masks.items()}
                else:
                    raise ValueError(f"Unsupported task: {request.task}")

        inference_ms = (time.perf_counter() - inference_started) * 1000
        postprocess_started = time.perf_counter()
        visualization = apply_mask(image_bgr, mask)
        quality = quality_summary(mask, probability)
        candidates = candidate_summary(mask, request.candidate_min_area_pixels)
        warnings = ["RemoteSAM returned an empty mask; this does not prove the target is absent"] if quality["empty_mask"] else []
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
        peak_vram_mb = (
            round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
            if DEVICE.startswith("cuda")
            else None
        )

        response: Dict[str, object] = {
            "status": "success",
            "task": request.task,
            "mask": mask_to_base64(mask),
            "result_image": cv2_to_base64(visualization),
            "original_size": [int(image_bgr.shape[1]), int(image_bgr.shape[0])],
            "output_size": [int(mask.shape[1]), int(mask.shape[0])],
            "quality": quality,
            "quality_mode": resolved_quality_mode,
            "requested_quality_mode": requested_quality_mode,
            "mode_parameters": mode_parameters(
                requested_quality_mode, resolved_quality_mode, tile_count
            ),
            "candidate_stats": candidates,
            "warnings": warnings,
            "timing_ms": {
                "preprocess": round(preprocess_ms, 2),
                "inference": round(inference_ms, 2),
                "postprocess": round(postprocess_ms, 2),
                "total": round((time.perf_counter() - total_started) * 1000, 2),
            },
            "peak_vram_mb": peak_vram_mb,
            "model": runtime_metadata(),
        }
        if probability is not None and request.return_probability:
            response["probability_map"] = probability_to_base64(probability)
        if per_class_masks is not None:
            response["masks"] = per_class_masks
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("RemoteSAM inference failed")
        return {
            "status": "error",
            "task": request.task,
            "message": f"{type(exc).__name__}: {exc}",
            "timing_ms": {"total": round((time.perf_counter() - total_started) * 1000, 2)},
            "model": runtime_metadata(),
        }


if __name__ == "__main__":
    logger.info("RemoteSAM service: http://%s:%s", HOST, PORT)
    logger.info("API documentation: http://%s:%s/docs", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, workers=1)
