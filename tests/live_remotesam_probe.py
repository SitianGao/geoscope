"""Run a real-image RemoteSAM probe and persist compact verification artifacts."""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.request
from pathlib import Path


def _decode_base64(value: str) -> bytes:
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--url", default="http://10.199.156.100:6657/predict")
    parser.add_argument("--text", default="buildings")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "referring_seg",
        "image": base64.b64encode(args.image.read_bytes()).decode("ascii"),
        "text": args.text,
        "return_probability": False,
    }
    request = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        status_code = response.status
        result = json.load(response)
    round_trip_ms = round((time.perf_counter() - started) * 1000, 2)

    mask_value = result.get("mask")
    if status_code != 200 or result.get("status") != "success" or not mask_value:
        raise RuntimeError(
            f"RemoteSAM probe failed: HTTP {status_code}, "
            f"status={result.get('status')!r}, message={result.get('message')!r}"
        )

    image_bytes = args.image.read_bytes()
    mask_bytes = _decode_base64(mask_value)
    (args.output_dir / f"input{args.image.suffix.lower()}").write_bytes(image_bytes)
    (args.output_dir / "mask.png").write_bytes(mask_bytes)

    compact = dict(result)
    for key, value in tuple(compact.items()):
        if isinstance(value, str) and len(value) > 1000:
            compact[key] = f"<omitted {len(value)} base64 characters>"
    compact["client_round_trip_ms"] = round_trip_ms
    (args.output_dir / "response.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    verification = {
        "http_status": status_code,
        "status": result.get("status"),
        "client_round_trip_ms": round_trip_ms,
        "timing_ms": result.get("timing_ms"),
        "peak_vram_mb": result.get("peak_vram_mb"),
        "checkpoint_sha256": result.get("checkpoint_sha256")
        or (result.get("model") or {}).get("checkpoint_sha256"),
        "original_size": result.get("original_size"),
        "output_size": result.get("output_size"),
        "quality": result.get("quality"),
    }

    try:
        from PIL import Image

        original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        pixels = (
            mask.get_flattened_data()
            if hasattr(mask, "get_flattened_data")
            else mask.getdata()
        )
        values = sorted(set(pixels))
        verification.update(
            {
                "decoded_original_size": list(original.size),
                "decoded_mask_size": list(mask.size),
                "mask_values": values,
                "mask_is_binary_0_255": set(values).issubset({0, 255}),
            }
        )
        if original.size != mask.size:
            raise RuntimeError(f"mask size {mask.size} != image size {original.size}")
        if not set(values).issubset({0, 255}):
            raise RuntimeError(f"mask contains non-binary pixel values: {values[:20]}")
        red = Image.new("RGB", original.size, (255, 0, 0))
        overlay = Image.blend(original, Image.composite(red, original, mask), 0.5)
        overlay.save(args.output_dir / "overlay.png")
        if result.get("result_image"):
            (args.output_dir / "service_result.png").write_bytes(
                _decode_base64(result["result_image"])
            )
    except ImportError:
        verification["pixel_validation"] = "skipped: Pillow is not installed"

    (args.output_dir / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
