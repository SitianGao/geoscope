"""Run one real RemoteSAM request and preserve the evidence artifacts."""
from __future__ import annotations

import argparse
import base64
import json
import urllib.error
import urllib.request
from pathlib import Path


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def post_json(url: str, payload: dict):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:6657")
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", default="buildings")
    parser.add_argument("--output", default="outputs/smoke_test")
    parser.add_argument("--probability", action="store_true")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    try:
        health = get_json(args.url.rstrip("/") + "/health")
        ready = get_json(args.url.rstrip("/") + "/ready")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"RemoteSAM service is not ready: HTTP {exc.code} {detail}") from exc

    result = post_json(
        args.url.rstrip("/") + "/predict",
        {
            "task": "referring_seg",
            "image": base64.b64encode(image_path.read_bytes()).decode("ascii"),
            "text": args.text,
            "return_probability": args.probability,
        },
    )
    evidence = {"health": health, "ready": ready, "request": vars(args), "response": result}
    (output_dir / "smoke_test.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for key, filename in (
        ("mask", "mask.png"),
        ("result_image", "overlay.png"),
        ("probability_map", "probability.png"),
    ):
        if result.get(key):
            (output_dir / filename).write_bytes(base64.b64decode(result[key]))

    if result.get("status") != "success":
        raise SystemExit(f"Inference failed: {result.get('message', result)}")
    print(json.dumps({
        "status": result["status"],
        "output": str(output_dir),
        "size": result.get("output_size"),
        "quality": result.get("quality"),
        "timing_ms": result.get("timing_ms"),
        "peak_vram_mb": result.get("peak_vram_mb"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
