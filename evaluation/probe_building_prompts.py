"""Compare RemoteSAM building prompts on one image without claiming accuracy."""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image


CASES = (
    ("ref_all_buildings", {"task": "referring_seg", "text": "all buildings"}),
    ("ref_buildings", {"task": "referring_seg", "text": "buildings"}),
    ("ref_all_houses", {"task": "referring_seg", "text": "all houses"}),
    ("semantic_building", {"task": "semantic_seg", "classes": ["building"]}),
)


def post_json(url, payload, timeout=180):
    request = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def foreground_pixels(mask):
    return sum(mask.convert("L").histogram()[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--sam-url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    image_path = Path(args.image)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    summary = []

    for name, request in CASES:
        response = post_json(args.sam_url, {**request, "image": encoded_image, "quality_mode": "accurate"})
        encoded_mask = response.get("mask")
        if not encoded_mask and response.get("masks"):
            encoded_mask = response["masks"].get("building")
        if response.get("status") != "success" or not encoded_mask:
            summary.append({"case": name, "status": response.get("status", "error"), "message": response.get("message")})
            continue
        mask = Image.open(io.BytesIO(base64.b64decode(encoded_mask))).convert("L")
        if mask.size != image.size:
            raise ValueError(f"{name}: mask size {mask.size} does not match image {image.size}")
        area = foreground_pixels(mask)
        mask.save(output / f"{name}_mask.png")
        overlay = image.copy()
        tint = Image.new("RGB", image.size, (69, 213, 152))
        overlay.paste(Image.blend(image, tint, 0.48), (0, 0), mask)
        overlay.save(output / f"{name}_overlay.jpg", quality=92)
        summary.append({
            "case": name,
            "status": "success",
            "pixel_area": area,
            "area_ratio": area / (image.width * image.height),
            "candidate_stats": response.get("candidate_stats"),
            "quality_mode": response.get("quality_mode"),
        })

    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
