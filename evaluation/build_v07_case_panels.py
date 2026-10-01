"""Build local best/middle/worst v0.7 comparison panels.

The generated dataset-derived images are intentionally written outside Git.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def overlay_truth(image: Image.Image, truth_path: Path) -> Image.Image:
    with Image.open(truth_path) as source:
        mask = np.asarray(source.convert("L"), dtype=np.uint8) > 0
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    array[mask] = np.rint(
        array[mask].astype(np.float32) * 0.5
        + np.array([0, 255, 255], dtype=np.float32) * 0.5
    ).astype(np.uint8)
    return Image.fromarray(array)


def tile(image: Image.Image, title: str, width: int, height: int) -> Image.Image:
    header = 36
    body = ImageOps.contain(image.convert("RGB"), (width, height - header))
    panel = Image.new("RGB", (width, height), "white")
    panel.paste(body, ((width - body.width) // 2, header + (height - header - body.height) // 2))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 10), title, fill="black", font=ImageFont.load_default())
    return panel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/dataset_manifest.csv"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        manifest = {row["image_id"]: row for row in csv.DictReader(stream)}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for label in ("best", "middle", "worst"):
        case = summary["cases"][label]
        image_id = case["image_id"]
        source = manifest[image_id]
        with Image.open(args.data_root / source["image_path"]) as raw:
            original = raw.convert("RGB")
        truth = overlay_truth(original, args.data_root / source["ground_truth_path"])
        with Image.open(args.artifact_dir / image_id / "fast" / "overlay.png") as raw:
            fast = raw.convert("RGB")
        with Image.open(args.artifact_dir / image_id / "accurate" / "overlay.png") as raw:
            accurate = raw.convert("RGB")

        width, height = 720, 540
        canvas = Image.new("RGB", (width * 2, height * 2), "#eeeeee")
        panels = [
            tile(original, f"Original - {image_id}", width, height),
            tile(truth, "Ground truth overlay", width, height),
            tile(fast, "Fast prediction overlay", width, height),
            tile(
                accurate,
                f"Accurate overlay - IoU {float(case['iou']):.4f}, Dice {float(case['dice']):.4f}",
                width,
                height,
            ),
        ]
        for index, panel in enumerate(panels):
            canvas.paste(panel, ((index % 2) * width, (index // 2) * height))
        output = args.output_dir / f"{label}_{image_id}.png"
        canvas.save(output, optimize=True)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
