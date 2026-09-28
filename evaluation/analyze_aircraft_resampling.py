"""Measure whether fixed aircraft ground-truth objects vanish under input resampling."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from run_remotesam_aircraft_experiment import positions
from run_remotesam_eval import MANIFEST_SHA256
from verify_dataset_manifest import verify as verify_dataset_manifest


FIELDS = (
    "image_id",
    "width",
    "height",
    "ground_truth_foreground_pixels",
    "ground_truth_components",
    "small_components_le_64px",
    "minimum_component_pixels",
    "median_component_pixels",
    "direct_896_roundtrip_iou",
    "direct_896_lost_components",
    "direct_896_lost_small_components",
    "upscale_1_25_then_896_roundtrip_iou",
    "upscale_1_25_then_896_lost_components",
    "upscale_1_25_then_896_lost_small_components",
    "tile_1024_overlap_256_roundtrip_iou",
    "tile_1024_overlap_256_lost_components",
    "tile_1024_overlap_256_lost_small_components",
)


def resize_binary(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(image.resize(size, Image.Resampling.NEAREST)) > 0


def direct_roundtrip(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    return resize_binary(resize_binary(mask, (896, 896)), (width, height))


def upscale_roundtrip(mask: np.ndarray, scale: float = 1.25) -> np.ndarray:
    height, width = mask.shape
    scaled_size = max(1, round(width * scale)), max(1, round(height * scale))
    scaled = resize_binary(mask, scaled_size)
    model_input = resize_binary(scaled, (896, 896))
    model_output = resize_binary(model_input, scaled_size)
    return resize_binary(model_output, (width, height))


def tiled_roundtrip(mask: np.ndarray, tile_size: int = 1024, overlap: int = 256) -> np.ndarray:
    height, width = mask.shape
    reconstructed = np.zeros_like(mask)
    for y in positions(height, tile_size, overlap):
        for x in positions(width, tile_size, overlap):
            crop = mask[y : min(y + tile_size, height), x : min(x + tile_size, width)]
            crop_height, crop_width = crop.shape
            padded = np.pad(
                crop,
                ((0, tile_size - crop_height), (0, tile_size - crop_width)),
                mode="edge",
            )
            restored = resize_binary(resize_binary(padded, (896, 896)), (tile_size, tile_size))
            reconstructed[y : y + crop_height, x : x + crop_width] |= restored[
                :crop_height, :crop_width
            ]
    return reconstructed


def analyze_reconstruction(
    truth: np.ndarray,
    reconstructed: np.ndarray,
    labels: np.ndarray,
    component_sizes: np.ndarray,
) -> tuple[float, int, int]:
    intersection = int(np.count_nonzero(truth & reconstructed))
    union = int(np.count_nonzero(truth | reconstructed))
    seen = np.unique(labels[reconstructed & truth])
    seen = seen[seen > 0]
    all_ids = np.arange(1, len(component_sizes) + 1)
    lost = np.setdiff1d(all_ids, seen, assume_unique=True)
    lost_small = sum(component_sizes[component_id - 1] <= 64 for component_id in lost)
    return intersection / union if union else 1.0, int(len(lost)), int(lost_small)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    if args.output_csv.exists():
        raise FileExistsError("Refusing to overwrite an existing analysis")
    verify_dataset_manifest(args.manifest, args.data_root, MANIFEST_SHA256)
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        sources = [row for row in csv.DictReader(stream) if row["class_name"] == "plane"]

    rows: list[dict[str, object]] = []
    structure = np.ones((3, 3), dtype=np.uint8)
    for source in sources:
        with Image.open(args.data_root / source["ground_truth_path"]) as image:
            truth = np.asarray(image) > 0
        labels, count = ndimage.label(truth, structure=structure)
        sizes = np.bincount(labels.ravel())[1:]
        direct = analyze_reconstruction(truth, direct_roundtrip(truth), labels, sizes)
        upscale = analyze_reconstruction(truth, upscale_roundtrip(truth), labels, sizes)
        tiled = analyze_reconstruction(truth, tiled_roundtrip(truth), labels, sizes)
        row = {
            "image_id": source["image_id"],
            "width": source["width"],
            "height": source["height"],
            "ground_truth_foreground_pixels": int(np.count_nonzero(truth)),
            "ground_truth_components": int(count),
            "small_components_le_64px": int(np.count_nonzero(sizes <= 64)),
            "minimum_component_pixels": int(sizes.min()),
            "median_component_pixels": float(np.median(sizes)),
            "direct_896_roundtrip_iou": direct[0],
            "direct_896_lost_components": direct[1],
            "direct_896_lost_small_components": direct[2],
            "upscale_1_25_then_896_roundtrip_iou": upscale[0],
            "upscale_1_25_then_896_lost_components": upscale[1],
            "upscale_1_25_then_896_lost_small_components": upscale[2],
            "tile_1024_overlap_256_roundtrip_iou": tiled[0],
            "tile_1024_overlap_256_lost_components": tiled[1],
            "tile_1024_overlap_256_lost_small_components": tiled[2],
        }
        rows.append(row)
        print(
            f"{source['image_id']}: components={count}, "
            f"lost direct/upscale/tile={direct[1]}/{upscale[1]}/{tiled[1]}",
            flush=True,
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
