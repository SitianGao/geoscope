"""Validate the frozen team manifest against an external local data root."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_PROMPTS = {"building": "all buildings", "plane": "all airplanes"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe manifest path: {relative}")
    target = (root / candidate).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes data root: {relative}")
    return target


def verify(manifest: Path, data_root: Path, expected_sha256: str | None) -> int:
    manifest_sha256 = file_sha256(manifest)
    if expected_sha256 and manifest_sha256.lower() != expected_sha256.lower():
        raise ValueError("Manifest SHA-256 does not match the frozen version")
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 20:
        raise ValueError(f"Expected 20 rows, got {len(rows)}")
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image IDs")
    classes = {name: 0 for name in EXPECTED_PROMPTS}
    for row in rows:
        image_id = row["image_id"]
        class_name = row["class_name"]
        if class_name not in EXPECTED_PROMPTS:
            raise ValueError(f"Unknown class: {image_id}")
        classes[class_name] += 1
        if row["protocol_version"] != "sam-eval-v1.0":
            raise ValueError(f"Unexpected protocol version: {image_id}")
        if row["task"] != "referring_seg" or row["prompt"] != EXPECTED_PROMPTS[class_name]:
            raise ValueError(f"Unexpected task/prompt: {image_id}")
        image_path = safe_path(data_root, row["image_path"])
        mask_path = safe_path(data_root, row["ground_truth_path"])
        if file_sha256(image_path) != row["image_sha256"]:
            raise ValueError(f"Image hash mismatch: {image_id}")
        if file_sha256(mask_path) != row["ground_truth_sha256"]:
            raise ValueError(f"Ground-truth hash mismatch: {image_id}")
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            expected_size = int(row["width"]), int(row["height"])
            if image.size != expected_size or mask.size != expected_size:
                raise ValueError(f"Image/mask size mismatch: {image_id}")
            if image.mode != "RGB" or mask.mode != "L":
                raise ValueError(f"Unexpected image/mask mode: {image_id}")
            pixels = np.asarray(mask)
            if not np.isin(pixels, (0, 255)).all():
                raise ValueError(f"Non-binary ground truth: {image_id}")
            if int(np.count_nonzero(pixels)) != int(row["ground_truth_foreground_pixels"]):
                raise ValueError(f"Foreground count mismatch: {image_id}")
    if classes != {"building": 10, "plane": 10}:
        raise ValueError(f"Unexpected class counts: {classes}")
    print(f"Verified 20 paired images (10 building, 10 plane); manifest SHA-256: {manifest_sha256}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    verify(args.manifest, args.data_root, args.expected_sha256)


if __name__ == "__main__":
    main()
