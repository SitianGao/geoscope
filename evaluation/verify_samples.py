"""Independently verify selected image/mask pairs and their manifest hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> int:
    with (root / "selected_manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty manifest: {root}")
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {root}")
    for row in rows:
        image_path = root / row["image_path"]
        mask_path = root / row["ground_truth_path"]
        if file_sha256(image_path) != row["image_sha256"]:
            raise ValueError(f"Image hash mismatch: {row['image_id']}")
        if file_sha256(mask_path) != row["ground_truth_sha256"]:
            raise ValueError(f"Mask hash mismatch: {row['image_id']}")
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            expected_size = (int(row["width"]), int(row["height"]))
            if image.size != expected_size or mask.size != expected_size:
                raise ValueError(f"Image/mask size mismatch: {row['image_id']}")
            if image.mode != "RGB" or mask.mode != "L":
                raise ValueError(f"Unexpected image/mask mode: {row['image_id']}")
            pixels = np.asarray(mask)
            if not np.isin(pixels, (0, 255)).all():
                raise ValueError(f"Non-binary mask: {row['image_id']}")
            foreground = int(np.count_nonzero(pixels))
            if foreground != int(row["ground_truth_foreground_pixels"]):
                raise ValueError(f"Foreground count mismatch: {row['image_id']}")
            if foreground == 0 or foreground == pixels.size:
                raise ValueError(f"Degenerate mask: {row['image_id']}")
    return len(rows)


def verify_dota_archive(root: Path, archive_path: Path) -> int:
    with (root / "selected_manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with zipfile.ZipFile(archive_path) as archive:
        for row in rows:
            if row["class_name"] != "plane":
                raise ValueError(f"Non-plane row in DOTA check: {row['image_id']}")
            source_bytes = archive.read(row["source_file"])
            if hashlib.sha256(source_bytes).hexdigest() != row["source_image_sha256"]:
                raise ValueError(f"DOTA source hash mismatch: {row['image_id']}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_dirs", nargs="+", type=Path)
    parser.add_argument("--dota-archive", type=Path)
    args = parser.parse_args()
    for root in args.sample_dirs:
        print(f"Verified {verify(root)} paired samples: {root}")
    if args.dota_archive:
        plane_roots = []
        for root in args.sample_dirs:
            with (root / "selected_manifest.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if rows and all(row["class_name"] == "plane" for row in rows):
                plane_roots.append(root)
        if len(plane_roots) != 1:
            raise ValueError("Pass exactly one iSAID sample directory with --dota-archive")
        print(
            f"Matched {verify_dota_archive(plane_roots[0], args.dota_archive)} "
            f"source images: {args.dota_archive}"
        )


if __name__ == "__main__":
    main()
