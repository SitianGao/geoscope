"""Select reproducible, paired WHU Building Dataset I evaluation samples.

The source archive is downloaded from the WHU GPCV page. This script never
extracts the whole archive or writes dataset images into the Git repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

from PIL import Image


SOURCE_URL = (
    "https://gpcv.whu.edu.cn/data/"
    "Satellite%20dataset%20%E2%85%A0%20%28global%20cities%29.zip"
)
FIELDS = (
    "image_id",
    "dataset",
    "class_name",
    "source_file",
    "image_path",
    "ground_truth_path",
    "width",
    "height",
    "ground_truth_foreground_pixels",
    "ground_truth_foreground_ratio",
    "source_image_sha256",
    "source_label_sha256",
    "image_sha256",
    "ground_truth_sha256",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archive_entries(archive: zipfile.ZipFile) -> dict[str, tuple[str, str]]:
    by_kind: dict[str, dict[str, str]] = {"image": {}, "label": {}}
    for name in archive.namelist():
        parts = name.replace("\\", "/").split("/")
        if len(parts) != 3 or parts[1] not in by_kind:
            continue
        if not parts[2].lower().endswith((".tif", ".tiff")):
            continue
        kind, filename = parts[1], parts[2]
        if filename in by_kind[kind]:
            raise ValueError(f"Duplicate {kind} file: {filename}")
        by_kind[kind][filename] = name
    images, labels = by_kind["image"], by_kind["label"]
    if not images or images.keys() != labels.keys():
        raise ValueError("WHU image and label filenames are not paired one-to-one")
    return {name: (images[name], labels[name]) for name in images}


def validate_pair(
    archive: zipfile.ZipFile, image_entry: str, label_entry: str
) -> tuple[int, int, int]:
    with Image.open(io.BytesIO(archive.read(image_entry))) as image:
        size = image.size
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB source image: {image_entry}")
    with Image.open(io.BytesIO(archive.read(label_entry))) as label:
        if label.size != size:
            raise ValueError(f"Mismatched label size: {label_entry}")
        colors = label.convert("RGB").getcolors(maxcolors=3)
        if colors is None or any(
            color not in ((0, 0, 0), (255, 255, 255)) for _, color in colors
        ):
            raise ValueError(f"Non-binary WHU label: {label_entry}")
        foreground = sum(count for count, color in colors if color == (255, 255, 255))
    return size[0], size[1], foreground


def select_names(candidates: list[str], count: int, seed: str) -> list[str]:
    ordered = sorted(
        candidates,
        key=lambda name: (sha256(f"{seed}:{name}".encode()), name),
    )
    selected: list[str] = []
    groups: set[str] = set()
    for name in ordered:
        group = name.split("_", 1)[0]
        if group not in groups:
            selected.append(name)
            groups.add(group)
        if len(selected) == count:
            return selected
    for name in ordered:
        if name not in selected:
            selected.append(name)
        if len(selected) == count:
            return selected
    raise ValueError(f"Only {len(candidates)} eligible images for {count} requested samples")


def write_png(image: Image.Image, path: Path) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    path.write_bytes(data)
    return sha256(data)


def prepare(archive_path: Path, output_dir: Path, count: int, seed: str) -> list[dict]:
    if count < 1:
        raise ValueError("Sample count must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"Archive CRC check failed: {bad_member}")
        pairs = archive_entries(archive)
        inspected = {
            name: validate_pair(archive, *entries)
            for name, entries in sorted(pairs.items())
        }
        eligible = [
            name
            for name, (width, height, foreground) in inspected.items()
            if 0 < foreground < width * height
        ]
        selected = select_names(eligible, count, seed)

        image_dir = output_dir / "images"
        label_dir = output_dir / "ground_truth"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        rows = []
        for filename in selected:
            image_entry, label_entry = pairs[filename]
            source_image = archive.read(image_entry)
            source_label = archive.read(label_entry)
            stem = Path(filename).stem
            image_id = f"WHU_SAT1_{stem}"
            image_rel = Path("images") / f"{image_id}.png"
            label_rel = Path("ground_truth") / f"{image_id}.png"
            with Image.open(io.BytesIO(source_image)) as image:
                image_hash = write_png(image.convert("RGB"), output_dir / image_rel)
            with Image.open(io.BytesIO(source_label)) as label:
                binary = label.convert("RGB").convert("L")
                label_hash = write_png(binary, output_dir / label_rel)
            width, height, foreground = inspected[filename]
            rows.append(
                {
                    "image_id": image_id,
                    "dataset": "WHU Building Dataset - Satellite Dataset I",
                    "class_name": "building",
                    "source_file": filename,
                    "image_path": image_rel.as_posix(),
                    "ground_truth_path": label_rel.as_posix(),
                    "width": width,
                    "height": height,
                    "ground_truth_foreground_pixels": foreground,
                    "ground_truth_foreground_ratio": f"{foreground / (width * height):.8f}",
                    "source_image_sha256": sha256(source_image),
                    "source_label_sha256": sha256(source_label),
                    "image_sha256": image_hash,
                    "ground_truth_sha256": label_hash,
                }
            )

    with (output_dir / "selected_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "selection_metadata.json").write_text(
        json.dumps(
            {
                "source_url": SOURCE_URL,
                "archive_sha256": sha256(archive_path.read_bytes()),
                "candidate_pairs": len(pairs),
                "eligible_positive_pairs": len(eligible),
                "selection": "SHA-256 rank by seed and filename; prefer distinct filename prefixes",
                "seed": seed,
                "selected_count": len(rows),
                "accuracy_scope": "selected diagnostic sample, not a full-dataset benchmark",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", default="geoscope-competition-20260927")
    args = parser.parse_args()
    rows = prepare(args.archive, args.output_dir, args.count, args.seed)
    print(f"Prepared {len(rows)} paired WHU building samples in {args.output_dir}")


if __name__ == "__main__":
    main()
