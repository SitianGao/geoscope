"""Build the frozen team manifest from the two validated local sample manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROTOCOL_VERSION = "sam-eval-v1.0"
WHU_PREFIX = Path("WHU/selected_20260927")
ISAID_PREFIX = Path("iSAID/selected_20260927")
WHU_SOURCE = "https://gpcv.whu.edu.cn/data/building_dataset.html"
ISAID_SOURCE = "https://captain-whu.github.io/iSAID/dataset.html"
DOTA_SOURCE = "https://captain-whu.github.io/DOTA/dataset.html"
WHU_ARCHIVE_SHA256 = "7208f419d3f900e2b00fe76f47e8635ad5a3265ecbc68a0f48620df43fed3681"
ISAID_MASK_ARCHIVE_SHA256 = "f3b333b4b0469edb1079c49c163c915ae8dd39d116191798633564bac1998d69"
DOTA_ARCHIVE_SHA256 = "e8f5602dc22c5142b3d2f26982c54df4e9e5e9b6b92015bc20a434e12b45ca9d"
FIELDS = (
    "protocol_version",
    "image_id",
    "dataset",
    "split",
    "project_target",
    "class_name",
    "task",
    "prompt",
    "image_path",
    "ground_truth_path",
    "width",
    "height",
    "ground_truth_foreground_pixels",
    "image_sha256",
    "ground_truth_sha256",
    "source_image_sha256",
    "source_label_sha256",
    "source_file",
    "source_image_archive_sha256",
    "source_label_archive_sha256",
    "source_image_url",
    "source_label_url",
)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10:
        raise ValueError(f"Expected exactly 10 rows in {path}; found {len(rows)}")
    return rows


def row_for_team(source: dict[str, str], *, building: bool) -> dict[str, str]:
    expected_class = "building" if building else "plane"
    if source["class_name"] != expected_class:
        raise ValueError(f"Unexpected class in {source['image_id']}")
    prefix = WHU_PREFIX if building else ISAID_PREFIX
    if not source["image_id"].startswith("WHU_SAT1_" if building else "ISAID_VAL_"):
        raise ValueError(f"Unexpected image ID: {source['image_id']}")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "image_id": source["image_id"],
        "dataset": "WHU Building Dataset - Satellite Dataset I" if building else "iSAID + DOTA v1.0",
        "split": "diagnostic" if building else "validation",
        "project_target": "building" if building else "aircraft",
        "class_name": expected_class,
        "task": "referring_seg",
        "prompt": "all buildings" if building else "all airplanes",
        "image_path": (prefix / source["image_path"]).as_posix(),
        "ground_truth_path": (prefix / source["ground_truth_path"]).as_posix(),
        "width": source["width"],
        "height": source["height"],
        "ground_truth_foreground_pixels": source["ground_truth_foreground_pixels"],
        "image_sha256": source["image_sha256"],
        "ground_truth_sha256": source["ground_truth_sha256"],
        "source_image_sha256": source["source_image_sha256"],
        "source_label_sha256": source["source_label_sha256"],
        "source_file": source["source_file"],
        "source_image_archive_sha256": WHU_ARCHIVE_SHA256 if building else DOTA_ARCHIVE_SHA256,
        "source_label_archive_sha256": WHU_ARCHIVE_SHA256 if building else ISAID_MASK_ARCHIVE_SHA256,
        "source_image_url": WHU_SOURCE if building else DOTA_SOURCE,
        "source_label_url": WHU_SOURCE if building else ISAID_SOURCE,
    }


def build(whu_manifest: Path, isaid_manifest: Path, output: Path) -> list[dict[str, str]]:
    rows = [row_for_team(row, building=True) for row in read_manifest(whu_manifest)]
    rows.extend(row_for_team(row, building=False) for row in read_manifest(isaid_manifest))
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image IDs")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whu-manifest", type=Path, required=True)
    parser.add_argument("--isaid-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build(args.whu_manifest, args.isaid_manifest, args.output)
    print(f"Frozen {len(rows)} rows in {args.output} ({PROTOCOL_VERSION})")


if __name__ == "__main__":
    main()
