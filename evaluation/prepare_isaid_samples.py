"""Select paired iSAID plane samples from official validation downloads.

The iSAID archive holds pixel labels. Its matching RGB images live in the
DOTA-v1.0 validation images archive. The DOTA archive may be local or read via
HTTP byte ranges from the authors' public Google Drive file, without fetching
the entire multi-gigabyte ZIP.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import re
import zipfile
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import requests
from PIL import Image


ISAID_MASK_SOURCE = "https://drive.google.com/drive/folders/17MErPhWQrwr92Ca1Maf4mwiarPS5rcWM"
DOTA_IMAGE_SOURCE = "https://drive.google.com/drive/folders/1n5w45suVOyaqY84hltJhIZdtVFD9B224"
DOTA_VAL_PART1_FILE_ID = "1uCCCFhFQOJLfjBpcL5MC0DHJ9lgOaXWP"
PLANE_RGB = np.array((0, 127, 255), dtype=np.uint8)
MASK_NAME = re.compile(r"(?:^|/)(P\d+)_instance_color_RGB\.png$")
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_drive_download(session: requests.Session, file_id: str) -> str:
    response = session.get(
        "https://drive.google.com/uc",
        params={"export": "download", "id": file_id},
        timeout=60,
        stream=True,
    )
    response.raise_for_status()
    try:
        if "text/html" not in response.headers.get("Content-Type", ""):
            return response.url
        page = response.text
    finally:
        response.close()
    form = re.search(r'<form[^>]+action="([^"]+)"[^>]*>(.*?)</form>', page, re.S)
    if not form:
        raise ValueError("Google Drive did not expose a download form")
    action = html.unescape(form.group(1))
    inputs = {
        name: html.unescape(value)
        for name, value in re.findall(
            r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', form.group(2)
        )
    }
    if inputs.get("id") != file_id or inputs.get("confirm") != "t":
        raise ValueError("Unexpected Google Drive confirmation form")
    return f"{action}?{urlencode(inputs)}"


class RemoteZipReader(io.RawIOBase):
    """Seekable, cached HTTP-range reader accepted by ``zipfile.ZipFile``."""

    BLOCK = 4 * 1024 * 1024
    MAX_BLOCKS = 8

    def __init__(self, session: requests.Session, download_url: str) -> None:
        self.session = session
        self.url = download_url
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        response = session.head(
            download_url, headers={"Range": "bytes=0-0"}, timeout=60
        )
        response.raise_for_status()
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes 0-0/(\d+)", content_range)
        if response.status_code != 206 or not match:
            raise ValueError("DOTA archive endpoint does not support byte ranges")
        self.size = int(match.group(1))

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_CUR:
            offset += self.position
        elif whence == io.SEEK_END:
            offset += self.size
        elif whence != io.SEEK_SET:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if offset < 0:
            raise ValueError("Negative archive seek")
        self.position = offset
        return offset

    def _block(self, index: int) -> bytes:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start = index * self.BLOCK
        end = min(start + self.BLOCK, self.size) - 1
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=120,
        )
        response.raise_for_status()
        expected_range = f"bytes {start}-{end}/{self.size}"
        if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
            raise IOError(f"Unexpected range response for {start}-{end}")
        data = response.content
        if len(data) != end - start + 1:
            raise IOError(f"Truncated range response for {start}-{end}")
        self.cache[index] = data
        if len(self.cache) > self.MAX_BLOCKS:
            self.cache.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        pieces = []
        while size > 0:
            index, offset = divmod(self.position, self.BLOCK)
            block = self._block(index)
            part = block[offset : offset + size]
            pieces.append(part)
            self.position += len(part)
            size -= len(part)
        return b"".join(pieces)


def mask_candidates(archive: zipfile.ZipFile, min_pixels: int) -> dict[str, dict]:
    if archive.testzip() is not None:
        raise ValueError("iSAID mask archive failed CRC validation")
    found: dict[str, dict] = {}
    for entry in archive.namelist():
        match = MASK_NAME.search(entry)
        if not match:
            continue
        stem = match.group(1)
        if stem in found:
            raise ValueError(f"Duplicate iSAID mask ID: {stem}")
        source = archive.read(entry)
        with Image.open(io.BytesIO(source)) as image:
            if image.mode != "RGB":
                raise ValueError(f"Expected RGB iSAID mask: {entry}")
            array = np.asarray(image)
            foreground = int(np.count_nonzero(np.all(array == PLANE_RGB, axis=2)))
            width, height = image.size
        found[stem] = {
            "entry": entry,
            "width": width,
            "height": height,
            "foreground": foreground,
        }
    if not found:
        raise ValueError("No iSAID validation masks found")
    return {
        stem: item
        for stem, item in found.items()
        if min_pixels <= item["foreground"] < item["width"] * item["height"]
    }


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def prepare(
    masks_path: Path,
    output_dir: Path,
    count: int,
    min_pixels: int,
    seed: str,
    dota_archive_path: Path | None,
    dota_drive_file_id: str | None,
) -> list[dict]:
    if count < 1 or min_pixels < 1:
        raise ValueError("Count and minimum foreground pixels must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if (dota_archive_path is None) == (dota_drive_file_id is None):
        raise ValueError("Provide exactly one DOTA image archive source")

    with zipfile.ZipFile(masks_path) as masks:
        candidates = mask_candidates(masks, min_pixels)
        ranked = sorted(
            candidates,
            key=lambda stem: (sha256(f"{seed}:{stem}".encode()), stem),
        )
        if len(ranked) < count:
            raise ValueError(f"Only {len(ranked)} eligible plane images for {count} requested")

        session = requests.Session()
        download_url = None
        if dota_archive_path is not None:
            dota_stream = dota_archive_path.open("rb")
        else:
            download_url = resolve_drive_download(session, dota_drive_file_id)
            dota_stream = RemoteZipReader(session, download_url)

        try:
            with dota_stream, zipfile.ZipFile(dota_stream) as dota:
                image_entries: dict[str, str] = {}
                for entry in dota.namelist():
                    stem = Path(entry).stem
                    if stem in candidates and entry.lower().endswith(".png"):
                        if stem in image_entries:
                            raise ValueError(f"Duplicate DOTA image ID: {stem}")
                        image_entries[stem] = entry
                paired = [stem for stem in ranked if stem in image_entries]
                if len(paired) < count:
                    raise ValueError(f"Only {len(paired)} plane masks have DOTA images")
                selected = paired[:count]

                image_dir = output_dir / "images"
                mask_dir = output_dir / "ground_truth"
                image_dir.mkdir(parents=True)
                mask_dir.mkdir(parents=True)
                rows = []
                for stem in selected:
                    image_source = dota.read(image_entries[stem])
                    mask_source = masks.read(candidates[stem]["entry"])
                    with Image.open(io.BytesIO(image_source)) as image:
                        width, height = image.size
                        image_output = png_bytes(image.convert("RGB"))
                    if (width, height) != (
                        candidates[stem]["width"],
                        candidates[stem]["height"],
                    ):
                        raise ValueError(f"DOTA/iSAID size mismatch for {stem}")
                    with Image.open(io.BytesIO(mask_source)) as mask:
                        array = np.asarray(mask.convert("RGB"))
                    foreground = np.all(array == PLANE_RGB, axis=2)
                    count_foreground = int(np.count_nonzero(foreground))
                    if count_foreground != candidates[stem]["foreground"]:
                        raise ValueError(f"Inconsistent plane pixel count for {stem}")
                    mask_output = png_bytes(Image.fromarray(foreground.astype(np.uint8) * 255, "L"))

                    image_id = f"ISAID_VAL_{stem}"
                    image_rel = Path("images") / f"{image_id}.png"
                    mask_rel = Path("ground_truth") / f"{image_id}.png"
                    (output_dir / image_rel).write_bytes(image_output)
                    (output_dir / mask_rel).write_bytes(mask_output)
                    rows.append(
                        {
                            "image_id": image_id,
                            "dataset": "iSAID validation + DOTA v1.0 validation",
                            "class_name": "plane",
                            "source_file": image_entries[stem],
                            "image_path": image_rel.as_posix(),
                            "ground_truth_path": mask_rel.as_posix(),
                            "width": width,
                            "height": height,
                            "ground_truth_foreground_pixels": count_foreground,
                            "ground_truth_foreground_ratio": f"{count_foreground / (width * height):.8f}",
                            "source_image_sha256": sha256(image_source),
                            "source_label_sha256": sha256(mask_source),
                            "image_sha256": sha256(image_output),
                            "ground_truth_sha256": sha256(mask_output),
                        }
                    )
                    print(f"Prepared {image_id} ({width}x{height}, {count_foreground} plane pixels)", flush=True)
        finally:
            session.close()

    with (output_dir / "selected_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "selection_metadata.json").write_text(
        json.dumps(
            {
                "iSAID_mask_source": ISAID_MASK_SOURCE,
                "iSAID_mask_archive_sha256": file_sha256(masks_path),
                "DOTA_image_source": DOTA_IMAGE_SOURCE,
                "DOTA_local_archive_sha256": file_sha256(dota_archive_path) if dota_archive_path else None,
                "DOTA_drive_file_id": dota_drive_file_id,
                "candidate_positive_masks": len(candidates),
                "selection": "SHA-256 rank by fixed seed and image ID among positive, paired masks",
                "seed": seed,
                "min_plane_pixels": min_pixels,
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
    parser.add_argument("--masks-archive", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dota-archive", type=Path)
    source.add_argument("--dota-drive-file-id", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--min-plane-pixels", type=int, default=1000)
    parser.add_argument("--seed", default="geoscope-competition-20260927")
    args = parser.parse_args()
    rows = prepare(
        args.masks_archive,
        args.output_dir,
        args.count,
        args.min_plane_pixels,
        args.seed,
        args.dota_archive,
        args.dota_drive_file_id,
    )
    print(f"Prepared {len(rows)} paired iSAID plane samples in {args.output_dir}")


if __name__ == "__main__":
    main()
