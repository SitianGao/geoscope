"""Prepare the local remote-sensing image set for the Agent evaluation.

This script is intentionally deterministic and re-runnable: it reproduces
``evaluation/images/`` from documented, on-disk sources and writes
``evaluation/dataset_manifest.csv`` with provenance, size and SHA-256.

The binary images themselves are *not* committed (the upstream sample images
have license terms the team has not fully cleared for redistribution). The
manifest plus this script are committed so a fresh checkout can rebuild the
image set from the documented sources.

Run from the repository root::

    python evaluation/prepare_images.py

Only Pillow and the standard library are required.
"""
from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REMOTE_AGENT = Path(r"D:\AI算法\GeoScope-RemoteAgent-workflow\RemoteAgent-main")
IMAGES_DIR = Path(__file__).resolve().parent / "images"
MANIFEST_PATH = Path(__file__).resolve().parent / "dataset_manifest.csv"

MAX_DIM = 1600  # cap the long side so vLLM's mm-processor-kwargs stay within budget

# image_id -> (category, source_path, license_note)
# source_path is either repo-relative or absolute under UPSTREAM_REMOTE_AGENT.
IMAGES = {
    "urban": {
        "category": "building",
        "source": "web/assets/urban.png",
        "abs": ROOT / "web" / "assets" / "urban.png",
        "license": "SM3Det CC BY-NC 4.0 (original 5428.tif); individual scene provenance not independently verified",
    },
    "airport": {
        "category": "aircraft",
        "source": "web/assets/airport.jpg",
        "abs": ROOT / "web" / "assets" / "airport.jpg",
        "license": "RemoteSAM-master assets/demo.jpg (public demo image; license not independently verified)",
    },
    "hkust": {
        "category": "building",
        "source": "RemoteAgent-main/servers/RemoteSAM/assets/HKUST.jpg",
        "abs": UPSTREAM_REMOTE_AGENT / "servers" / "RemoteSAM" / "assets" / "HKUST.jpg",
        "license": "RemoteSAM public repo sample (license not independently verified)",
    },
    "nuist": {
        "category": "building",
        "source": "RemoteAgent-main/servers/RemoteSAM/assets/NUIST.jpg",
        "abs": UPSTREAM_REMOTE_AGENT / "servers" / "RemoteSAM" / "assets" / "NUIST.jpg",
        "license": "RemoteSAM public repo sample (license not independently verified)",
    },
    "aerial_1831": {
        "category": "building",
        "source": "RemoteAgent-main/servers/SM3Det/tools/cam/1831.tif",
        "abs": UPSTREAM_REMOTE_AGENT / "servers" / "SM3Det" / "tools" / "cam" / "1831.tif",
        "license": "SM3Det CC BY-NC 4.0; individual scene provenance not independently verified",
    },
    "satellite_qingming": {
        "category": "building",
        "source": "RemoteAgent-main/servers/DirectSAM/assets/DirectSAM_qingming.jpg",
        "abs": UPSTREAM_REMOTE_AGENT / "servers" / "DirectSAM" / "assets" / "DirectSAM_qingming.jpg",
        "license": "DirectSAM public repo sample (license not independently verified)",
    },
    "satellite_mixed": {
        "category": "mixed",
        "source": "RemoteAgent-main/servers/DirectSAM/assets/DirectSAM_visualizations.jpg",
        "abs": UPSTREAM_REMOTE_AGENT / "servers" / "DirectSAM" / "assets" / "DirectSAM_visualizations.jpg",
        "license": "DirectSAM public repo sample (license not independently verified)",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def to_rgb_png(src: Path, dst: Path) -> None:
    im = Image.open(src)
    im = im.convert("RGB")
    w, h = im.size
    long_side = max(w, h)
    if long_side > MAX_DIM:
        scale = MAX_DIM / long_side
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    im.save(dst, "PNG")


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_id, meta in IMAGES.items():
        src = meta["abs"]
        if not src.is_file():
            raise SystemExit(f"Missing source image: {src}")
        dst = IMAGES_DIR / f"{image_id}.png"
        to_rgb_png(src, dst)
        im = Image.open(dst)
        w, h = im.size
        rows.append(
            {
                "image_id": image_id,
                "filename": dst.name,
                "category": meta["category"],
                "width": w,
                "height": h,
                "sha256": sha256(dst),
                "source": meta["source"],
                "license_note": meta["license"],
            }
        )

    fields = ["image_id", "filename", "category", "width", "height", "sha256", "source", "license_note"]
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Prepared {len(rows)} images under {IMAGES_DIR}")
    print(f"Wrote manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
