"""Verify the official Agent -> Harness -> RemoteSAM -> Agent round trip."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import server  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--query", default="Extract all buildings in this image using RemoteSAM.")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")
    for key in ("GEO_AGENT_BASE_URL", "GEO_AGENT_MODEL", "GEO_REMOTESAM_URL"):
        if not os.environ.get(key):
            parser.error(f"missing {key}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    server.DATA = args.output_dir
    server.SESSIONS.clear()
    uploaded = base64.b64encode(args.image.read_bytes()).decode("ascii")
    session = server.SESSIONS[
        server.new_session(uploaded=uploaded, name=args.image.name)["id"]
    ]
    raw: dict[str, str | None] = {"decision": None, "completion": None}
    feedback = None
    actual_plan = server.live_plan
    actual_finish = server.finish_with_agent

    def record_plan(*positional):
        plan, agent, decision = actual_plan(*positional)
        raw["decision"] = decision.raw_response
        return plan, agent, decision

    def record_completion(*positional):
        nonlocal feedback
        feedback = positional[-1]
        decision = actual_finish(*positional)
        raw["completion"] = decision.raw_response
        return decision

    with (
        patch.object(server, "live_plan", side_effect=record_plan),
        patch.object(server, "finish_with_agent", side_effect=record_completion),
    ):
        result = server.run_task(session, {"query": args.query, "mode": "live"})

    run_dir = args.output_dir / session["id"] / result["id"]
    with Image.open(args.image) as original, Image.open(args.output_dir / session["id"] / "original.png") as session_image:
        same_pixels = (
            original.size == session_image.size
            and ImageChops.difference(original.convert("RGB"), session_image.convert("RGB")).getbbox() is None
        )
    mask_path = run_dir / "mask.png"
    mask_check = None
    if mask_path.is_file():
        with Image.open(mask_path) as mask:
            histogram = mask.convert("L").histogram()
            mask_check = {
                "size": list(mask.size),
                "values": [index for index, count in enumerate(histogram) if count],
                "foreground_pixels": sum(histogram[1:]),
                "sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            }

    record = {
        "service": os.environ["GEO_AGENT_BASE_URL"],
        "model": os.environ["GEO_AGENT_MODEL"],
        "remotesam_url": os.environ["GEO_REMOTESAM_URL"],
        "source_image": str(args.image.resolve()),
        "source_image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "session_image_sha256": result["provenance"]["image_sha256"],
        "session_image_pixels_equal_source": same_pixels,
        "query": args.query,
        "status": result["status"],
        "duration_ms": result["duration_ms"],
        "message": result["message"],
        "task": result.get("task"),
        "agent": result.get("agent"),
        "raw_agent_responses": raw,
        "feedback_to_agent": feedback,
        "metrics": result.get("metrics"),
        "mask_check": mask_check,
        "full_mask_sha256": result.get("full_mask_sha256"),
        "trace": result["trace"],
        "session_id": session["id"],
        "run_id": result["id"],
    }
    (args.output_dir / "harness_verification.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))

    success = (
        record["status"] == "completed"
        and record["session_image_pixels_equal_source"]
        and record["agent"]["decision"]["status"] == "tool_call"
        and record["agent"]["decision"]["tool_call"]["name"] in {
            "semantic_segmentation", "referring_expression_segmentation"
        }
        and record["agent"]["completion"]["status"] == "completed"
        and record["mask_check"] is not None
        and set(record["mask_check"]["values"]).issubset({0, 255})
        and record["mask_check"]["size"] == [session["width"], session["height"]]
        and record["mask_check"]["foreground_pixels"] == record["metrics"]["pixel_area"]
        and record["metrics"]["pixel_area"] > 0
        and record["metrics"]["spatial_check"] is True
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
