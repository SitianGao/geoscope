"""Record two real-image decisions from an official RemoteAgent vLLM server."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "services" / "remoteagent" / "upstream"
sys.path.insert(0, str(UPSTREAM))

from remoteagent import RemoteAgent  # noqa: E402
from remoteagent.llm.vllm import VLLMChatClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")

    client = VLLMChatClient(args.url)
    models = client._request("models", timeout=10.0)
    entries = models.get("data")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("/v1/models returned no deployed model")
    model_name = args.model or entries[0].get("id")
    if not model_name:
        raise RuntimeError("/v1/models returned no model id")

    agent = RemoteAgent(
        vllm_url=args.url,
        model_name=model_name,
        allowed_tools={
            "referring_expression_segmentation",
            "semantic_segmentation",
        },
    )
    cases = (
        ("description", "Describe this remote sensing image."),
        ("buildings", "Extract all buildings in this image using RemoteSAM."),
    )
    output = {
        "url": args.url,
        "model": model_name,
        "models_response": models,
        "image": str(args.image.resolve()),
        "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "cases": {},
    }
    for name, query in cases:
        started = time.perf_counter()
        result = agent.plan(query, args.image)
        output["cases"][name] = {
            "query": query,
            "status": result.status,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "raw_response": result.raw_response,
            "text": result.text,
            "tool_call": (
                {"name": result.tool_call.name, "arguments": result.tool_call.arguments}
                if result.tool_call else None
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "remoteagent_verification.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))

    description = output["cases"]["description"]
    buildings = output["cases"]["buildings"]
    return 0 if (
        description["status"] == "completed" and description["text"]
        and buildings["status"] == "tool_call"
        and buildings["tool_call"]["name"] in {
            "referring_expression_segmentation",
            "semantic_segmentation",
        }
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
