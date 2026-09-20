"""Exercise the GeoScope Harness against a real RemoteSAM service.

The Agent decision/final wording are deterministic stubs because this probe is
for the cross-machine tool boundary.  Official RemoteAgent parsing and tool
feedback are covered separately by unit tests and require a running model server.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--url", default="http://10.199.156.100:6657/predict")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--query", default="提取右侧建筑并计算面积占比")
    parser.add_argument("--prompt", default="all buildings")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    server.DATA = args.output_dir
    server.SESSIONS.clear()
    uploaded = base64.b64encode(args.image.read_bytes()).decode("ascii")
    session = server.SESSIONS[
        server.new_session(uploaded=uploaded, name=args.image.name)["id"]
    ]
    decision = SimpleNamespace(
        status="tool_call",
        text="",
        tool_call=SimpleNamespace(
            name="referring_expression_segmentation",
            arguments={"image_path": "ignored-by-binding", "prompt": args.prompt},
        ),
        history=[{"round": 1}],
    )
    final = SimpleNamespace(
        status="completed",
        text="已依据 RemoteSAM 返回和工作台统计完成。",
        tool_call=None,
        history=[{"round": 1}, {"round": 2}],
    )
    plan = {
        "action": "segment",
        "target": "building",
        "side": "right",
        "tool": "referring_expression_segmentation",
        "prompt": args.prompt,
    }
    env = {
        "GEO_REMOTESAM_URL": args.url,
        "GEO_REMOTESAM_REVISION": "live-contract-probe",
        "GEO_AGENT_MODEL": "RemoteAgent-contract-probe",
    }
    with (
        patch.object(server, "live_plan", return_value=(plan, object(), decision)),
        patch.object(server, "finish_with_agent", return_value=final),
        patch.dict(os.environ, env),
    ):
        result = server.run_task(session, {"query": args.query, "mode": "live"})

    compact = {
        "status": result.get("status"),
        "message": result.get("message"),
        "duration_ms": result.get("duration_ms"),
        "task": result.get("task"),
        "metrics": result.get("metrics"),
        "agent": result.get("agent"),
        "full_mask_sha256": result.get("full_mask_sha256"),
        "run_id": result.get("id"),
        "session_id": result.get("session_id"),
        "remote_url": args.url,
        "agent_inference": "stubbed for contract probe; official plan/continue covered by tests",
    }
    (args.output_dir / "workbench_verification.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    if result.get("status") not in ("completed", "needs_review") or not result.get("metrics"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
