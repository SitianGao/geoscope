"""Official RemoteAgent adapter for the GeoScope workbench Harness.

RemoteAgent chooses an action.  The workbench remains the only component allowed
to execute RemoteSAM, validate artifacts, apply deterministic geometry, and save
experiment records.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "services" / "remoteagent" / "upstream"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from remoteagent import RemoteAgent  # noqa: E402


ALLOWED_TOOLS = {"referring_expression_segmentation"}
TARGET_PATTERNS = (
    ("building", r"建筑|房屋|楼|building"),
    ("aircraft", r"飞机|aircraft|airplane"),
    ("road", r"道路|公路|road"),
    ("water", r"水体|河流|湖泊|water"),
    ("tree", r"植被|树木|森林|tree|vegetation"),
    ("ship", r"船舶|船|ship"),
)
SIDE_PATTERNS = (
    ("left", r"左|left"),
    ("right", r"右|right"),
    ("top", r"上半|上方|top"),
    ("bottom", r"下半|下方|bottom"),
    ("all", r"全图|全部|整幅|all|entire"),
)


def _vllm_root(value: str) -> str:
    """RemoteAgent's client appends /v1, while the UI accepts either form."""
    root = value.strip().rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def create_agent() -> RemoteAgent:
    base = os.environ.get("GEO_AGENT_BASE_URL", "").strip()
    model = os.environ.get("GEO_AGENT_MODEL", "").strip()
    if not base or not model:
        raise ValueError("RemoteAgent 服务未配置。请填写认知服务地址和模型名称。")
    agent = RemoteAgent(
        vllm_url=_vllm_root(base),
        model_name=model,
        allowed_tools=ALLOWED_TOOLS,
        max_rounds=2,
        max_tokens=650,
    )
    api_key = os.environ.get("GEO_AGENT_API_KEY", "").strip()
    if api_key:
        # VLLMChatClient intentionally defers SDK construction; setting the key
        # here keeps the upstream public API unchanged and avoids leaking it.
        agent._llm._api_key = api_key
    return agent


def plan_with_agent(
    query: str,
    image_path: Path,
    context: Dict[str, Any],
    agent: Optional[RemoteAgent] = None,
) -> Tuple[RemoteAgent, Any]:
    agent = agent or create_agent()
    decision = agent.plan(
        query,
        image_path,
        context={
            "previous_result": context,
            "supported_tool": "referring_expression_segmentation",
            "supported_targets": ["building", "aircraft", "road", "water", "tree", "ship"],
            "supported_scopes": ["all", "left", "right", "top", "bottom"],
            "harness_policy": "Choose the tool only. The Harness executes it and computes measurements.",
        },
    )
    return agent, decision


def resolve_scope(query: str, context: Dict[str, Any], tool_arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize target/scope deterministically after Agent selected the tool."""
    prompt = str(tool_arguments.get("prompt") or "").strip()
    combined = f"{query}\n{prompt}".lower()
    target = next((name for name, pattern in TARGET_PATTERNS if re.search(pattern, combined)), None)
    side = next((name for name, pattern in SIDE_PATTERNS if re.search(pattern, query.lower())), None)
    target = target or context.get("target")
    side = side or context.get("side", "all")
    if not target:
        raise ValueError("RemoteAgent 已选择分割工具，但没有给出工作台支持的目标类别；请明确建筑、飞机、道路、水体、植被或船舶。")
    if side not in {"all", "left", "right", "top", "bottom"}:
        raise ValueError("RemoteAgent 返回了工作台不支持的空间范围。")
    if not prompt:
        raise ValueError("RemoteAgent 的分割工具调用缺少文本提示。")
    return {
        "action": "segment",
        "target": target,
        "side": side,
        "tool": "referring_expression_segmentation",
        "prompt": prompt,
        "inherited_target": target == context.get("target"),
        "reuse": False,
    }


def decision_to_plan(query: str, context: Dict[str, Any], decision: Any) -> Dict[str, Any]:
    if decision.status == "completed":
        return {"action": "scene", "answer": decision.text, "agent_status": decision.status}
    if decision.status == "tool_call" and decision.tool_call is not None:
        if decision.tool_call.name not in ALLOWED_TOOLS:
            raise ValueError(f"RemoteAgent 请求了未部署工具：{decision.tool_call.name}")
        plan = resolve_scope(query, context, decision.tool_call.arguments)
        plan["agent_status"] = decision.status
        return plan
    if decision.status == "rejected_tool":
        return {
            "action": "clarify",
            "question": "RemoteAgent 请求了当前比赛工作流未部署的工具，本次未执行。请改为目标分割任务。",
            "agent_status": decision.status,
        }
    if decision.status == "unparsed" and isinstance(decision.raw_response, str):
        # Transitional compatibility for teammate B's already-merged fake
        # workbench service.  Production RemoteAgent should emit T_call or
        # <answer>; this path accepts only the old, closed schema.
        try:
            legacy = json.loads(decision.raw_response)
        except (json.JSONDecodeError, TypeError):
            legacy = None
        if isinstance(legacy, dict) and legacy.get("action") == "segment":
            if legacy.get("target") in {name for name, _ in TARGET_PATTERNS} and legacy.get("side") in {
                "all", "left", "right", "top", "bottom"
            }:
                return {
                    "action": "segment",
                    "target": legacy["target"],
                    "side": legacy["side"],
                    "agent_status": "legacy_json_compat",
                    "legacy_compat": True,
                }
    return {
        "action": "clarify",
        "question": "RemoteAgent 没有返回可验证的答案或工具调用，本次未执行。请更明确地描述目标和范围。",
        "agent_status": decision.status,
    }


def public_decision(decision: Any) -> Dict[str, Any]:
    """Audit metadata without model chain-of-thought or large prompt contents."""
    tool = None
    if decision.tool_call is not None:
        tool = {
            "name": decision.tool_call.name,
            "arguments": {
                key: ("<bound current image>" if key == "image_path" else value)
                for key, value in decision.tool_call.arguments.items()
            },
        }
    return {"status": decision.status, "tool_call": tool, "rounds": len(decision.history)}


def finish_with_agent(
    agent: RemoteAgent,
    query: str,
    image_path: Path,
    context: Dict[str, Any],
    decision: Any,
    tool_result: Dict[str, Any],
) -> Any:
    return agent.continue_with_tool_result(
        query,
        image_path,
        decision,
        tool_result,
        context={"previous_result": context, "harness_policy": "Measurements below are authoritative."},
    )
