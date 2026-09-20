from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from remoteagent.config.defaults import DEFAULT_VLLM_URL, ENV_URL_KEYS
from remoteagent.config.tools_schema import MCP_TOOLS
from remoteagent.core.types import AgentResult, ToolCall
from remoteagent.llm.vllm import VLLMChatClient
from remoteagent.parsing import ToolCallParser
from remoteagent.services import ServiceExecutor
from remoteagent.utils.image import image_data_url
from remoteagent.utils.text import extract_answer_tag, load_system_prompt


class RemoteAgent:
    def __init__(
        self,
        vllm_url: str = DEFAULT_VLLM_URL,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        system_prompt_path: Optional[Union[str, Path]] = None,
        api_urls: Optional[Dict[str, Optional[str]]] = None,
        allowed_tools: Optional[Iterable[str]] = None,
        max_rounds: int = 3,
        max_tokens: int = 512,
    ) -> None:
        self.vllm_url = vllm_url.rstrip("/")
        self.model_name = model_name
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self._parser = ToolCallParser()
        self._allowed_tools = set(allowed_tools) if allowed_tools is not None else None
        unknown_tools = (self._allowed_tools or set()) - set(MCP_TOOLS)
        if unknown_tools:
            raise ValueError(f"Unknown allowed_tools: {sorted(unknown_tools)}")

        if system_prompt is not None:
            self._system_prompt = system_prompt.strip()
        else:
            p = Path(system_prompt_path) if system_prompt_path else self._default_prompt_path()
            self._system_prompt = load_system_prompt(p)

        self._api_urls = api_urls or self._api_urls_from_env()
        self._llm = VLLMChatClient(self.vllm_url)
        self._executor = ServiceExecutor(self._api_urls)

    @staticmethod
    def _api_urls_from_env() -> Dict[str, Optional[str]]:
        return {name: os.environ.get(key) for name, key in ENV_URL_KEYS.items()}

    @staticmethod
    def _default_prompt_path() -> Path:
        return Path(__file__).resolve().parents[1] / "prompts" / "prompt.txt"

    @classmethod
    def from_env(cls, **kwargs: Any) -> "RemoteAgent":
        vllm_url = kwargs.pop("vllm_url", os.environ.get("VLLM_URL", DEFAULT_VLLM_URL))
        model_name = kwargs.pop("model_name", os.environ.get("VLLM_MODEL"))
        return cls(vllm_url=vllm_url, model_name=model_name, **kwargs)

    def resolve_model_name(self) -> str:
        if self.model_name:
            return self.model_name
        discovered = VLLMChatClient.list_first_model_id(self.vllm_url)
        if discovered:
            return discovered
        return "RemoteAgent"

    def run(self, query: str, image_path: Union[str, Path]) -> AgentResult:
        """Run the original self-executing agent loop.

        GeoScope's Harness should normally use :meth:`plan` so it remains the
        sole owner of tool execution and result assets.
        """
        clean_path = self._normalize_image_path(image_path)
        return self._agent_loop(query, clean_path)

    def plan(
        self,
        query: str,
        image_path: Union[str, Path],
        *,
        context: Optional[Dict[str, Any]] = None,
        conversation: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentResult:
        """Perform one multimodal decision step without executing any tool.

        The result is either ``status='completed'`` with a direct answer,
        ``status='tool_call'`` with a validated :class:`ToolCall`, or a
        fail-closed status such as ``rejected_tool``/``unparsed``.
        """
        clean_path = self._normalize_image_path(image_path)
        messages = self._build_initial_messages(query, clean_path, context, conversation)
        response = self._run_llm(messages)
        result = self._interpret_response(response, clean_path)
        result.history.append(
            {
                "round": 1,
                "response": response,
                "status": result.status,
                "tool_call": self._tool_call_dict(result.tool_call),
            }
        )
        return result

    def continue_with_tool_result(
        self,
        query: str,
        image_path: Union[str, Path],
        decision: AgentResult,
        tool_result: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
        conversation: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentResult:
        """Return a user-facing answer after a Harness-owned tool execution."""
        if decision.status != "tool_call" or decision.tool_call is None:
            raise ValueError("decision must contain a pending tool_call")
        if not decision.raw_response:
            raise ValueError("decision is missing the original model response")

        clean_path = self._normalize_image_path(image_path)
        messages = self._build_initial_messages(query, clean_path, context, conversation)
        messages.append({"role": "assistant", "content": decision.raw_response})
        messages.append(
            {
                "role": "user",
                "content": self._tool_feedback_prompt(decision.tool_call, tool_result),
            }
        )
        response = self._run_llm(messages)
        result = self._interpret_response(response, clean_path)
        result.tool_results.append(tool_result)
        result.history.extend(decision.history)
        result.history.append(
            {
                "round": len(result.history) + 1,
                "response": response,
                "status": result.status,
                "tool_call": self._tool_call_dict(result.tool_call),
                "tool_result_status": tool_result.get("status"),
            }
        )
        return result

    def _run_llm(self, messages: List[Dict[str, Any]]) -> str:
        return self._llm.chat(messages, model=self.resolve_model_name(), max_tokens=self.max_tokens)

    @staticmethod
    def _normalize_image_path(image_path: Union[str, Path]) -> str:
        clean = str(image_path).strip().strip("'").strip('"')
        path = Path(clean).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return str(path.resolve())

    def _runtime_system_prompt(self) -> str:
        if self._allowed_tools is None:
            return self._system_prompt
        names = ", ".join(sorted(self._allowed_tools)) or "none"
        return (
            self._system_prompt
            + "\n\nRuntime tool policy: only these deployed tools are available: "
            + names
            + ". Do not call any other tool."
        )

    def _build_initial_messages(
        self,
        user_query: str,
        image_path: str,
        context: Optional[Dict[str, Any]] = None,
        conversation: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        context_text = json.dumps(context or {}, ensure_ascii=False, default=str)
        user_text = (
            f"Image asset path (for tool arguments): {image_path}\n"
            f"Current experiment context: {context_text}\n\n"
            f"User request: {user_query}"
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._runtime_system_prompt()},
        ]
        for item in conversation or []:
            if item.get("role") in ("user", "assistant") and isinstance(item.get("content"), str):
                messages.append({"role": item["role"], "content": item["content"]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                ],
            }
        )
        return messages

    @staticmethod
    def _tool_call_dict(tool_call: Optional[ToolCall]) -> Optional[Dict[str, Any]]:
        if tool_call is None:
            return None
        return {"name": tool_call.name, "arguments": tool_call.arguments}

    def _interpret_response(self, response: str, image_path: str) -> AgentResult:
        answer_content = extract_answer_tag(response)
        if answer_content is not None:
            return AgentResult(
                text=answer_content,
                status="completed",
                raw_response=response,
            )

        parsed = self._parser.parse_tool_call_with_image_fallback(response, Path(image_path))
        if parsed:
            tool_name, raw_args = parsed
            args = dict(raw_args)
            if "image_path" in MCP_TOOLS.get(tool_name, []):
                # The model may never redirect the Harness to an arbitrary local path.
                args["image_path"] = image_path
            tool_call = ToolCall(name=tool_name, arguments=args)
            if self._allowed_tools is not None and tool_name not in self._allowed_tools:
                return AgentResult(
                    text=f"Rejected unavailable tool: {tool_name}",
                    status="rejected_tool",
                    raw_response=response,
                    tool_call=tool_call,
                )
            return AgentResult(
                text="",
                status="tool_call",
                raw_response=response,
                tool_call=tool_call,
            )

        return AgentResult(
            text=response,
            status="unparsed",
            raw_response=response,
        )

    @staticmethod
    def _redact_large_artifacts(value: Any) -> Any:
        if isinstance(value, dict):
            cleaned: Dict[str, Any] = {}
            for key, child in value.items():
                if key.lower() in {"mask", "image", "overlay"} and isinstance(child, str):
                    cleaned[key] = f"<artifact omitted; {len(child)} characters>"
                else:
                    cleaned[key] = RemoteAgent._redact_large_artifacts(child)
            return cleaned
        if isinstance(value, list):
            return [RemoteAgent._redact_large_artifacts(item) for item in value]
        if isinstance(value, str) and len(value) > 2048:
            return f"<large value omitted; {len(value)} characters>"
        return value

    @classmethod
    def _tool_feedback_prompt(cls, tool_call: ToolCall, tool_result: Dict[str, Any]) -> str:
        safe_result = cls._redact_large_artifacts(tool_result)
        return (
            f"[Tool Call] {tool_call.name}"
            f"({json.dumps(tool_call.arguments, ensure_ascii=False, default=str)})\n"
            f"[Execution Result]\n{json.dumps(safe_result, ensure_ascii=False, default=str)}\n\n"
            "Provide a concise final response based only on this execution result. "
            "Do not invent measurements. If execution failed, state that it failed. "
            "Call another tool only if the request is still genuinely incomplete."
        )

    def _agent_loop(self, user_query: str, image_path: str) -> AgentResult:
        messages = self._build_initial_messages(user_query, image_path)

        history: List[Dict[str, Any]] = []
        tool_results: List[Dict[str, Any]] = []
        final_text = ""
        final_status = "max_rounds"
        raw_response: Optional[str] = None
        pending_tool_call: Optional[ToolCall] = None

        for round_idx in range(self.max_rounds):
            response = self._run_llm(messages)
            raw_response = response
            decision = self._interpret_response(response, image_path)
            record: Dict[str, Any] = {
                "round": round_idx + 1,
                "response": response,
                "status": decision.status,
                "tool_call": self._tool_call_dict(decision.tool_call),
            }

            if decision.status in ("completed", "unparsed", "rejected_tool"):
                final_text = decision.text
                final_status = decision.status
                pending_tool_call = decision.tool_call
                history.append(record)
                break

            if decision.tool_call is None:
                final_text = "RemoteAgent produced no executable decision."
                final_status = "unparsed"
                history.append(record)
                break

            pending_tool_call = decision.tool_call
            result = self._executor.execute_structured(
                pending_tool_call.name, pending_tool_call.arguments
            )
            tool_results.append(result)
            record["tool_result_status"] = result.get("status")
            record["tool_result_message"] = result.get("message")
            history.append(record)
            final_text = str(result.get("message") or "")
            final_status = "tool_failed" if result.get("status") != "success" else "tool_completed"
            messages.append({"role": "assistant", "content": response})
            messages.append(
                {
                    "role": "user",
                    "content": self._tool_feedback_prompt(pending_tool_call, result),
                }
            )

        return AgentResult(
            text=final_text,
            history=history,
            status=final_status,
            raw_response=raw_response,
            tool_call=pending_tool_call,
            tool_results=tool_results,
        )
