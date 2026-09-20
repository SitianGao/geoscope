from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """A parsed, validated request for an external RemoteAgent tool."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    text: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "completed"
    raw_response: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
