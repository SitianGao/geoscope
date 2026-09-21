"""Small dependency-free OpenAI-compatible client for vLLM servers."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class VLLMChatClient:
    def __init__(self, base_url: str, api_key: str = "dummy") -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = 120.0

    @staticmethod
    def _opener():
        mode = os.environ.get("GEO_SERVICE_PROXY_MODE", "direct")
        if mode not in ("direct", "environment"):
            raise ValueError("GEO_SERVICE_PROXY_MODE must be direct or environment")
        handler = urllib.request.ProxyHandler({}) if mode == "direct" else urllib.request.ProxyHandler()
        return urllib.request.build_opener(handler)

    def _request(self, path: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 120.0) -> Dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            f"{self._base_url}/v1/{path.lstrip('/')}", data=data, headers=headers
        )
        try:
            with self._opener().open(request, timeout=timeout) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError("vLLM response is too large")
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError("vLLM must return a JSON object")
            return result
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
            raise ValueError(f"vLLM returned HTTP {code}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ValueError("vLLM is unreachable or timed out") from None
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, TypeError):
            raise ValueError("vLLM returned an invalid response") from None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: int = 512,
    ) -> str:
        result = self._request(
            "chat/completions",
            {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0},
            timeout=self._timeout,
        )
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ValueError("vLLM response is missing choices[0].message.content") from None
        if not isinstance(content, str):
            raise ValueError("vLLM response content must be text")
        return content.strip()

    @staticmethod
    def list_first_model_id(vllm_url: str, timeout: float = 10.0) -> Optional[str]:
        try:
            client = VLLMChatClient(vllm_url)
            data = client._request("models", timeout=timeout)
            models = data.get("data") or []
            if isinstance(models, list) and models:
                return models[0].get("id")
        except Exception:
            pass
        return None
