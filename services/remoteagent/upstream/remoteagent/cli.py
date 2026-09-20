"""CLI inference entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from remoteagent.config.defaults import DEFAULT_VLLM_URL, ENV_URL_KEYS, SERVICE_PORTS
from remoteagent.core import RemoteAgent


class RemoteAgentCLI:
    @staticmethod
    def _default_service_url(service: str) -> str:
        env_key = ENV_URL_KEYS[service]
        return os.environ.get(env_key, f"http://127.0.0.1:{SERVICE_PORTS[service]}")

    @staticmethod
    def build_arg_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="RemoteAgent: vLLM + EO vision HTTP tools")
        p.add_argument("--vllm_url", type=str, default=os.environ.get("VLLM_URL", DEFAULT_VLLM_URL))
        p.add_argument("--vllm_model", type=str, default=os.environ.get("VLLM_MODEL"))
        p.add_argument(
            "--system_prompt_path",
            type=str,
            default=None,
            help="System prompt file (default: bundled remoteagent/prompts/prompt.txt)",
        )
        p.add_argument(
            "--remote_sam_url",
            type=str,
            default=RemoteAgentCLI._default_service_url("remotesam"),
        )
        p.add_argument(
            "--change3d_url",
            type=str,
            default=RemoteAgentCLI._default_service_url("change3d"),
        )
        p.add_argument(
            "--sm3det_url",
            type=str,
            default=RemoteAgentCLI._default_service_url("sm3det"),
        )
        p.add_argument(
            "--crossearth_url",
            type=str,
            default=RemoteAgentCLI._default_service_url("crossearth"),
        )
        p.add_argument(
            "--skysense_det_url",
            type=str,
            default=RemoteAgentCLI._default_service_url("skysense_det"),
        )
        p.add_argument(
            "--directsam_url",
            type=str,
            default=RemoteAgentCLI._default_service_url("directsam"),
        )
        p.add_argument("--image_path", type=str, required=True)
        p.add_argument("--max_rounds", type=int, default=3)
        p.add_argument(
            "--plan_only",
            action="store_true",
            help="Return one official RemoteAgent decision without executing a tool.",
        )
        p.add_argument(
            "--context_json",
            type=str,
            default="{}",
            help="Current Harness context as a JSON object (used with --plan_only).",
        )
        p.add_argument(
            "--allowed_tools",
            type=str,
            default=None,
            help="Comma-separated deployed tool names; unavailable calls are rejected.",
        )
        p.add_argument("query", type=str, nargs="*", help="User query")
        return p

    @staticmethod
    def main(argv: list[str] | None = None) -> int:
        args = RemoteAgentCLI.build_arg_parser().parse_args(argv)
        query = " ".join(args.query).strip() if args.query else "Please describe this image."
        prompt_path = Path(args.system_prompt_path) if args.system_prompt_path else None

        api_urls = {
            "remotesam": args.remote_sam_url,
            "change3d": args.change3d_url,
            "sm3det": args.sm3det_url,
            "crossearth": args.crossearth_url,
            "skysense_det": args.skysense_det_url,
            "directsam": args.directsam_url,
        }
        allowed_tools = None
        if args.allowed_tools is not None:
            allowed_tools = {name.strip() for name in args.allowed_tools.split(",") if name.strip()}

        agent = RemoteAgent(
            vllm_url=args.vllm_url,
            model_name=args.vllm_model,
            system_prompt_path=prompt_path,
            api_urls=api_urls,
            allowed_tools=allowed_tools,
            max_rounds=args.max_rounds,
        )

        print("=" * 60)
        print("RemoteAgent")
        print("=" * 60)
        print(f"  vLLM: {args.vllm_url} | model: {agent.resolve_model_name()}")
        for k, v in api_urls.items():
            print(f"  {k}: {v or '(not set)'}")
        print("-" * 60)
        print(f"  image: {args.image_path}")
        print(f"  query: {query[:100]}{'...' if len(query) > 100 else ''}")
        print("-" * 60)

        if args.plan_only:
            try:
                context = json.loads(args.context_json)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid --context_json: {exc}") from exc
            if not isinstance(context, dict):
                raise SystemExit("--context_json must decode to a JSON object")
            result = agent.plan(query=query, image_path=args.image_path, context=context)
            output = {
                "status": result.status,
                "text": result.text,
                "raw_response": result.raw_response,
                "tool_call": (
                    {
                        "name": result.tool_call.name,
                        "arguments": result.tool_call.arguments,
                    }
                    if result.tool_call
                    else None
                ),
            }
            print("\n[Decision]\n", json.dumps(output, ensure_ascii=False, indent=2), sep="")
        else:
            result = agent.run(query=query, image_path=args.image_path)
            print("\n[Final response]\n", result.text, sep="")
        print("=" * 60)
        return 0


build_arg_parser = RemoteAgentCLI.build_arg_parser
main = RemoteAgentCLI.main


__all__ = ["build_arg_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
