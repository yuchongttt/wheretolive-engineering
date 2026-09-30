#!/usr/bin/env python3
"""Direct tool invoker for the Qwen3.6-27B chat backend.

Bypasses MCP stdio protocol by importing each MCP server's handler function
and calling it directly. Used by web/src/app/api/chat/route.ts when the
Qwen model emits a tool_call — much faster than spawning a fresh MCP server
per call (avoids ~1-2s handshake per tool).

Usage:
    python3 tool_runner.py <tool_name> <args_json>
Output (stdout): single-line JSON with the tool's text response.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path

SKILLS = Path(__file__).resolve().parent
sys.path.insert(0, str(SKILLS))


# Map external tool name → (module_name, internal_handler_name)
# Most MCP servers in chat-skills/ have `handle_call_tool` or `call_tool`.
# (Extract note: the production map also routes search_properties,
# geocode_address, lookup_area_by_landmark, analyze_floorplan,
# get_area_profile and create_radar; those modules are not part of this
# extract, so their entries are omitted here.)
TOOL_MAP: dict[str, tuple[str, str]] = {
    "get_postcode_scores":     ("get_postcode_scores",     "handle_call_tool"),
    "get_area_overview":       ("get_area_overview",       "handle_call_tool"),
    "get_comparables":         ("get_comparables",         "handle_call_tool"),
    "get_sold_nearby":         ("get_sold_nearby",         "handle_call_tool"),
}


async def invoke(tool_name: str, args: dict) -> str:
    if tool_name not in TOOL_MAP:
        return json.dumps({"error": f"unknown tool: {tool_name}"})

    module_name, handler_name = TOOL_MAP[tool_name]
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        return json.dumps({"error": f"import failed: {e}"})

    handler = getattr(mod, handler_name, None)
    if handler is None:
        # Some modules register the handler via @server.call_tool() decorator
        # → it's stored inside the server. Try probing the module's `server` object.
        server = getattr(mod, "server", None)
        if server is not None:
            # MCP server stores call_tool in _tool_handlers (private API but stable)
            handlers = getattr(server, "_tool_handlers", None) or getattr(server, "request_handlers", None)
            if isinstance(handlers, dict):
                # Find a CallToolRequest handler
                for key, fn in handlers.items():
                    if "CallTool" in str(key):
                        handler = fn
                        break

    if handler is None:
        return json.dumps({"error": f"no handler found in {module_name}"})

    try:
        # MCP handlers are async, take (name, arguments) → list[TextContent]
        result = handler(tool_name, args)
        if asyncio.iscoroutine(result):
            result = await result
    except Exception as e:
        return json.dumps({"error": f"handler raised: {type(e).__name__}: {str(e)[:300]}"})

    # MCP handlers return list[TextContent]; we want the JSON text.
    if isinstance(result, list):
        texts = []
        for item in result:
            txt = getattr(item, "text", None)
            if txt is not None:
                texts.append(txt)
        return "\n".join(texts) if texts else json.dumps({"error": "empty result"})
    if isinstance(result, str):
        return result
    return json.dumps(result)


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: tool_runner.py <tool> [args_json]"}), file=sys.stderr)
        return 2
    tool_name = sys.argv[1]
    args_raw = sys.argv[2] if len(sys.argv) > 2 else "{}"
    try:
        args = json.loads(args_raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"bad args JSON: {e}"}))
        return 1
    out = asyncio.run(invoke(tool_name, args))
    sys.stdout.write(out)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
