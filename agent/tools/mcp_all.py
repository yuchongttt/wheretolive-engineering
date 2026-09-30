#!/usr/bin/env python3
"""Aggregated MCP server: one process hosts every chat-skills tool
(2026-07-13, latency optimisation #1).

Motivation: every chat turn, the agent runtime launched 24 separate Python MCP
server processes; 24 interpreter cold starts + handshakes were one of the
biggest contributors to time-to-first-byte (p50 ≈ 22s). This server imports
every module in ONE process and exposes all tools through ONE handshake.

Three safeguards (design constraints — do not remove):
1. Per-module isolated import — one broken module only loses its own tools
   (and logs why); it never takes the whole server down.
2. asyncio.to_thread dispatch — the module handlers have async signatures but
   blocking internals (sqlite / HTTP). Awaiting them directly on the event loop
   would serialise the model's parallel tool calls; each request instead runs
   in a worker thread with its own asyncio.run, restoring real concurrency
   (each tool opens its own DB connection, so this is thread-safe).
3. The module list is parsed from mcp.json — it stays in sync with the
   per-tool config by construction; there is no second registry to maintain.
   (If mcp.json has been switched to contain only this server, fall back to
   the module list in mcp.legacy.json.)

Tools are exposed as mcp__<this server's name>__<short name>; short names are
identical to the original per-tool servers, so prompts that reference tools by
short name are unaffected. The chat route's tool allowlist must switch prefix.
"""
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s mcp_all %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_SELF_NAMES = {"mcp_all", "wtl"}


def _module_names() -> list[str]:
    """Parse the module list from mcp.json (before the switch) or mcp.legacy.json (after)."""
    for fname in ("mcp.json", "mcp.legacy.json"):
        p = HERE / fname
        if not p.exists():
            continue
        try:
            servers = json.loads(p.read_text()).get("mcpServers", {})
        except (ValueError, OSError) as e:
            logger.error(f"parse {fname} failed: {e}")
            continue
        mods = []
        for key, spec in servers.items():
            if key in _SELF_NAMES:
                continue
            args = spec.get("args") or []
            if not args:
                continue
            mods.append(Path(args[-1]).stem)  # …/<module>.py → <module>
        if mods:
            logger.info(f"module list from {fname}: {len(mods)} modules")
            return mods
    logger.error("no module list found (mcp.json / mcp.legacy.json)")
    return []


# tool short name → (module name, that module's call handler, original Tool definition)
_TOOLS: dict[str, tuple[str, Any, types.Tool]] = {}
_FAILED: dict[str, str] = {}


def _load_all() -> None:
    # Must be called BEFORE the event loop starts: below we use asyncio.run()
    # to fetch each module's tool list, and calling that inside a running loop
    # always raises RuntimeError — which this function's per-module try/except
    # would swallow as an "isolated single-module failure". The result: all 27
    # modules fail, and the server silently exposes 0 tools (the agent still
    # answers fluently, just with no data behind it). This exact bug sat in the
    # repo for a month (introduced 2026-07-13 → found 2026-08-18), because the
    # original gate only called _load_all() from a synchronous context instead
    # of going through the real entrypoint. The guard lives here: better to
    # refuse to start than to start empty.
    import importlib
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # no running loop — the correct call site
    else:
        raise RuntimeError(
            "_load_all() was called inside a running event loop — it must run before "
            "asyncio.run(), otherwise every module silently fails to load. "
            "See tests/test_mcp_all_entrypoint.py"
        )
    for mod_name in _module_names():
        try:
            mod = importlib.import_module(mod_name)
            lt = getattr(mod, "list_tools", None) or getattr(mod, "handle_list_tools", None)
            ct = getattr(mod, "call_tool", None) or getattr(mod, "handle_call_tool", None)
            if lt is None or ct is None:
                raise AttributeError("missing list_tools/call_tool by name")
            tools = asyncio.run(lt())
            for t in tools:
                if t.name in _TOOLS:
                    logger.warning(f"tool name collision: {t.name} ({mod_name} vs {_TOOLS[t.name][0]}) — first wins")
                    continue
                _TOOLS[t.name] = (mod_name, ct, t)
        except Exception as e:  # isolation: one module failing only loses its own tools
            _FAILED[mod_name] = f"{type(e).__name__}: {e}"
            logger.exception(f"module {mod_name} failed to load — its tools are OFFLINE")
    logger.info(f"loaded {len(_TOOLS)} tools from {len(set(m for m, _, _ in _TOOLS.values()))} modules"
                + (f"; FAILED: {list(_FAILED)}" if _FAILED else ""))


def _run_handler_sync(handler: Any, name: str, arguments: dict) -> Any:
    """Run a (possibly async) handler inside a worker thread — fresh event loop, no interference."""
    result = handler(name, arguments)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


server = Server("wtl")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [t for _, _, t in _TOOLS.values()]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    entry = _TOOLS.get(name)
    if entry is None:
        hint = f"(module failed: {_FAILED})" if _FAILED else ""
        return [types.TextContent(type="text", text=json.dumps(
            {"error": f"unknown tool: {name} {hint}"}, ensure_ascii=False))]
    _, handler, _ = entry
    # to_thread: a blocking handler yields the event loop → parallel tool_use
    # calls run truly concurrently (safeguard #2)
    return await asyncio.to_thread(_run_handler_sync, handler, name, arguments or {})


async def main():
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(read, write, InitializationOptions(
            server_name="wtl", server_version="1.0.0",
            capabilities=server.get_capabilities(
                notification_options=NotificationOptions(), experimental_capabilities={})))


if __name__ == "__main__":
    _load_all()          # before the loop starts — see the guard note at the top of _load_all()
    asyncio.run(main())
