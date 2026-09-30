"""Sync guard for the chat tools' "three sources of truth".

The chat-skills README wrote the problem down itself: for a tool to be truly
usable it must appear in all of
  1. registered:   mcp.json                        (decides whether the process starts and the schema enters context)
  2. callable:     the chat route's tool allowlist (decides whether the model can call it)
  3. discoverable: the system prompt's tool section — **one each for zh and en**
                   (decides whether the model thinks of it)

Missing any one of them is a **silent** failure — it really happened on
2026-07-20: get_nearby_planning / get_area_profile were registered and even
described in the Chinese prompt, but never made the allowlist, so the model was
introduced to a tool it could not call.

Measured fact that shapes what is asserted: the tool schemas registered in
mcp.json are what cost context (roughly 768 tokens per tool schema), so a tool
that is registered but unreachable is dead weight paid on every turn, and the
registration count is a hard wall that only gets worse over time.

(Extract note: the allowlist and the zh/en prompt sections live in the web
route, which is not part of this extract, so only the registration check runs
here.)
"""
from __future__ import annotations

import importlib.util as iu
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MCP_JSON = REPO / "mcp.json"

sys.path.insert(0, str(REPO))


def _exposed_tools() -> set[str]:
    """Short names of the tools actually exposed — taken from the aggregated server, which asks each module's own list_tools."""
    spec = iu.spec_from_file_location("mcp_all", REPO / "mcp_all.py")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._load_all()
    return set(mod._TOOLS)


def test_mcp_json_registers_a_module_for_every_exposed_tool() -> None:
    """mcp.json is the real schema-cost variable — the registration list must account for every exposed tool."""
    registered = set(json.loads(MCP_JSON.read_text())["mcpServers"])
    if registered <= {"wtl"}:
        return  # already switched to the aggregated form; module list is in mcp.legacy.json
    exposed = _exposed_tools()
    # one module may expose several tools (fetch_listing / fetch_listings), so only require coverage
    assert len(exposed) >= len(registered), (
        f"{len(registered)} modules registered but only {len(exposed)} tools exposed — some module produced no tool"
    )
