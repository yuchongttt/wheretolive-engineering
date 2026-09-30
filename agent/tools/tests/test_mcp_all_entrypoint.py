"""Smoke test of the aggregated mcp_all.py server through its REAL entrypoint.

Why this must go through a stdio subprocess instead of importing and calling
`_load_all()` directly: when mcp_all.py was introduced (2026-07-13) the local
gate tested the `_load_all()` function (called from a synchronous context → all
28 tools green), while the real entrypoint `asyncio.run(main())` ran
`_load_all()` on an already-running event loop, where its inner
`asyncio.run(lt())` always raises RuntimeError — every one of the 27 modules
failed to load and the server exposed 0 tools. "The function works" ≠ "the
process starts", which is why the bug sat in the repo for a month (found
2026-08-18).

So this test may ONLY start the process from outside, do a real handshake and
count real tools. Any rewrite to "import it and assert" would make it
meaningless again.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SERVER = REPO / "mcp_all.py"
MCP_JSON = REPO / "mcp.json"

# One handshake + tools/list measured at ~0.4s; allow plenty of margin, but don't wait forever.
TIMEOUT_S = 60


def _handshake() -> tuple[list[dict], str]:
    """Start the real process, do initialize → initialized → tools/list; return (tools, stderr)."""
    proc = subprocess.Popen(
        [sys.executable, str(SERVER)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=str(REPO),
    )
    try:
        def send(obj: dict) -> None:
            assert proc.stdin is not None
            proc.stdin.write(json.dumps(obj) + "\n")
            proc.stdin.flush()

        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "1"}}})
        assert proc.stdout is not None
        proc.stdout.readline()  # initialize result
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        line = proc.stdout.readline()
        tools = json.loads(line).get("result", {}).get("tools", [])
    finally:
        proc.kill()
        try:
            _, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            err = ""
    return tools, err or ""


def test_aggregated_server_exposes_tools_through_real_entrypoint() -> None:
    """After a real start the server must expose tools — 0 tools is the month-long bug recurring."""
    tools, err = _handshake()
    assert tools, (
        "the aggregated server exposed 0 tools through the real stdio entrypoint — "
        "most likely _load_all() is running inside a live event loop again. stderr:\n" + err[:2000]
    )


def test_no_module_fails_to_load() -> None:
    """Per-module isolated import is a design constraint, but 'isolated' doesn't mean 'allowed to stay broken'."""
    _, err = _handshake()
    assert "failed to load" not in err, (
        "a module failed to load and its tools are OFFLINE. stderr:\n" + err[:2000]
    )


def test_tool_count_matches_registered_modules() -> None:
    """The aggregate must expose at least one tool per module registered in mcp.json."""
    tools, err = _handshake()
    registered = json.loads(MCP_JSON.read_text())["mcpServers"]
    # After the switch mcp.json may contain only the aggregate itself; then this doesn't apply.
    modules = [k for k in registered if k not in {"wtl", "mcp_all"}]
    if not modules:
        pytest.skip("mcp.json already switched to the aggregated form; module list is in mcp.legacy.json")
    assert len(tools) >= len(modules), (
        f"{len(modules)} modules registered but only {len(tools)} tools exposed. stderr:\n" + err[:1500]
    )
