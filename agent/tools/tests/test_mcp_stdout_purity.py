"""stdout purity: a single byte written to stdout by any registered module
corrupts the whole JSON-RPC stream.

With 27 separate servers, one module's stray output only broke its own tool;
**once merged into one aggregated server, it breaks all 28 tools** — the
agent still answers fluently, just with no data behind it. This is the one
fatal failure mode of the aggregation that can only happen "later", and the
only one a test can pin down — so pin it.

The chat-skills README convention already said "stderr for logs, stdout for
JSON-RPC"; this turns the convention into a gate with three assertions:
  1. Static — no bare print() in registered modules; the easiest mistake to
     make and the easiest to block.
  2. Runtime, import — really spawn a subprocess that imports every registered
     module and assert stdout stays empty. Catches what a static scan can't
     (a third-party library greeting at import time, a warning routed to
     stdout).
  3. Runtime, protocol — really start the aggregated process, handshake, make
     a few real tool calls and validate JSON-RPC line by line. Closest to
     production: that is exactly how the agent runtime reads it.

Each of the three was verified to go red by injecting a print and reverting —
dropping any one of them leaves a blind spot.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SKILLS = REPO
MCP_JSON = SKILLS / "mcp.json"
AGG = SKILLS / "mcp_all.py"


def _registered_modules() -> list[str]:
    regs = json.loads(MCP_JSON.read_text())["mcpServers"]
    return [k for k in regs if k not in {"wtl", "mcp_all"}]


def test_no_registered_module_prints_to_stdout() -> None:
    """Static layer: a bare print() is the easiest mistake to make and to block."""
    offenders: list[str] = []
    for mod in _registered_modules():
        path = SKILLS / f"{mod}.py"
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.match(r"^\s*print\(", line) and "stderr" not in line:
                offenders.append(f"{mod}.py:{i}")
    assert not offenders, (
        "these registered modules print to stdout, which would corrupt the whole "
        "JSON-RPC stream once aggregated into one process: " + ", ".join(offenders))


def test_importing_every_registered_module_keeps_stdout_silent() -> None:
    """Runtime layer 1: stray output at import time (libraries greeting, warnings on stdout)."""
    mods = _registered_modules()
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import importlib\n"
        "for m in %r:\n"
        "    importlib.import_module(m)\n"
    ) % (str(SKILLS), mods)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=str(REPO), timeout=180)
    assert r.returncode == 0, f"importing the registered modules failed:\n{r.stderr[-2000:]}"
    assert r.stdout == "", (
        f"stdout not clean while importing {len(mods)} registered modules:\n{r.stdout[:1000]!r}")


def test_aggregated_server_emits_only_valid_jsonrpc_on_stdout() -> None:
    """Runtime layer 2: start the aggregated process for real, handshake, make a
    few real tool calls, and validate every stdout line as JSON-RPC.

    This is the layer closest to production — it is exactly how the agent
    runtime reads the stream.
    """
    # PYTHONUNBUFFERED: when stdout is a pipe it is BLOCK-buffered by default,
    # so stray import-time output would only be flushed at process exit — and
    # this layer would miss it (verified: injecting an import-time print turned
    # layers 1 and 2 red but left this one green). In production such output
    # would suddenly land in the middle of some response once the buffer
    # filled, so it is a real danger, just hard to see. Unbuffering makes it
    # show up immediately.
    proc = subprocess.Popen(
        [sys.executable, str(AGG)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=str(REPO),
        env={**os.environ, "PYTHONUNBUFFERED": "1"})
    lines: list[str] = []
    try:
        def send(obj: dict) -> None:
            assert proc.stdin is not None
            proc.stdin.write(json.dumps(obj) + "\n")
            proc.stdin.flush()

        assert proc.stdout is not None
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "1"}}})
        lines.append(proc.stdout.readline())
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        # A few real calls of different shapes: plain sqlite, multi-area, aggregate read.
        # (Extract note: the original multi-area call was search_properties, which is
        # not part of this extract; compare_postcodes stands in for it. Without a
        # database the calls return JSON-RPC error results — still valid JSON-RPC,
        # which is all this layer asserts.)
        calls = [
            (2, "get_postcode_scores", {"postcode": "N1 8DE"}),
            (3, "compare_postcodes", {"postcodes": ["N1", "N5"]}),
            (4, "get_area_overview", {"postcode": "N1 8DE"}),
        ]
        for cid, name, args in calls:
            send({"jsonrpc": "2.0", "id": cid, "method": "tools/call",
                  "params": {"name": name, "arguments": args}})
            lines.append(proc.stdout.readline())
    finally:
        proc.kill()

    for n, raw in enumerate(lines):
        assert raw.strip(), f"line {n} is empty — did the server die early?"
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            pytest.fail(f"stdout line {n} is not JSON (the stream is polluted): {raw[:300]!r}")
        assert msg.get("jsonrpc") == "2.0", f"line {n} is not JSON-RPC: {raw[:300]!r}"
