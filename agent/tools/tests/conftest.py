"""Suite-wide safety net: put the tool modules on sys.path, and install an `mcp`
stub when the real package is missing (2026-08-14 Simp2/Reuse3).

The modules under test import `mcp` at top level. Without a stub, running under
an interpreter that lacks the real package turns a whole test file into a
silent collection error — i.e. no protection at all.

Previously nine test files each carried their own stub, and they had drifted
into two shapes: a half stub (whole-package try/except, no TextContent) could
be collected first and fool the `import mcp` probe of later files, so their
cases failed with AttributeError while each file passed on its own. Centralising
the stub in conftest removes that ordering problem: pytest guarantees conftest
is imported before any test module, at which point sys.modules holds either the
real mcp or nothing — never a half stub. The stub installs the full surface the
tests need; when the real mcp is importable nothing is touched.
"""
import sys
import types as _types
from pathlib import Path

# Extract layout: the tool modules live one level up from tests/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _Stub:  # must swallow Server() / @server.list_tools() and friends
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return lambda fn: fn

    def __getattr__(self, _name):
        return _Stub()


class _TextContent:
    """handle_call_tool returns [types.TextContent(...)]; tests read back .text."""

    def __init__(self, type="text", text=""):
        self.type = type
        self.text = text


try:  # pragma: no cover - depends on the interpreter running the suite
    import mcp  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    for _name in ("mcp", "mcp.server", "mcp.server.models", "mcp.server.stdio",
                  "mcp.types"):
        sys.modules.setdefault(_name, _types.ModuleType(_name))
    sys.modules["mcp.server"].Server = _Stub
    sys.modules["mcp.server"].NotificationOptions = _Stub
    sys.modules["mcp.server.models"].InitializationOptions = _Stub
    sys.modules["mcp.server.stdio"].stdio_server = _Stub()
    sys.modules["mcp.types"].TextContent = _TextContent
    sys.modules["mcp.types"].Tool = _Stub
    sys.modules["mcp"].server = sys.modules["mcp.server"]
    sys.modules["mcp"].types = sys.modules["mcp.types"]
