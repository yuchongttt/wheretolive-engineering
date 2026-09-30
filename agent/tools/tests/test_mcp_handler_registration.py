"""@server.call_tool() must be pinned on the real handler, not on whatever
function happens to have been inserted right below it.

Real case (2026-09-23 user simulation, rental persona): the model called
get_area_overview 4 times in a row and got back, all 4 times,

    _scope_and_outcode() takes 1 positional argument but 2 were given

so it told the user "we have no rental listings inventory at all" and sent
them to a property portal — while get_area_overview is exactly the tool the
system prompt describes as "**Use it for any what rent / rental yield
question** — never tell the user rental data is unavailable". The tool-event
log showed this tool failing 100% of the time from 2026-09-05 — a full 18 days,
real users included.

Root cause: an outcode fix inserted two new functions between
`@server.call_tool()` and `async def handle_call_tool`. The MCP SDK's call_tool
decorator registers **the function immediately below it** as the
CallToolRequest handler and returns it unchanged — so `_scope_and_outcode(...)`
kept working as a plain function, all 6 cases in
tests/test_area_overview_outcode.py that call it directly stayed green, and the
handler on the real call path had become a one-argument helper.

The signature of this class of error: unit tests all green, tool 100% dead.
So this checks not a function's return value but **which function the
decorator is pinned to**: a static pass over every MCP module (no import, zero
cost, full coverage).

(Extract note: the original file also made one live call to get_area_overview
through its registered handler; that case needs the production database and is
not included here.)
"""
import ast
from pathlib import Path

import pytest

SKILLS = Path(__file__).resolve().parents[1]

DECORATOR_CONTRACT = {
    # decorator attr -> (must be async, exact positional arg count)
    "call_tool": (True, 2),   # (name, arguments)
    "list_tools": (True, 0),
}


def _decorated(path: Path):
    """[(decorator_attr, FunctionDef)] for every @server.<attr>() in a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            call = dec.func if isinstance(dec, ast.Call) else dec
            if (isinstance(call, ast.Attribute)
                    and isinstance(call.value, ast.Name)
                    and call.value.id == "server"
                    and call.attr in DECORATOR_CONTRACT):
                out.append((call.attr, node))
    return out


def _is_mcp_server_module(path: Path) -> bool:
    """Only a module-level `server = Server(...)` makes a module an MCP server.

    Don't select by searching for the string "@server.call_tool(" — that very
    line appears in tool_runner.py's comments, and string matching would pull
    in a dispatcher that has no server.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover
        return False
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        value = getattr(node, "value", None)
        if (any(isinstance(t, ast.Name) and t.id == "server" for t in targets)
                and isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name) and value.func.id == "Server"):
            return True
    return False


MCP_MODULES = sorted(p for p in SKILLS.glob("*.py") if _is_mcp_server_module(p))


class TestDecoratorLandsOnTheHandler:
    @pytest.mark.parametrize("path", MCP_MODULES, ids=lambda p: p.stem)
    def test_registered_function_has_the_handler_shape(self, path):
        seen = _decorated(path)
        assert any(attr == "call_tool" for attr, _ in seen), \
            f"{path.name}: @server.call_tool() is not on any function — the tool won't be registered"
        for attr, fn in seen:
            want_async, want_args = DECORATOR_CONTRACT[attr]
            is_async = isinstance(fn, ast.AsyncFunctionDef)
            nargs = len(fn.args.args) + len(fn.args.posonlyargs)
            assert (is_async, nargs) == (want_async, want_args), (
                f"{path.name}: @server.{attr}() is pinned on {fn.name}("
                f"{'async ' if is_async else ''}{nargs} positional args), "
                f"expected an async handler with {want_args} args. "
                f"Most common cause: someone inserted a new function between the decorator and the handler."
            )
