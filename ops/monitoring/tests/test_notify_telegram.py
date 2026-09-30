import importlib.util
import io
import sys
import urllib.error
import urllib.parse
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
# The modules under test do a plain `from wtl_tg import send_telegram`, which
# needs the module directory importable. Without this the file only collected
# when some EARLIER test happened to put it on sys.path — alone it died at
# import, in a suite it ran. Own the dependency here instead of inheriting it.
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import wtl_tg  # noqa: E402 — the single home for Telegram credentials + sending


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cca = _load("claude_cost_alert")
nt = _load("notify_telegram")


class _FakeResp:
    def read(self):
        return b'{"ok":true}'

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http400():
    return urllib.error.HTTPError(
        "https://api.telegram.org", 400, "Bad Request", None,
        io.BytesIO(b'{"ok":false,"description":"can\'t parse entities"}'))


def _capture_posts(monkeypatch, fail_first_n=0, code=400):
    """Patch urlopen; record each request body (decoded); optionally fail the
    first N posts with an HTTPError of the given code."""
    posts = []

    def fake_urlopen(req, timeout=None):
        body = dict(urllib.parse.parse_qsl(req.data.decode()))
        posts.append(body)
        if len(posts) <= fail_first_n:
            raise urllib.error.HTTPError(
                req.full_url, code, "err", None, io.BytesIO(b"{}"))
        return _FakeResp()

    # Patch wtl_tg's urlopen, not claude_cost_alert's: sending was consolidated
    # into wtl_tg (cca now only re-exports send_telegram for old importers, and
    # no longer imports urllib at all). Patching the old seam raised
    # "module 'claude_cost_alert' has no attribute 'urllib'".
    monkeypatch.setattr(wtl_tg.urllib.request, "urlopen", fake_urlopen)
    return posts


def test_send_telegram_defaults_to_plain_text(monkeypatch):
    # The 2026-07-23 consolidation into wtl_tg deliberately flipped the default:
    # parse_mode is None unless the caller asks for it (claude_cost_alert now
    # passes parse_mode="Markdown" explicitly at its one call site). Plain is the
    # safe default — an unbalanced '_' or '*' in an alert 400s under Markdown.
    posts = _capture_posts(monkeypatch)
    cca.send_telegram("*bold* alert")
    assert len(posts) == 1
    assert "parse_mode" not in posts[0]
    assert posts[0]["text"] == "*bold* alert"


def test_send_telegram_markdown_is_opt_in(monkeypatch):
    posts = _capture_posts(monkeypatch)
    cca.send_telegram("*bold* alert", parse_mode="Markdown")
    assert len(posts) == 1
    assert posts[0]["parse_mode"] == "Markdown"


def test_send_telegram_400_falls_back_to_plain(monkeypatch):
    # A message with an unbalanced '_' 400s under Markdown; it must be
    # re-sent (once) without parse_mode instead of being dropped.
    msg = "backup: unclassified DB 'security' — add to ALLOW or DENY in backup_dbs.py"
    posts = _capture_posts(monkeypatch, fail_first_n=1)
    cca.send_telegram(msg, parse_mode="Markdown")
    assert len(posts) == 2
    assert posts[0]["parse_mode"] == "Markdown"
    assert "parse_mode" not in posts[1]
    assert posts[1]["text"] == msg


def test_send_telegram_plain_mode_sends_once_no_parse_mode(monkeypatch):
    posts = _capture_posts(monkeypatch)
    cca.send_telegram("plain text with under_scores", parse_mode=None)
    assert len(posts) == 1
    assert "parse_mode" not in posts[0]


def test_send_telegram_non_400_reraises(monkeypatch):
    posts = _capture_posts(monkeypatch, fail_first_n=99, code=403)
    try:
        cca.send_telegram("hi")
        raised = False
    except urllib.error.HTTPError as e:
        raised = e.code == 403
    assert raised
    assert len(posts) == 1  # no retry loop on non-parse errors


def test_send_telegram_plain_400_reraises(monkeypatch):
    # Already plain -> a 400 is not a Markdown problem; no infinite fallback.
    posts = _capture_posts(monkeypatch, fail_first_n=99, code=400)
    try:
        cca.send_telegram("hi", parse_mode=None)
        raised = False
    except urllib.error.HTTPError:
        raised = True
    assert raised
    assert len(posts) == 1


def test_notify_cli_uses_plain_text(monkeypatch):
    # The generic CLI relays arbitrary text (db names, paths); it must not
    # let Telegram interpret it as Markdown at all.
    calls = []
    monkeypatch.setattr(nt, "send_telegram",
                        lambda msg, **kw: calls.append((msg, kw)))
    monkeypatch.setattr(sys, "argv", ["notify_telegram.py", "db 'area_intel' note"])
    assert nt.main() == 0
    assert calls == [("db 'area_intel' note", {"parse_mode": None})]
