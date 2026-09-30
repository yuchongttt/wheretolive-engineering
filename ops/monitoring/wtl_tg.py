#!/usr/bin/env python3
"""wtl_tg.py — the single home for helper-bot Telegram sending (consolidated 2026-07-23).

Before this, the token was scattered in plaintext across 8+ scripts, each with
its own copy of tg(). Now:
- Credentials live only here: env WTL_TG_TOKEN / WTL_TG_CHAT first, otherwise
  ~/.config/wtl/telegram.env.
- Mac-side scripts all do `from wtl_tg import send_telegram` (scripts in the
  same directory can import it directly).
- Deliberately kept independent copies: heartbeat_check.py (runs on gpu-box and
  must be self-contained); the web routes (process.env).

2026-08-05: removed the built-in token fallback. Credentials now live
only in ~/.config/wtl/telegram.env (chmod 600, outside the repo), one copy each
on the Mac and gpu-box. The launchd plists inject only PATH/HOME, not the token —
reading a file rather than env is deliberate: a new scheduled job needs no plist
change, so it can't be misconfigured. With credentials missing, send_telegram
raises loudly and never swallows the alert (a previous credential refactor left
one job silently broken for 13 days).

Three-level prefix convention (triage on the phone): [ALERT] = broken / look now ·
[TODO] = waiting on a decision · [DAILY]/[WEEKLY] = digests. Otherwise stay
silent — "no message is good news" is the contract for every sender.

send_telegram: under parse_mode, a 400 (unbalanced Markdown entities such as a
stray '_' or '*' in the text) automatically falls back to a plain-text resend so
the alert isn't lost; other errors propagate for the caller to decide;
WTL_TG_DISABLE=1 silences everything (tests / drills).
"""
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CRED_FILE = Path(os.environ.get(
    "WTL_TG_ENV", str(Path.home() / ".config" / "wtl" / "telegram.env")))


def _cred(key: str) -> str | None:
    """env var wins; otherwise read the untracked credential file."""
    val = os.environ.get(key)
    if val:
        return val
    try:
        for line in CRED_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == key:
                    return v.strip()
    except OSError:
        return None
    return None


TG_TOKEN = _cred("WTL_TG_TOKEN")
TG_CHAT = _cred("WTL_TG_CHAT")


def send_telegram(text: str, parse_mode: str | None = None) -> None:
    if os.environ.get("WTL_TG_DISABLE"):
        return
    if not TG_TOKEN or not TG_CHAT:
        # Loud on purpose: a missing credential must not look like "no news".
        raise RuntimeError(
            f"Telegram credentials missing. Set WTL_TG_TOKEN/WTL_TG_CHAT, or create {CRED_FILE}"
            " (chmod 600, containing WTL_TG_TOKEN=... / WTL_TG_CHAT=...)")
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"

    def _post(fields: dict) -> None:
        data = urllib.parse.urlencode(fields).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()

    fields = {"chat_id": TG_CHAT, "text": text}
    if parse_mode:
        fields["parse_mode"] = parse_mode
    try:
        _post(fields)
    except urllib.error.HTTPError as e:
        if e.code != 400 or "parse_mode" not in fields:
            raise
        _post({"chat_id": TG_CHAT, "text": text})
