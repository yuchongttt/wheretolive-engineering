#!/usr/bin/env python3
"""CLI Telegram notifier — single source of creds via wtl_tg.send_telegram.

Usage:
    python3 notify_telegram.py "message text"
    echo "message" | python3 notify_telegram.py

Creds live in wtl_tg.py only (env WTL_TG_TOKEN/WTL_TG_CHAT, else the untracked cred file).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wtl_tg import send_telegram  # noqa: E402


def main() -> int:
    msg = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    msg = (msg or "").strip()
    if not msg:
        return 0
    try:
        # Plain text: callers pass arbitrary strings (db names, paths with '_')
        # that must never be parsed as Markdown entities.
        send_telegram(msg, parse_mode=None)
    except Exception as e:  # best-effort — never crash the caller
        print(f"notify_telegram failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
