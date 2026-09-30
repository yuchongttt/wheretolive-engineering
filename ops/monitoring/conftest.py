"""Test harness for ops/monitoring.

- Puts this directory on sys.path so the flat scripts import each other the
  same way they do in production (`from wtl_tg import send_telegram`).
- Pins Telegram credentials to dummy values and points the credential file at a
  path that does not exist, so no test can ever read a real token or reach a
  real chat. (Tests that exercise sending patch `urlopen`.)
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

os.environ["WTL_TG_ENV"] = str(HERE / "tests" / "_no_such_telegram.env")
os.environ["WTL_TG_TOKEN"] = "test-token"
os.environ["WTL_TG_CHAT"] = "test-chat"
