#!/usr/bin/env python3
"""POST a watchdog liveness heartbeat to the existing admin host-metrics API.
The gpu-box heartbeat checker reads mac-watchdog freshness to catch the
watchdog itself dying."""
import os
import json
import urllib.request
from datetime import datetime, timezone

API = os.environ.get("WTL_API", "http://127.0.0.1:3000")
ENDPOINT = f"{API}/api/admin/host-metrics"


def post_heartbeat(now=None, opener=urllib.request.urlopen) -> bool:
    key = os.environ.get("WTL_ADMIN_KEY", "")
    payload = {"host_id": "mac-watchdog", "hostname": "mac", "os": "Darwin",
               "metrics": {"watchdog_ts": (now or datetime.now(timezone.utc)).isoformat()}}
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "x-admin-key": key}, method="POST")
    try:
        with opener(req, timeout=10) as r:
            return b'"ok":true' in r.read()
    except Exception:
        return False
