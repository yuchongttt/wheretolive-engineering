import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json  # noqa: E402
import watchdog_heartbeat as H  # noqa: E402


def test_post_heartbeat_targets_watchdog_host(monkeypatch):
    captured = {}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok":true}'

    def fake_opener(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        captured["key"] = req.headers.get("X-admin-key")
        return FakeResp()

    monkeypatch.setenv("WTL_ADMIN_KEY", "secret")
    ok = H.post_heartbeat(opener=fake_opener)
    assert ok is True
    assert captured["body"]["host_id"] == "mac-watchdog"
    assert captured["key"] == "secret"
    assert "/api/admin/host-metrics" in captured["url"]
