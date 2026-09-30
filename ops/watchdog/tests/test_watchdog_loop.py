import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timezone  # noqa: E402
import watchdog as W  # noqa: E402
import watchdog_ledger as L  # noqa: E402


def _now():
    return datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)


def _deps(**over):
    sent = []
    d = dict(
        job_state=lambda label: "-\t78\tx" if label == "radar-matcher" else "-\t0\tx",
        job_fresh=lambda label: False if label == "radar-matcher" else True,
        kickstart=lambda label: sent.append(("kickstart", label)) or 0,
        hetzner_ok=lambda: True,
        restart_worker=lambda: 0,
        send=lambda text: sent.append(("tg", text)),
        post_heartbeat=lambda: sent.append(("hb", None)),
        log_tail=lambda label: "exit=78",
        # Reachability healthy by default — the net-wedged rule has its own
        # suite in test_watchdog_net.py; here it must stay out of the way.
        ha_connections=lambda: 4,
        raw_internet_ok=lambda: True,
        repair_net=lambda fault: sent.append(("repair", fault)) or 0,
        net_diag=lambda: "default=router@en1 gw=up wifi=on",
    )
    d.update(over)
    return d, sent


def test_observe_mode_never_executes_fix(tmp_path):
    conn = L.open_db(str(tmp_path / "ops.db"))
    deps, sent = _deps()
    W.run_once(conn, "observe", _now(), deps)
    assert not any(a == "kickstart" for a, _ in sent)   # no real fix
    assert not any(a == "tg" for a, _ in sent)          # observe is silent
    assert any(a == "hb" for a, _ in sent)              # heartbeat still posted
    assert L.last_open_problem(conn, "launchd-job", "radar-matcher")["action"] == "detect"


def test_enforce_kickstarts_unhealthy_job(tmp_path):
    conn = L.open_db(str(tmp_path / "ops.db"))
    deps, sent = _deps()
    W.run_once(conn, "enforce", _now(), deps)
    assert ("kickstart", "radar-matcher") in sent
    assert any(a == "hb" for a, _ in sent)


def test_enforce_heartbeat_posts_even_when_all_healthy(tmp_path):
    conn = L.open_db(str(tmp_path / "ops.db"))
    deps, sent = _deps(job_state=lambda l: "-\t0\tx", job_fresh=lambda l: True)
    W.run_once(conn, "enforce", _now(), deps)
    assert any(a == "hb" for a, _ in sent) and not any(a == "kickstart" for a, _ in sent)
