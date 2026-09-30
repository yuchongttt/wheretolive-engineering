import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mac"))
import queue_status_report as Q  # noqa: E402

QN = "image-download-queue"  # a real name from Q.QUEUES


def _data(fail=0, stalled=False):
    return {"queues": [{"name": QN, "done_last_hour": 10, "backlog": 5,
                        "failed_lifetime": fail, "stats_age_seconds": 4000 if stalled else 10}]}


def test_healthy_is_silent():
    msg, st = Q.decide({}, _data())
    assert msg is None


def test_failure_burst_at_the_threshold_triggers_onset():
    # Drive off the constant, not a magic number. This asserted fail=5 back when
    # ANY delta >= 1 alerted; 5c686c6d (2026-07-08) deliberately raised the bar to
    # FAIL_BURST to stop self-healing blips spamming red/green, and the test was
    # left behind asserting the retired contract.
    prev = {"failed": {QN: 0}, "problem": {}}
    msg, st = Q.decide(prev, _data(fail=Q.FAIL_BURST))
    assert msg is not None and "🔴" in msg
    assert st["problem"].get(QN)


def test_a_small_failure_blip_stays_silent():
    # The whole point of the threshold: transient failures self-heal, and paging
    # on them is what made the alert ignorable.
    prev = {"failed": {QN: 0}, "problem": {}}
    msg, _ = Q.decide(prev, _data(fail=Q.FAIL_BURST - 1))
    assert msg is None


def test_a_sustained_streak_triggers_onset_even_below_the_burst_bar():
    # The second arm of 5c686c6d: not big enough to burst, but not self-healing
    # either — new failures every run for FAIL_STREAK_RUNS runs.
    state, msg = {"failed": {QN: 0}, "problem": {}}, None
    for i in range(1, Q.FAIL_STREAK_RUNS + 1):
        msg, state = Q.decide(state, _data(fail=i))
    assert msg is not None and "🔴" in msg


def test_stalled_triggers_onset():
    prev = {"failed": {QN: 0}, "problem": {}}
    msg, st = Q.decide(prev, _data(fail=0, stalled=True))
    assert msg is not None and "🔴" in msg


def test_clear_after_problem_notifies_once():
    prev = {"failed": {QN: 5}, "problem": {QN: True}}
    msg, st = Q.decide(prev, _data(fail=5))          # no NEW failures, not stalled -> cleared
    assert msg is not None and "back to normal" in msg
    assert not st["problem"].get(QN)
    msg2, _ = Q.decide(st, _data(fail=5))            # stays silent afterwards
    assert msg2 is None
