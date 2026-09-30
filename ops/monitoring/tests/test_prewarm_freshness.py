"""Watchdog for the hourly commute prewarm — the gap the prewarm work left open.

The job's failure mode is silence, not noise. If it stops, new listings simply
stop entering `postcode_hub_commute`, the candidate hit rate drifts down from
99.9%, and every answer stays correct — just slower and more often "estimated"
rather than "verified". Nothing errors, nobody is paged. An idle run writes no
skill_runs row either (deliberately, so quiet hours don't spam), so "no record
for days" reads identically to "nothing to do for days".

The discriminator is therefore NOT "did it run" but "is there work sitting
undone": pending > 0 with no successful run behind it means the job is not
draining its queue. Pending == 0 stays silent no matter how long it has been —
that is the legitimately quiet case.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prewarm_freshness_check import check  # noqa: E402

HOUR = 3600.0
NOW = 1_800_000_000.0


def test_silent_when_there_is_nothing_to_do():
    # No pending work means a long gap since the last run is expected, not broken.
    assert check(pending=0, last_success=NOW - 400 * 24 * HOUR, now=NOW) is None


def test_silent_when_work_is_pending_but_a_run_recently_succeeded():
    assert check(pending=5_000, last_success=NOW - 2 * HOUR, now=NOW) is None


def test_alerts_when_work_sits_undone_past_the_window():
    msg = check(pending=5_000, last_success=NOW - 72 * HOUR, now=NOW)
    assert msg and "5,000" in msg


def test_alerts_when_no_run_has_ever_succeeded_and_work_is_waiting():
    msg = check(pending=5_000, last_success=None, now=NOW)
    assert msg and "never" in msg.lower()


def test_silent_when_only_a_handful_are_pending():
    # A postcode that failed once is re-queued until its second failure retires
    # it, so a small residue is normal and must not alert forever.
    assert check(pending=3, last_success=NOW - 72 * HOUR, now=NOW) is None


def test_a_dead_job_crosses_the_floor_within_a_day():
    # ~943 new postcodes a week x 5 hubs x ~57% band coverage is ~385/day, so the
    # floor cannot be so high that a stopped job hides behind it.
    from prewarm_freshness_check import MIN_PENDING
    assert MIN_PENDING < 385
