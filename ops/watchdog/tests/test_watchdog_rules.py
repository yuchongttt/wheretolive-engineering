import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import watchdog_rules as R  # noqa: E402


def test_parse_launchctl_running_and_exit():
    assert R.parse_launchctl("1234\t0\txyz.wheretolive.radar-matcher") == {"pid": 1234, "last_exit": 0}
    assert R.parse_launchctl("-\t78\txyz.wheretolive.radar-matcher") == {"pid": None, "last_exit": 78}


def test_job_unhealthy_requires_bad_exit_AND_stale():
    out = "-\t78\txyz.wheretolive.radar-matcher"
    assert R.job_unhealthy("radar-matcher", out, fresh=False) is True    # bad exit + stale
    assert R.job_unhealthy("radar-matcher", out, fresh=True) is False    # bad exit but fresh -> healthy
    good = "-\t0\txyz.wheretolive.radar-matcher"
    assert R.job_unhealthy("radar-matcher", good, fresh=False) is False  # idle one-shot exit 0 -> healthy


def test_allowlist_excludes_next_prod():
    assert "next-prod" not in R.ALLOWED_JOBS
    assert "radar-matcher" in R.ALLOWED_JOBS


# Added in the public extract: the allowlist moved from code to watchdog_jobs.json.
def test_load_jobs_reads_budgets_and_ledger_skills(tmp_path):
    p = tmp_path / "jobs.json"
    p.write_text(json.dumps({"jobs": {
        "a": {"fresh_minutes": 12},
        "b": {},
        "c": {"fresh_minutes": 70, "ledger_skill": "c-skill"},
    }}))
    allowed, fresh, ledger = R.load_jobs(p)
    assert allowed == ["a", "b", "c"]                 # order preserved
    assert fresh == {"a": 12, "b": 70, "c": 70}        # 70 min default
    assert ledger == {"c": "c-skill"}
