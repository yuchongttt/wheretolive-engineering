"""Static checks for the launchd plists in ../launchd (added in the public extract).

They encode the scheduling conventions the README describes, so a new plist
that breaks one of them fails here instead of at 3 a.m. under launchd."""
import plistlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

LAUNCHD = Path(__file__).resolve().parents[1] / "launchd"
PLISTS = sorted(LAUNCHD.glob("*.plist"))
CHECKOUT = "/opt/wheretolive"


def _load(p):
    return plistlib.loads(p.read_bytes())


def test_there_are_plists():
    assert len(PLISTS) >= 6


@pytest.mark.parametrize("p", PLISTS, ids=lambda p: p.stem)
def test_plist_is_strict_xml_and_label_matches_filename(p):
    d = _load(p)  # plistlib is stricter than launchd (e.g. rejects "--" inside comments)
    assert d["Label"] == p.stem
    if shutil.which("plutil"):
        assert subprocess.run(["plutil", "-lint", str(p)], capture_output=True).returncode == 0


@pytest.mark.parametrize("p", PLISTS, ids=lambda p: p.stem)
def test_exactly_one_scheduling_model(p):
    """Either an always-on daemon (KeepAlive=true) or a periodic job
    (StartInterval / StartCalendarInterval), never both. KeepAlive as a dict
    (e.g. SuccessfulExit=false) is a retry policy on a periodic job, not a daemon."""
    d = _load(p)
    daemon = d.get("KeepAlive") is True
    periodic = ("StartInterval" in d) or ("StartCalendarInterval" in d)
    assert daemon != periodic, f"{p.name}: daemon={daemon} periodic={periodic}"
    assert not (("StartInterval" in d) and ("StartCalendarInterval" in d))


@pytest.mark.parametrize("p", PLISTS, ids=lambda p: p.stem)
def test_pyrun_wrapper_is_called_correctly(p):
    args = _load(p)["ProgramArguments"]
    if args[0].endswith("wtl-pyrun.sh"):
        # wrapper lives OUTSIDE the checkout (launchd cannot exec scripts inside it)
        assert not args[0].startswith(CHECKOUT)
        assert args[1].endswith("/python3") and args[2].endswith(".py")


@pytest.mark.parametrize("p", PLISTS, ids=lambda p: p.stem)
def test_launchd_stdio_is_outside_the_checkout(p):
    """Lesson from synthetic-smoke (2026-06-27..07-07): a launchd log path inside
    the protected checkout can become unopenable and the job fails with exit 78
    and no log at all."""
    d = _load(p)
    for k in ("StandardOutPath", "StandardErrorPath"):
        assert k in d, f"{p.name}: {k} missing"
        assert not d[k].startswith(CHECKOUT + "/"), f"{p.name}: {k} inside checkout"


@pytest.mark.parametrize("p", PLISTS, ids=lambda p: p.stem)
def test_no_secret_values_committed(p):
    env = _load(p).get("EnvironmentVariables", {})
    for k, v in env.items():
        if re.search(r"KEY|TOKEN|SECRET|PASSWORD", k):
            assert v == "<set-me>", f"{p.name}: {k} must be a placeholder"
