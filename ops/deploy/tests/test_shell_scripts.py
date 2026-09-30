"""Shell-level checks for the deploy scripts (added in the public extract)."""
import os
import stat
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
SCRIPTS = ["prod-deploy.sh", "prod-rollback.sh", "pyrun.sh"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_bash_syntax(name):
    r = subprocess.run(["bash", "-n", str(HERE / name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def _fake_python(tmp_path, fail_probes):
    """A stand-in interpreter: `-c ''` is the startup probe (fails the first
    `fail_probes` times), anything else is the real run (logged, exits 7)."""
    fake = tmp_path / "python3"
    fake.write_text(f"""#!/bin/bash
if [ "$1" = "-c" ]; then
  n=$(cat "{tmp_path}/probes" 2>/dev/null || echo 0); n=$((n+1)); echo $n > "{tmp_path}/probes"
  [ "$n" -gt {fail_probes} ] && exit 0 || exit 1
fi
echo "$@" >> "{tmp_path}/runs"
exit 7
""")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    return fake


@pytest.mark.parametrize("fail_probes", [0, 1])
def test_pyrun_probes_then_execs_the_target_exactly_once(tmp_path, fail_probes):
    fake = _fake_python(tmp_path, fail_probes)
    r = subprocess.run(["bash", str(HERE / "pyrun.sh"), str(fake), "job.py", "--flag", "x"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 7                                   # target's exit code survives exec
    assert (tmp_path / "runs").read_text().splitlines() == ["job.py --flag x"]
    assert int((tmp_path / "probes").read_text()) == fail_probes + 1
