# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""execute_with_time_limit(backend = "process") under spawn and forkserver, the only
start methods on Windows and the ones the docstring recommends for CUDA. Each case is a
script run as __main__, because spawn re-imports the main module in the child."""

import multiprocessing as mp
import os
import subprocess
import sys
import textwrap

import pytest

RL_ENVIRONMENTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unsloth_zoo", "rl_environments.py"
)

HARNESS = textwrap.dedent('''
    import importlib.util, sys, time
    # Loaded by path: importing the unsloth_zoo package pulls in torch.
    spec = importlib.util.spec_from_file_location("rl_environments", sys.argv[2])
    R = importlib.util.module_from_spec(spec)
    sys.modules["rl_environments"] = R
    spec.loader.exec_module(R)
    METHOD = sys.argv[1]

    def add(a, b):
        return a + b

    @R.execute_with_time_limit(30, backend = "process", start_method = METHOD)
    def run_strategy(strategy, x):
        return strategy(x)

    def sleepy():
        time.sleep(60)

    if __name__ == "__main__":
        print("plain", R.execute_with_time_limit(30, backend = "process", start_method = METHOD)(add)(1, 2))
        strategy = R.create_locked_down_function("def strategy(x):\\n    return x * 2\\n")
        print("decorated", run_strategy(strategy, 21))
        start = time.monotonic()
        try:
            R.execute_with_time_limit(1, backend = "process", start_method = METHOD)(sleepy)()
            print("timeout none")
        except TimeoutError:
            print("timeout", time.monotonic() - start < 20)
''')


def _run(tmp_path, method):
    script = tmp_path / "harness.py"
    script.write_text(HARNESS)
    out = subprocess.run(
        [sys.executable, str(script), method, RL_ENVIRONMENTS],
        capture_output = True, text = True, timeout = 180,
    )
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    return out.stdout.split()


@pytest.mark.parametrize("method", [m for m in ("spawn", "forkserver") if m in mp.get_all_start_methods()])
def test_process_backend_runs_under(tmp_path, method):
    assert _run(tmp_path, method) == ["plain", "3", "decorated", "42", "timeout", "True"]


def _load():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rl_environments_inproc", RL_ENVIRONMENTS)
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    return R


def test_default_start_method_exists_on_this_platform(monkeypatch):
    R = _load()
    seen = []
    monkeypatch.setattr(R, "_run_in_subprocess", lambda *a, **kw: seen.append(kw["start_method"]))
    R.execute_with_time_limit(1, backend = "process")(len)("x")
    assert seen == ["fork" if "fork" in mp.get_all_start_methods() else "spawn"]


def test_failed_start_closes_both_pipe_ends(monkeypatch):
    R = _load()
    closed = []

    class Conn:
        def __init__(self, name): self.name = name
        def close(self): closed.append(self.name)

    class FailingProcess:
        def __init__(self, target, args): self.target = target
        def start(self): raise OSError("cannot start")

    class Ctx:
        Process = FailingProcess
        def Pipe(self, duplex): return Conn("parent"), Conn("child")

    monkeypatch.setattr(R.mp, "get_context", lambda method: Ctx())
    with pytest.raises(OSError, match = "cannot start"):
        R._run_in_subprocess(len, 1, ("x",), {}, start_method = "spawn")
    assert sorted(closed) == ["child", "parent"]
