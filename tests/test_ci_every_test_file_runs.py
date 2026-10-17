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

"""Every test file is executed by some workflow, not only collected.

`pytest tests/ --collect-only` proves a file imports. A file that only ever gets
collected can't block the PR that adds it, and that is how most new regression
tests landed: outside every allowlist, run only later by unsloth's Core zoo job
against zoo main.

MLX files are left out here: they need mlx installed and the MLX jobs pick them
by name.
"""

from __future__ import annotations

import fnmatch
import shlex
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

_TAKES_VALUE = {
    "--ignore", "--ignore-glob", "--deselect", "-k", "-m", "-n", "--dist", "--tb",
    "-p", "--basetemp", "--junitxml", "--junit-xml", "-o", "--timeout", "-c",
    "--rootdir", "--maxfail",
}


def _run_bodies():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        doc = yaml.safe_load(path.read_text(encoding = "utf-8"))
        for job in (doc.get("jobs") or {}).values():
            for step in (job or {}).get("steps") or []:
                if isinstance(step, dict) and isinstance(step.get("run"), str):
                    yield step["run"]


def _pytest_commands(run):
    """Argument lists of every pytest call that executes tests."""
    for line in run.replace("\\\n", " ").splitlines():
        if "--collect-only" in line or line.strip().startswith("#"):
            continue
        try:
            tokens = shlex.split(line, comments = True)
        except ValueError:
            continue
        for i, token in enumerate(tokens):
            if token.endswith(("pytest", "_run_then_exit_hard.py")):
                yield tokens[i + 1:]
                break


def _covered(test_files):
    covered = set()
    for run in _run_bodies():
        for args in _pytest_commands(run):
            targets, ignores, globs = [], [], []
            it = iter(args)
            for arg in it:
                if arg.startswith(("|", ">", "&", ";")):
                    break
                if arg in _TAKES_VALUE:
                    value = next(it, "")
                    if arg == "--ignore":
                        ignores.append(value.rstrip("/"))
                    elif arg == "--ignore-glob":
                        globs.append(value)
                    continue
                if arg.startswith("--ignore="):
                    ignores.append(arg.split("=", 1)[1].rstrip("/"))
                elif arg.startswith("--ignore-glob="):
                    globs.append(arg.split("=", 1)[1])
                elif arg.startswith("tests"):
                    targets.append(arg.split("::", 1)[0].rstrip("/"))
            for name in test_files:
                if any(name == i or name.startswith(i + "/") for i in ignores):
                    continue
                if any(fnmatch.fnmatch(name, g) for g in globs):
                    continue
                if any(name == t or name.startswith(t + "/") or fnmatch.fnmatch(name, t) for t in targets):
                    covered.add(name)
    return covered


def _test_files():
    return sorted(
        p.relative_to(REPO_ROOT).as_posix()
        for p in (REPO_ROOT / "tests").rglob("test_*.py")
        if not p.name.startswith("test_mlx_")
    )


def test_the_parser_sees_the_known_gates():
    covered = _covered(_test_files())
    assert "tests/test_ci_cpu_gate_results.py" in covered
    assert any(name.startswith("tests/security/") for name in covered)


def test_every_test_file_is_executed_by_a_workflow():
    files = _test_files()
    missing = sorted(set(files) - _covered(files))
    assert not missing, (
        f"{len(missing)} of {len(files)} test files are run by no workflow (collect-only "
        f"doesn't count), so they can't fail the PR that adds them: {missing}"
    )
