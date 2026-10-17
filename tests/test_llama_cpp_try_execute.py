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

"""`try_execute` judges a step by its exit code, and a missing binary reaches auto-install."""

from __future__ import annotations

import importlib
import sys

import pytest


llama_cpp = importlib.import_module("unsloth_zoo.llama_cpp")


def _emit(tmp_path, text, exit_code = 0):
    out = tmp_path / "out.txt"
    out.write_text(text, encoding = "utf-8")
    script = tmp_path / "emit.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.write(open(sys.argv[1], encoding='utf-8').read())\n"
        f"sys.exit({exit_code})\n",
        encoding = "utf-8",
    )
    return f'"{sys.executable}" -I "{script}" "{out}"'


@pytest.mark.parametrize("text", [
    # llama.cpp's CMake on Apple Clang
    "CMake Warning at ggml/src/CMakeLists.txt:347 (message):\n  OpenMP not found\n\n-- Configuring done\n",
    # FindThreads on glibc < 2.34
    "-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed\n"
    "-- Looking for pthread_create in pthreads - not found\n"
    "-- Looking for pthread_create in pthread - found\n",
    # the build step runs with ignore_deprecation=False
    "tools/x.cpp:10:5: warning: 'llama_foo' is deprecated [-Wdeprecated-declarations]\n",
], ids = ["openmp_not_found", "findthreads", "deprecated_warning"])
def test_a_successful_step_with_warning_text_passes(tmp_path, text):
    outputs = []
    llama_cpp.try_execute(_emit(tmp_path, text), print_outputs = outputs)
    assert "".join(outputs) == text


def test_a_failing_step_still_raises_with_its_output(tmp_path):
    with pytest.raises(RuntimeError) as exc:
        llama_cpp.try_execute(_emit(tmp_path, "fatal: unable to access 'x'\n", exit_code = 2))
    message = str(exc.value)
    assert "exit code 2" in message
    assert "unable to access" in message
    assert "Command not found" not in message


def test_a_missing_binary_reaches_install_package(monkeypatch):
    calls = []
    monkeypatch.setattr(llama_cpp, "install_package", lambda package, *a, **k: calls.append(package))
    with pytest.raises(RuntimeError):
        llama_cpp.try_execute_with_auto_install("definitely_missing_tool_xyz --version")
    assert calls == ["definitely_missing_tool_xyz"]


def test_a_failing_real_binary_does_not_install(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(llama_cpp, "install_package", lambda package, *a, **k: calls.append(package))
    with pytest.raises(RuntimeError):
        llama_cpp.try_execute_with_auto_install(_emit(tmp_path, "boom\n", exit_code = 3))
    assert calls == []
