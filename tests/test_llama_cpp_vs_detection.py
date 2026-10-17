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

"""`_find_visual_studio` finds VS 2026 and VS outside Program Files, as setup.ps1 does.

Otherwise a VS-2026-only box reports `build-essential` missing and the Windows arm
of `install_package` winget-installs VS 2022 Build Tools. vswhere and cmake are stubbed;
the Program Files roots point at a temp tree.
"""

from __future__ import annotations

import importlib
import types

import pytest


llama_cpp = importlib.import_module("unsloth_zoo.llama_cpp")

CMAKE_HELP_WITH_VS18 = "Generators\n  Visual Studio 18 2026        = Generates Visual Studio 2026 project files.\n  Visual Studio 17 2022\n"
CMAKE_HELP_WITHOUT_VS18 = "Generators\n  Visual Studio 17 2022        = Generates Visual Studio 2022 project files.\n"


@pytest.fixture
def host(monkeypatch, tmp_path):
    pf, pf86 = tmp_path / "pf", tmp_path / "pf86"
    pf.mkdir(); pf86.mkdir()
    monkeypatch.setenv("ProgramFiles", str(pf))
    monkeypatch.setenv("ProgramFiles(x86)", str(pf86))
    state = types.SimpleNamespace(pf = pf, pf86 = pf86, cmake_help = CMAKE_HELP_WITH_VS18, vswhere = {}, calls = [])

    def run(argv, **kwargs):
        state.calls.append(argv)
        if argv[0] == "cmake":
            return types.SimpleNamespace(returncode = 0, stdout = state.cmake_help, stderr = "")
        if argv[0].endswith("vswhere.exe"):
            return types.SimpleNamespace(returncode = 0, stdout = state.vswhere.get(argv[-1], ""), stderr = "")
        raise AssertionError(f"unexpected command {argv}")

    monkeypatch.setattr(llama_cpp.subprocess, "run", run)
    return state


def _install(root, year, edition = "BuildTools"):
    path = root / "Microsoft Visual Studio" / year / edition
    (path / "VC" / "Tools" / "MSVC" / "14.50").mkdir(parents = True)
    return str(path)


def test_vs_2026_only_is_found(host):
    path = _install(host.pf, "18")
    assert llama_cpp._find_visual_studio() == ("Visual Studio 18 2026", path)


def test_vs_2026_with_an_unlisted_edition_folder_is_found(host):
    path = _install(host.pf, "18", "Insiders")
    assert llama_cpp._find_visual_studio() == ("Visual Studio 18 2026", path)


def test_vswhere_finds_vs_outside_program_files(host, tmp_path):
    installer = host.pf86 / "Microsoft Visual Studio" / "Installer"
    installer.mkdir(parents = True)
    (installer / "vswhere.exe").write_bytes(b"")
    elsewhere = str(tmp_path / "D" / "VS" / "BuildTools")
    host.vswhere = {"catalog_productLineVersion": "2022\n", "installationPath": elsewhere + "\n"}
    assert llama_cpp._find_visual_studio() == ("Visual Studio 17 2022", elsewhere)


def test_vs_2026_is_skipped_when_cmake_cannot_drive_it(host):
    _install(host.pf, "18")
    path_2022 = _install(host.pf86, "2022")
    host.cmake_help = CMAKE_HELP_WITHOUT_VS18
    assert llama_cpp._find_visual_studio() == ("Visual Studio 17 2022", path_2022)


def test_nothing_installed(host):
    assert llama_cpp._find_visual_studio() == (None, None)
