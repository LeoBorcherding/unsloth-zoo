# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.

"""The AMD install hint on a ROCm 10 host names AMD's whl-next index (pytorch.org has no
rocm10.x leaf) with the pinned 10.1 set, a device extra and the 4-bit caveat; CPU only."""

from __future__ import annotations

import pytest

device_type = pytest.importorskip("unsloth_zoo.device_type")

WHL_NEXT = "https://stable.repo.amd.com/rocm/whl-next/"


def _hint(monkeypatch, rocm, gfx = None, indices = ("rocm7.2", "rocm7.1")):
    def no_network(*args, **kwargs):
        raise AssertionError("a ROCm 10 hint must not probe pytorch.org")

    monkeypatch.setattr(device_type, "_detect_amd_rocm_runtime", lambda: True)
    monkeypatch.setattr(device_type, "_detect_rocm_major_minor", lambda: rocm)
    monkeypatch.setattr(device_type, "_detect_gfx_arch", lambda device = None: gfx)
    if rocm and rocm.startswith("10."):
        monkeypatch.setattr(device_type, "_available_pytorch_rocm_indices", no_network)
        monkeypatch.setattr(device_type, "_pytorch_rocm_index_exists", no_network)
    else:
        monkeypatch.setattr(device_type, "_available_pytorch_rocm_indices", lambda: list(indices))
        monkeypatch.setattr(device_type, "_pytorch_rocm_index_exists", lambda index: True)
    device_type._amd_installation_hint.cache_clear()
    try:
        return device_type._amd_installation_hint()
    finally:
        device_type._amd_installation_hint.cache_clear()


def test_a_rocm101_host_gets_the_whl_next_set_with_its_device_extra(monkeypatch):
    hint = _hint(monkeypatch, "10.1", gfx = "gfx1200")
    assert f"--index-url {WHL_NEXT}" in hint, hint
    assert '"torch[device-gfx1200]==2.14.0+rocm10.1.0"' in hint, hint
    assert '"torchvision[device-gfx1200]==0.29.0a0+rocm10.1.0"' in hint, hint
    assert '"torchaudio==2.11.0.3+rocm10.1.0"' in hint, hint
    assert "download.pytorch.org" not in hint, hint
    assert "bitsandbytes" in hint and "load_in_4bit" in hint, hint


def test_another_rocm10_minor_gets_the_10_1_set_and_says_so(monkeypatch):
    hint = _hint(monkeypatch, "10.2")
    assert "Detected ROCm version hint: 10.2" in hint, hint
    assert '"torch[device-gfxXXXX]==2.14.0+rocm10.1.0"' in hint, hint
    assert "rocminfo" in hint and "ROCm 10.1" in hint, hint


def test_a_rocm7_host_keeps_the_pytorch_org_hint(monkeypatch):
    hint = _hint(monkeypatch, "7.2")
    assert "--index-url https://download.pytorch.org/whl/rocm7.2/" in hint, hint
    assert WHL_NEXT not in hint and "load_in_4bit" not in hint, hint

