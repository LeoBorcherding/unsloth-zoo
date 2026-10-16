# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.

"""reset_unsloth_gradient_checkpointing_buffers must give back the training peak. Tensor.resize_ to a smaller size
only shrinks the view, so the CPU and GPU offload buffers kept their peak storage after every trainer.train()."""

from __future__ import annotations

import pytest
import torch

GROWN = 1 << 20


def _grown(numel, pinned = None):
    x = torch.empty(numel, dtype = torch.bfloat16)
    x.resize_(GROWN)  # what an offload of a large activation does mid-training
    if pinned is not None:
        setattr(x, "_unsloth_is_pinned", pinned)
    return x


@pytest.fixture
def gc_mod(monkeypatch):
    gc_mod = pytest.importorskip("unsloth_zoo.gradient_checkpointing")
    seen = {"released": [], "new_host": [], "host_cache": 0}
    monkeypatch.setattr(gc_mod, "DEVICE_TYPE", "cuda")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(gc_mod, "_double_buffer_disabled", lambda: False)
    monkeypatch.setattr(gc_mod, "NEXT_BUFFER_SLOT", None, raising = False)
    monkeypatch.setattr(gc_mod, "_release_pinned", lambda n: seen["released"].append(n))

    def new_host(numel, dtype):
        seen["new_host"].append((numel, dtype))
        return torch.empty(numel, dtype = dtype)
    monkeypatch.setattr(gc_mod, "_new_host_buffer", new_host)

    def host_cache():
        seen["host_cache"] += 1
        return False
    monkeypatch.setattr(gc_mod, "_empty_host_cache", host_cache)
    gc_mod._seen = seen
    return gc_mod


def test_reset_shrinks_cpu_and_gpu_storage(monkeypatch, gc_mod):
    cpu_n, gpu_n = gc_mod.INITIAL_CPU_BUFFER_SIZE, gc_mod.INITIAL_GPU_BUFFER_SIZE
    cpu = [_grown(cpu_n, pinned = False), torch.empty(cpu_n, dtype = torch.bfloat16)]
    untouched = cpu[1]
    monkeypatch.setattr(gc_mod, "CPU_BUFFERS", cpu, raising = False)
    monkeypatch.setattr(gc_mod, "GPU_BUFFERS", (_grown(gpu_n),), raising = False)
    monkeypatch.setattr(gc_mod, "GPU_BUFFERS_B", (_grown(gpu_n),), raising = False)

    gc_mod.reset_unsloth_gradient_checkpointing_buffers()

    for x in gc_mod.CPU_BUFFERS:
        assert x.numel() == cpu_n
        assert x.untyped_storage().nbytes() == cpu_n * 2
    for x in (*gc_mod.GPU_BUFFERS, *gc_mod.GPU_BUFFERS_B):
        assert x.numel() == gpu_n
        assert x.untyped_storage().nbytes() == gpu_n * 2
    # A slot that never grew is kept as is.
    assert gc_mod.CPU_BUFFERS[1] is untouched


def test_reset_hands_a_grown_pinned_slot_back(monkeypatch, gc_mod):
    n = gc_mod.INITIAL_CPU_BUFFER_SIZE
    monkeypatch.setattr(gc_mod, "CPU_BUFFERS", [_grown(n, pinned = True)], raising = False)
    monkeypatch.setattr(gc_mod, "GPU_BUFFERS", (), raising = False)
    monkeypatch.setattr(gc_mod, "GPU_BUFFERS_B", (), raising = False)

    gc_mod.reset_unsloth_gradient_checkpointing_buffers()

    seen = gc_mod._seen
    assert seen["released"] == [GROWN * 2]
    assert seen["new_host"] == [(n, torch.bfloat16)]
    assert seen["host_cache"] == 1
    assert gc_mod.CPU_BUFFERS[0].untyped_storage().nbytes() == n * 2
