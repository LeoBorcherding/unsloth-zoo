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

"""generate_batches on a small card: approx_max_num_seqs below the memory tier's divisor must give a batch
size of 1, not 0 (ZeroDivisionError, then cached in UNSLOTH_VLLM_BATCHES), and an empty input is no batches."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from unsloth_zoo import vllm_utils

GiB = 1024 ** 3


class _FakeLLM:
    def __init__(self, approx_max_num_seqs):
        self.approx_max_num_seqs = approx_max_num_seqs
        self.calls = []

    def generate(self, batch, *args, **kwargs):
        self.calls.append(list(batch))
        return [f"out:{x}" for x in batch]


@pytest.mark.parametrize("total_gb,seqs", [(8, 8), (16, 4), (16, 1), (24, 1), (80, 0)])
def test_small_tier_batch_size_is_at_least_one(monkeypatch, total_gb, seqs):
    monkeypatch.delenv("UNSLOTH_VLLM_BATCHES", raising = False)
    monkeypatch.setattr(vllm_utils, "get_mem_info", lambda: (total_gb * GiB, total_gb * GiB))
    llm = _FakeLLM(seqs)
    out = vllm_utils.generate_batches(llm, ["a", "b", "c"])
    assert out == ["out:a", "out:b", "out:c"]
    assert llm.calls == [["a"], ["b"], ["c"]]
    assert os.environ["UNSLOTH_VLLM_BATCHES"] == "1"
    # The cached size must keep working for the next call in the process.
    assert vllm_utils.generate_batches(llm, ["d"]) == ["out:d"]


def test_create_batches_empty():
    assert vllm_utils.create_batches([], 4) == []


def test_generate_batches_empty_input(monkeypatch):
    monkeypatch.setenv("UNSLOTH_VLLM_BATCHES", "4")
    llm = _FakeLLM(64)
    assert vllm_utils.generate_batches(llm, []) == []
    assert llm.calls == []
