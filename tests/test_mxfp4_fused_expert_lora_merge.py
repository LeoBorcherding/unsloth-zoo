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

"""merged_16bit from an MXFP4 gpt-oss base must merge a PEFT `target_parameters` expert LoRA.

The LoRA is keyed on `mlp.experts` (or `.base_layer`), never on the dequantized
`<experts>.gate_up_proj` / `.down_proj` group names, so the fused lookup missed it and the
delta was dropped with count 0.
"""

import collections
import shutil
from dataclasses import dataclass

import pytest
import torch
from safetensors.torch import load_file, save_file

import unsloth_zoo.saving_utils as S
from unsloth_zoo.saving_utils import LoraStats, _MOE_MERGE_STATE, _reset_moe_merge_state

E, H, I, R = 2, 64, 64, 2  # gate_up (E, H, 2I) is non-square; down (E, I, H) is square
P = "model.layers.0.mlp.experts"
FNAME = "model.safetensors"


@dataclass
class _Experts:
    num_experts: int


def _write_mxfp4_shard(path):
    torch.manual_seed(0)
    save_file({
        f"{P}.gate_up_proj_blocks": torch.randint(0, 255, (E, 2 * I, H // 32, 16), dtype = torch.uint8),
        f"{P}.gate_up_proj_scales": torch.full((E, 2 * I, H // 32), 127, dtype = torch.uint8),
        f"{P}.down_proj_blocks": torch.randint(0, 255, (E, H, I // 32, 16), dtype = torch.uint8),
        f"{P}.down_proj_scales": torch.full((E, H, I // 32), 127, dtype = torch.uint8),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(H, H),
    }, path)


def _merge(directory, lw):
    result = S._merge_and_overwrite_lora(
        str(directory), FNAME, lw, torch.float32, "GptOssForCausalLM",
        base_model_is_quantized = True, quant_type = "mxfp4",
    )
    return result[0] if isinstance(result, tuple) else result


def _delta(stats, e):
    # PEFT packing: lora_A expert-slowest, lora_B expert-fastest; GPT-OSS W[e] is (in, out).
    a = stats.lora_A[e * R:(e + 1) * R].double()
    b = stats.lora_B.reshape(stats.lora_B.shape[0], R, E)[:, :, e].double()
    return (stats.alpha * (b @ a)).T


@pytest.fixture
def no_lora_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "_active_merge_device", lambda: "cpu")
    _write_mxfp4_shard(str(tmp_path / FNAME))
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    shutil.copy(tmp_path / FNAME, base_dir / FNAME)
    _merge(base_dir, collections.defaultdict(lambda: LoraStats(None, None, None, 0)))
    return load_file(str(base_dir / FNAME))


@pytest.mark.parametrize("layout", ["gate_up_only", "down_only", "both"])
def test_fused_expert_lora_merges_into_dequantized_mxfp4(tmp_path, no_lora_baseline, layout):
    torch.manual_seed(1)
    gu = LoraStats(_Experts(E), torch.randn(R * E, H) * 0.05, torch.randn(2 * I, R * E) * 0.05, 0.5,
                   parameter_name = "gate_up_proj")
    dn = LoraStats(_Experts(E), torch.randn(R * E, I) * 0.05, torch.randn(H, R * E) * 0.05, 0.5,
                   parameter_name = "down_proj")
    lw = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    if layout == "gate_up_only":
        lw[P] = gu
    elif layout == "down_only":
        lw[P] = dn
    else:
        lw[P + ".base_layer"] = gu
        lw[P] = dn
    expected = {
        "gate_up_proj": gu if layout != "down_only" else None,
        "down_proj": dn if layout != "gate_up_only" else None,
    }

    _reset_moe_merge_state()
    count = _merge(tmp_path, lw)
    merged = load_file(str(tmp_path / FNAME))

    assert _MOE_MERGE_STATE["fallback"] == 0, _MOE_MERGE_STATE["first_error"]
    assert count == sum(v is not None for v in expected.values())
    for leaf, stats in expected.items():
        base = no_lora_baseline[f"{P}.{leaf}"].double()
        got = merged[f"{P}.{leaf}"].double()
        want = base.clone()
        if stats is not None:
            for e in range(E):
                want[e] += _delta(stats, e)
        err = (got - want).abs().max().item()
        assert err < 1e-4, f"{layout}: {leaf} off by {err:.2e} (unchanged: {torch.equal(got, base)})"
    _reset_moe_merge_state()
