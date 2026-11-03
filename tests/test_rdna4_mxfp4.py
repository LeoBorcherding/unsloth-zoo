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

"""gfx12 MXFP4 W4A8 linears: the reference math and repack layout on CPU, the kernels against the reference
on an RDNA4 GPU."""

import pytest
import torch

from unsloth_zoo_rdna4 import ref


def random_mx(N, K, spread = 4, seed = 0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, 256, (N, K // 2), generator = g, dtype = torch.uint8)
    scale = (127 - 8 + torch.randint(0, spread + 1, (N, K // 32), generator = g)).to(torch.uint8)
    return w, scale


def test_fold_is_exact_for_shifts_up_to_8():
    w, s = random_mx(32, 256, spread = 8)
    assert torch.equal(ref.folded_weight(w, s), ref.dequant_mxfp4(w, s))


def test_fold_rounds_only_shifts_past_8():
    w, s = random_mx(32, 256, spread = 14, seed = 1)
    _, d = ref.fold_scales(s)
    exact = (d.to(torch.int32) <= 8).repeat_interleave(32, dim = -1)
    assert torch.equal(ref.folded_weight(w, s)[exact], ref.dequant_mxfp4(w, s)[exact])


def test_repack_layout_matches_reference():
    w, s = random_mx(32, 256, spread = 10, seed = 2)
    x = torch.randn(5, 256)
    wp, dp, r = ref.repack(w, s)
    assert wp.shape == (2, 4, 32, 4) and dp.shape == (2, 2, 16, 4)
    torch.testing.assert_close(ref.emulate_repacked(x, wp, dp, r), ref.w4a8_ref(x, w, s), rtol = 1e-5, atol = 1e-5)


def test_w4a8_tracks_the_exact_dequant():
    w, s = random_mx(64, 512, spread = 3, seed = 3)
    x = torch.randn(8, 512)
    exact = x.double() @ ref.dequant_mxfp4(w, s).double().T
    got = ref.w4a8_ref(x, w, s).double()
    assert (got - exact).norm() / exact.norm() < 0.05


def _gpu_or_skip():
    if not (torch.cuda.is_available() and getattr(torch.version, "hip", None)):
        pytest.skip("needs a ROCm GPU")
    from unsloth_zoo_rdna4 import ops

    ok, reason = ops.is_available(torch.cuda.current_device())
    if not ok:
        pytest.skip(reason)
    return ops


def test_act_quant_matches_reference_bytes():
    ops = _gpu_or_skip()
    x = torch.randn(7, 1024) * 3
    xq = torch.empty(7, 1024, dtype = torch.uint8, device = "cuda")
    sx = torch.empty(7, dtype = torch.float32, device = "cuda")
    ops._launch(torch.cuda.current_device(), "act_quant_0", (7, 1, 1), [x.cuda(), 1024, xq, sx])
    rq, rs = ref.quant_fp8_per_token(x)
    assert torch.equal(sx.cpu(), rs)
    assert torch.equal(xq.cpu(), rq.view(torch.uint8))


@pytest.mark.parametrize("N,K", [(128, 512), (1024, 2048), (272, 384)])
@pytest.mark.parametrize("M", [1, 5, 16, 33, 64, 100, 129, 300])
@pytest.mark.parametrize("path", ["auto", "small_split", "large"])
def test_kernel_matches_reference(N, K, M, path):
    ops = _gpu_or_skip()
    w, s = random_mx(N, K, spread = 10, seed = M)
    x = torch.randn(M, K)
    tuning = {"auto": {}, "small_split": {"large": False, "tm": 2, "splits": 3}, "large": {"large": True}}[path]
    y = ops.w4a8_linear(x.cuda(), ops.PackedMxfp4(w.cuda(), s.cuda()), **tuning).cpu()
    want = ref.w4a8_ref(x, w, s)
    assert (y - want).norm() / want.norm() < 1e-5


def test_bf16_bias_and_custom_op():
    ops = _gpu_or_skip()
    w, s = random_mx(256, 1024)
    x = torch.randn(3, 9, 1024, dtype = torch.bfloat16)
    b = torch.randn(256, dtype = torch.bfloat16)
    p = ops.PackedMxfp4(w.cuda(), s.cuda())
    y = torch.ops.unsloth_zoo_rdna4.mxfp4_linear(x.cuda(), p.wp, p.dp, p.wref, b.cuda()).cpu().float()
    want = ref.w4a8_ref(x.reshape(-1, 1024).float(), w, s, b.float()).reshape(3, 9, 256)
    assert y.dtype == torch.float32 and (y - want).norm() / want.norm() < 5e-3


def _fake_vllm(monkeypatch, rocm):
    import enum
    import sys
    import types

    class PlatformEnum(enum.Enum):
        CUDA = 0
        ROCM = 1

    class MxFp4LinearKernel:
        pass

    registry = {PlatformEnum.ROCM: [type("AiterMxfp4LinearKernel", (), {}), type("EmulationMxfp4LinearKernel", (), {})]}
    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.envs": types.SimpleNamespace(environment_variables = {}),
        "vllm.platforms": types.SimpleNamespace(
            PlatformEnum = PlatformEnum, current_platform = types.SimpleNamespace(is_rocm = lambda: rocm)
        ),
        "vllm.model_executor.kernels.linear": types.SimpleNamespace(_POSSIBLE_MXFP4_KERNELS = registry),
        "vllm.model_executor.kernels.linear.mxfp4.base": types.SimpleNamespace(MxFp4LinearKernel = MxFp4LinearKernel),
        "vllm.model_executor.layers.quantization.utils.quant_utils": types.SimpleNamespace(kMxfp4Dynamic = object()),
    }
    modules["vllm"].envs = modules["vllm.envs"]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return registry[PlatformEnum.ROCM], modules["vllm.envs"].environment_variables


def test_vllm_plugin_goes_first_on_rocm_and_keys_the_compile_cache(monkeypatch):
    from unsloth_zoo_rdna4 import vllm_plugin

    kernels, env = _fake_vllm(monkeypatch, rocm = True)
    vllm_plugin.register()
    vllm_plugin.register()
    assert kernels[0].__name__ == "UnslothRdna4MxFp4LinearKernel" and len(kernels) == 3
    assert len(env["UNSLOTH_ZOO_RDNA4_MXFP4"]()) == 16


def test_vllm_plugin_is_inert_off_rocm(monkeypatch):
    from unsloth_zoo_rdna4 import vllm_plugin

    kernels, env = _fake_vllm(monkeypatch, rocm = False)
    vllm_plugin.register()
    assert len(kernels) == 2 and not env
