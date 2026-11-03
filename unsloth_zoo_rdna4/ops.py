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

"""Run-time build and launch of the gfx12 MXFP4 W4A8 kernels (kernels.hip).

The serving environment for AMD (vLLM's ROCm wheel over the host's /opt/rocm) carries the HIP runtime
and hiprtc but no compiler, so the source is compiled with hiprtc, cached per arch and source hash, and
launched through hipModuleLaunchKernel on torch's current stream. Under CUDA graph capture the
launches are captured like any other kernel.
"""

import ctypes
import functools
import hashlib
import os
from typing import Optional

import torch

from . import ref

__all__ = [
    "is_available",
    "PackedMxfp4",
    "repack_on_device",
    "w4a8_linear",
]

_SOURCE = os.path.join(os.path.dirname(__file__), "kernels.hip")
_TAG = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
LARGE_M = int(os.environ.get("UNSLOTH_ZOO_RDNA4_LARGE_M", "65"))


def _dll(names):
    roots = [os.environ.get("ROCM_PATH", "/opt/rocm"), "/opt/rocm"]
    for name in names:
        for candidate in [name] + [os.path.join(root, "lib", name) for root in roots]:
            try:
                return ctypes.CDLL(candidate)
            except OSError:
                continue
    raise OSError(f"none of {names} could be loaded")


@functools.cache
def _hip():
    hip = _dll(["libamdhip64.so", "libamdhip64.so.7"])
    hip.hipGetErrorString.restype = ctypes.c_char_p
    hip.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    return hip


def _check(err, what):
    if err != 0:
        raise RuntimeError(f"{what} failed: {_hip().hipGetErrorString(err).decode()} ({err})")


def _arch(device) -> str:
    return torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]


def _compile(arch: str, source: bytes) -> bytes:
    rtc = _dll(["libhiprtc.so", "libhiprtc.so.7"])
    rtc.hiprtcGetErrorString.restype = ctypes.c_char_p
    prog = ctypes.c_void_p()
    err = rtc.hiprtcCreateProgram(ctypes.byref(prog), source, b"rdna4_mxfp4.hip", 0, None, None)
    if err != 0:
        raise RuntimeError("hiprtcCreateProgram: " + rtc.hiprtcGetErrorString(err).decode())
    try:
        opts = [f"--offload-arch={arch}".encode(), b"-O3", b"-std=c++17"]
        err = rtc.hiprtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts))
        if err != 0:
            size = ctypes.c_size_t()
            rtc.hiprtcGetProgramLogSize(prog, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            rtc.hiprtcGetProgramLog(prog, log)
            raise RuntimeError(f"hiprtc compile for {arch} failed:\n" + log.value.decode(errors = "replace"))
        size = ctypes.c_size_t()
        rtc.hiprtcGetCodeSize(prog, ctypes.byref(size))
        code = ctypes.create_string_buffer(size.value)
        rtc.hiprtcGetCode(prog, code)
        return code.raw
    finally:
        rtc.hiprtcDestroyProgram(ctypes.byref(prog))


def _code_object(arch: str) -> bytes:
    with open(_SOURCE, "rb") as f:
        source = f.read()
    key = hashlib.sha256(source + arch.encode()).hexdigest()[:20]
    root = os.environ.get("UNSLOTH_ZOO_RDNA4_CACHE", os.path.expanduser("~/.cache/unsloth_zoo_rdna4"))
    cache = os.path.join(root, f"{arch}-{key}.co")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return f.read()
    code = _compile(arch, source)
    try:
        os.makedirs(root, exist_ok = True)
        tmp = f"{cache}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            f.write(code)
        os.replace(tmp, cache)
    except OSError:
        pass  # a read-only home only costs a recompile next time
    return code


class _Module:
    def __init__(self, device: int):
        self.image = ctypes.create_string_buffer(_code_object(_arch(device)))
        self.handle = ctypes.c_void_p()
        with torch.cuda.device(device):
            _check(_hip().hipModuleLoadData(ctypes.byref(self.handle), self.image), "hipModuleLoadData")
        self.functions = {}

    def function(self, name: str):
        fn = self.functions.get(name)
        if fn is None:
            fn = ctypes.c_void_p()
            _check(_hip().hipModuleGetFunction(ctypes.byref(fn), self.handle, name.encode()), name)
            self.functions[name] = fn
        return fn


@functools.cache
def _module(device: int) -> _Module:
    return _Module(device)


def _launch(device: int, name: str, grid, args):
    """args: tensors (or None) pass as pointers, ints as int32."""
    values = [
        ctypes.c_int(a) if isinstance(a, int) else ctypes.c_void_p(0 if a is None else a.data_ptr())
        for a in args
    ]
    params = (ctypes.c_void_p * len(values))(*[ctypes.addressof(v) for v in values])
    stream = torch.cuda.current_stream(device).cuda_stream
    fn = _module(device).function(name)
    _check(_hip().hipModuleLaunchKernel(fn, *grid, 256, 1, 1, 0, ctypes.c_void_p(stream), params, None), name)


@functools.cache
def is_available(device: int = 0):
    """(ok, reason): a gfx12 GPU on a ROCm build of torch, with hiprtc able to build the kernels."""
    if not (torch.cuda.is_available() and getattr(torch.version, "hip", None)):
        return False, "needs a ROCm build of torch with a visible GPU"
    arch = _arch(device)
    if not arch.startswith("gfx12"):
        return False, f"needs an RDNA4 (gfx12) GPU, found {arch}"
    try:
        _module(device)
    except Exception as e:  # no hiprtc, compile or load failure
        return False, f"kernel build failed: {e}"
    return True, None


@functools.cache
def _lut(device: int) -> torch.Tensor:
    return ref.fold_lut().to(f"cuda:{device}")


@functools.cache
def _counters(device: int) -> torch.Tensor:
    return torch.zeros(1 << 16, dtype = torch.int32, device = f"cuda:{device}")


@functools.cache
def _num_cus(device: int) -> int:
    # HIP reports WGPs on RDNA; two CUs each.
    return torch.cuda.get_device_properties(device).multi_processor_count * 2


def pick(M: int, N: int, K: int, cus: int):
    """(tm, splits) for the small-M kernel. Measured on an R9700: ~2.5 blocks per CU up to M=4, ~1.5 above,
    at most 4 splits, at least 512 k per split."""
    tm = min(4, (M + 15) // 16)
    blocks = -(-N // 128) * -(-M // (16 * tm))
    want = (5 if M <= 4 else 3) * cus // 2
    return tm, max(1, min(round(want / blocks), 4, K // 512))


def repack_on_device(weight: torch.Tensor, weight_scale: torch.Tensor, rows: int = 2048):
    """ref.repack on the weight's device, in row chunks to bound temporary memory."""
    N, K = weight.shape[0], weight.shape[1] * 2
    wp = torch.empty(N // 16, K // 64, 32, 4, dtype = torch.int32, device = weight.device)
    dp = torch.empty(N // 16, K // 128, 16, 4, dtype = torch.uint8, device = weight.device)
    wref = torch.empty(N, dtype = torch.int32, device = weight.device)
    for n0 in range(0, N, rows):
        n1 = min(N, n0 + rows)
        a, b, c = ref.repack(weight[n0:n1], weight_scale[n0:n1])
        wp[n0 // 16 : n1 // 16], dp[n0 // 16 : n1 // 16], wref[n0:n1] = a, b, c
    return wp, dp, wref


class PackedMxfp4:
    """An MXFP4 weight repacked for the gfx12 kernels."""

    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor):
        self.wp, self.dp, self.wref = repack_on_device(weight, weight_scale)


def _linear(x2, wp, dp, wref, bias, tm = None, splits = None, large = None, m_fast = -1):
    device = x2.device.index if x2.device.index is not None else torch.cuda.current_device()
    if x2.dtype not in _TAG:
        x2 = x2.float()
    tag = _TAG[x2.dtype]
    M, K = x2.shape
    T, G = wp.shape[0], wp.shape[1]
    N = T * 16
    out = torch.empty(M, N, dtype = x2.dtype, device = x2.device)
    if M == 0:
        return out
    if bias is not None:
        bias = bias.to(x2.dtype).contiguous()
    lut = _lut(device)
    if large if large is not None else M >= LARGE_M:
        Mp = (M + 15) // 16 * 16
        qt = torch.empty(Mp * K, dtype = torch.uint8, device = x2.device)
        sx = torch.empty(M, dtype = torch.float32, device = x2.device)
        _launch(device, f"act_quant_tiled_{tag}", (Mp, 1, 1), [x2, M, K, qt, sx])
        nb, mb = -(-T // 16), -(-M // 128)
        if m_fast < 0:
            m_fast = int(N * K // 2 > M * K)
        grid = (mb, nb, 1) if m_fast else (nb, mb, 1)
        _launch(device, f"large_m_{tag}", grid, [qt, sx, M, K, wp, dp, wref, lut, N, bias, out, m_fast])
        return out
    xq = torch.empty(M, K, dtype = torch.uint8, device = x2.device)
    sx = torch.empty(M, dtype = torch.float32, device = x2.device)
    _launch(device, f"act_quant_{tag}", (M, 1, 1), [x2, K, xq, sx])
    ptm, psp = pick(M, N, K, _num_cus(device))
    tm, splits = tm or ptm, splits or psp
    gps = -(-G // max(1, min(splits, G)))
    splits = -(-G // gps)
    ws = torch.empty(splits, M, N, dtype = torch.float32, device = x2.device) if splits > 1 else None
    grid = (-(-T // 8), splits, -(-M // (16 * tm)))
    _launch(device, f"small_m_{tm}_{tag}", grid,
            [xq, sx, M, K, wp, dp, wref, lut, N, bias, out, ws, _counters(device), splits, gps])
    return out


def w4a8_linear(x: torch.Tensor, w: PackedMxfp4, bias = None, **tuning) -> torch.Tensor:
    """y = x @ W^T (+ bias), W in MXFP4, x quantized per token to fp8 e4m3."""
    shape = x.shape
    y = _linear(x.reshape(-1, shape[-1]).contiguous(), w.wp, w.dp, w.wref, bias, **tuning)
    return y.reshape(*shape[:-1], y.shape[-1])


@torch.library.custom_op("unsloth_zoo_rdna4::mxfp4_linear", mutates_args = ())
def mxfp4_linear(
    x: torch.Tensor, wp: torch.Tensor, dp: torch.Tensor, wref: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    shape = x.shape
    y = _linear(x.reshape(-1, shape[-1]).contiguous(), wp, dp, wref, bias)
    return y.reshape(*shape[:-1], y.shape[-1])


@mxfp4_linear.register_fake
def _(x, wp, dp, wref, bias):
    return x.new_empty(*x.shape[:-1], wp.shape[0] * 16)
