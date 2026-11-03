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

"""Reference math and load-time repack for MXFP4-weight x FP8-activation linears on gfx12.

Written from the OCP MX v1.0 spec and gfx1201 probes of the WMMA fragment layout.

Checkpoint format (vLLM/quark): weight uint8 [N, K/2], low nibble = even k; weight_scale uint8 E8M0 [N, K/32].
"""

import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
LUT_ROWS = 16


def unpack_e2m1(w: torch.Tensor) -> torch.Tensor:
    """[N, K/2] uint8 -> [N, K] float32 codes (unscaled)."""
    w = w.to(torch.int32)
    codes = torch.stack([w & 0xF, w >> 4], dim = -1).flatten(-2)
    mag = E2M1.to(w.device)[codes & 7]
    return torch.where(codes & 8 != 0, -mag, mag)


def dequant_mxfp4(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Exact MXFP4 dequant to float32 [N, K]."""
    vals = unpack_e2m1(w)
    s = torch.exp2(scale.to(torch.float32) - 127.0)
    return vals * s.repeat_interleave(32, dim = -1)


def quant_fp8_per_token(x: torch.Tensor):
    """Per-row dynamic e4m3 quant. Returns (q fp8 [M,K], scale fp32 [M])."""
    amax = x.float().abs().amax(dim = -1).clamp_min(1e-12)
    s = amax / FP8_MAX
    q = (x.float() / s[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q, s


def fold_lut() -> torch.Tensor:
    """[LUT_ROWS, 8] uint8: e4m3 bytes of e2m1 magnitude * 2^-d, row d (round to nearest even)."""
    d = torch.arange(LUT_ROWS, dtype = torch.float32)[:, None]
    return (E2M1[None, :] * torch.exp2(-d)).to(FP8).view(torch.uint8)


def fold_scales(scale: torch.Tensor):
    """E8M0 [N, K/32] -> (row exponent ref [N] int32, shift d [N, K/32] uint8 clamped to LUT_ROWS-1)."""
    e = scale.to(torch.int32)
    ref = e.amax(dim = -1)
    d = (ref[:, None] - e).clamp(max = LUT_ROWS - 1)
    return ref, d.to(torch.uint8)


def folded_weight(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """What the kernel actually multiplies: fp8(e2m1 * 2^-d) * 2^(ref-127), as float32 [N, K]."""
    ref, d = fold_scales(scale)
    w = w.to(torch.int32)
    codes = torch.stack([w & 0xF, w >> 4], dim = -1).flatten(-2)
    lut = fold_lut().view(FP8).float()
    dk = d.to(torch.long).repeat_interleave(32, dim = -1)
    mag = lut[dk, (codes & 7).long()]
    vals = torch.where(codes & 8 != 0, -mag, mag)
    return vals * torch.exp2(ref.float() - 127.0)[:, None]


def w4a8_ref(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor, bias = None) -> torch.Tensor:
    """Kernel-exact numerics up to fp32 summation order. x [M, K] -> [M, N] float32."""
    q, s = quant_fp8_per_token(x)
    out = (q.double() @ folded_weight(w, scale).double().T) * s.double()[:, None]
    if bias is not None:
        out = out + bias.double()
    return out.float()


# Repacked layout, consumed by kernels.hip.
#
# Weights: [N/16, K/64, 32 lanes, 4 uint32] = per (n-tile, 64-k group) 512 contiguous bytes, 16 per lane.
#   Lane l covers row n = 16*t + l%16; uint32 j covers k-step s = 4*g + j, k = 16*s + (l/16)*8 + 0..7.
#   Inside the uint32, byte b holds k_b in its low nibble and k_{b+4} in its high nibble, so
#   (u & 0x07070707) indexes k0..k3 and ((u >> 4) & 0x07070707) indexes k4..k7.
# Shifts: [N/16, K/128, 16 rows, 4 uint8] = per (n-tile, 128-k group) the d of 4 MX blocks for each row.

def repack(w: torch.Tensor, scale: torch.Tensor):
    N, K2 = w.shape
    K = K2 * 2
    assert N % 16 == 0 and K % 128 == 0, (N, K)
    w = w.to(torch.int32)
    codes = torch.stack([w & 0xF, w >> 4], dim = -1).flatten(-2)          # [N, K]
    c = codes.view(N // 16, 16, K // 64, 4, 2, 8)                        # t, row, g, j, half, kk
    c = c.permute(0, 2, 4, 1, 3, 5)                                      # t, g, half, row, j, kk
    packed = c[..., :4] | (c[..., 4:] << 4)                              # byte b = k_b | k_{b+4} << 4
    u = (packed[..., 0] | packed[..., 1] << 8 | packed[..., 2] << 16 | packed[..., 3] << 24)
    u = u.to(torch.int64)
    wp = torch.where(u >= 2**31, u - 2**32, u).to(torch.int32).reshape(N // 16, K // 64, 32, 4)
    ref, d = fold_scales(scale)
    dp = d.view(N // 16, 16, K // 128, 4).permute(0, 2, 1, 3).contiguous()
    return wp.contiguous(), dp, ref.to(torch.int32)


def emulate_repacked(x: torch.Tensor, wp: torch.Tensor, dp: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """CPU emulation of the kernel's byte path on the repacked tensors. Validates the layout."""
    T, G = wp.shape[0], wp.shape[1]
    N, K = T * 16, G * 64
    lut = fold_lut().view(FP8).float()
    u = wp.to(torch.int64) & 0xFFFFFFFF                                  # [T, G, 32, 4]
    bytes_ = torch.stack([(u >> (8 * b)) & 0xFF for b in range(4)], dim = -1)  # [T,G,32,4,4]
    lo, hi = bytes_ & 0xF, bytes_ >> 4
    codes = torch.cat([lo, hi], dim = -1)                                  # [T,G,32,j,8] k order
    lane = torch.arange(32)
    row = lane % 16
    half = lane // 16
    W = torch.zeros(N, K)
    dd = dp.to(torch.long)                                               # [T, K/128, 16, 4]
    for t in range(T):
        for g in range(G):
            for j in range(4):
                s = 4 * g + j
                kblk = (16 * s) // 32
                for l in range(32):
                    n = 16 * t + row[l]
                    d = dd[t, kblk // 4, row[l], kblk % 4]
                    cc = codes[t, g, l, j]
                    mag = lut[d, cc & 7]
                    v = torch.where(cc & 8 != 0, -mag, mag)
                    k0 = 16 * s + half[l] * 8
                    W[n, k0:k0 + 8] = v
    W = W * torch.exp2(ref.float() - 127.0)[:, None]
    q, sa = quant_fp8_per_token(x)
    return ((q.double() @ W.double().T) * sa.double()[:, None]).float()
