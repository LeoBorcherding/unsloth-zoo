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

"""vLLM ``general_plugins`` entry point: the RDNA4 MXFP4 kernel first in vLLM's ROCm MXFP4 kernel list."""

import logging

logger = logging.getLogger(__name__)

_KERNEL_NAME = "UnslothRdna4MxFp4LinearKernel"


def _kernel_class():
    import torch
    from torch.nn.parameter import Parameter
    from vllm.model_executor.kernels.linear.mxfp4.base import MxFp4LinearKernel
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic

    class UnslothRdna4MxFp4LinearKernel(MxFp4LinearKernel):
        """MXFP4 weights on fp8 WMMA for RDNA4 (gfx12). Activations run as per-token fp8 e4m3, never
        coarser than an MXFP4-activation checkpoint asks for."""

        @classmethod
        def is_supported(cls, compute_capability = None):
            from . import ops

            try:
                return ops.is_available(torch.cuda.current_device())
            except Exception as e:
                return False, str(e)

        @classmethod
        def can_implement(cls, config):
            if config.activation_quant_key not in (None, kMxfp4Dynamic):
                return False, "only weight-only or MXFP4-activation checkpoints"
            return True, None

        def process_weights_after_loading(self, layer):
            from . import ops

            w, s = layer.weight.data, layer.weight_scale.data
            N, K = w.shape[0], w.shape[1] * 2
            layer.unsloth_rdna4 = N % 16 == 0 and K % 128 == 0
            if not layer.unsloth_rdna4:
                logger.warning("unsloth_zoo: %dx%d MXFP4 linear is not tileable, using dequant + linear", N, K)
                layer.weight_scale = Parameter(s, requires_grad = False)
                return
            wp, dp, wref = ops.repack_on_device(w, s)
            # The repacked tensors have the checkpoint tensors' byte counts, so they replace them.
            layer.weight = Parameter(wp, requires_grad = False)
            layer.weight_scale = Parameter(dp, requires_grad = False)
            layer.unsloth_rdna4_wref = Parameter(wref, requires_grad = False)

        def apply_weights(self, layer, x, bias = None):
            if not layer.unsloth_rdna4:
                from vllm.model_executor.layers.quantization.utils.mxfp4_utils import dequant_mxfp4

                return torch.nn.functional.linear(x, dequant_mxfp4(layer.weight, layer.weight_scale, x.dtype), bias)
            return torch.ops.unsloth_zoo_rdna4.mxfp4_linear(
                x, layer.weight, layer.weight_scale, layer.unsloth_rdna4_wref, bias
            )

    return UnslothRdna4MxFp4LinearKernel


def register():
    """Put the RDNA4 MXFP4 kernel first in vLLM's ROCm MXFP4 kernel list (quark OCP-MX, compressed-tensors
    mxfp4-pack-quantized and online mxfp4 all select from it). Opt out with
    VLLM_DISABLED_KERNELS=UnslothRdna4MxFp4LinearKernel."""
    try:
        from vllm.platforms import PlatformEnum, current_platform

        if not current_platform.is_rocm():
            return
        from vllm.model_executor.kernels.linear import _POSSIBLE_MXFP4_KERNELS
    except Exception as e:
        logger.debug("unsloth_zoo: vLLM has no MXFP4 kernel registry (%s)", e)
        return
    from . import ops  # noqa: F401  registers torch.ops.unsloth_zoo_rdna4.mxfp4_linear before tracing

    kernels = _POSSIBLE_MXFP4_KERNELS.setdefault(PlatformEnum.ROCM, [])
    if not any(k.__name__ == _KERNEL_NAME for k in kernels):
        kernels.insert(0, _kernel_class())
    _key_compile_cache()


def _key_compile_cache():
    """vLLM keys its torch.compile cache on its env registry and config, not on plugin code, so a graph
    traced through the stock MXFP4 path (or an older version of this one) would be replayed against
    these repacked weights. A fingerprint of this package in that registry keeps the keys apart."""
    import hashlib
    import pathlib

    from vllm import envs

    digest = hashlib.sha256()
    for path in sorted(pathlib.Path(__file__).parent.glob("*.*")):
        if path.suffix in (".py", ".hip"):
            digest.update(path.name.encode() + path.read_bytes())
    fingerprint = digest.hexdigest()[:16]
    envs.environment_variables.setdefault("UNSLOTH_ZOO_RDNA4_MXFP4", lambda: fingerprint)
