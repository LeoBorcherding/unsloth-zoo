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

"""MXFP4-weight x FP8-activation linears for AMD RDNA4 (gfx12), plus their vLLM plugin.

A top-level package rather than part of ``unsloth_zoo`` because it is imported inside vLLM's engine
processes, where neither ``unsloth_zoo``'s GPU init nor its skip-mode stubs (which shadow Triton) may run.
Importing it is cheap; the kernels are compiled with hiprtc on first use.
"""
