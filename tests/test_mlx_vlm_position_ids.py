# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""GLM-OCR M-RoPE position ids built by the MLX VLM collator, checked against
HF's GlmOcrModel.get_rope_index layout. Real MLX on macOS, the torch shim elsewhere."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True, scope="module")
def _mlx():
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        pytest.importorskip("torch")
        from mlx_simulation import simulate_mlx_on_torch

        simulate_mlx_on_torch()


T, S, E, I, V = 5, 100, 103, 101, 102  # text, begin/end of image, image token, video token


def _positions(rows, image_grids=None, video_grids=None):
    from unsloth_zoo.mlx.utils import _vlm_positions_for_compile

    width = max(map(len, rows))
    ids = np.array([row + [0] * (width - len(row)) for row in rows], np.int32)
    mask = np.array([[1] * len(row) + [0] * (width - len(row)) for row in rows], np.int32)
    config = {"model_type": "glm_ocr", "image_start_token_id": S, "image_token_id": I,
              "video_token_id": V, "vision_config": {"spatial_merge_size": 2}}
    built = _vlm_positions_for_compile(
        {"input_ids": ids, "attention_mask": mask, "image_grid_thw": image_grids,
         "video_grid_thw": video_grids}, config)
    out = np.asarray(built["position_ids"])
    return [[out[axis, b, :len(row)].tolist() for axis in range(3)] for b, row in enumerate(rows)]


def test_glm_ocr_lays_out_every_image_in_a_row():
    row = [T, S, I, I, I, I, E, T, S, I, I, I, I, T]
    (t, h, w), = _positions([row], [(1, 4, 4), (1, 4, 4)])
    assert t == [0, 1, 2, 2, 2, 2, 4, 5, 6, 7, 7, 7, 7, 9]
    assert h == [0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7, 8, 8, 9]
    assert w == [0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 7, 8, 9]


def test_glm_ocr_later_rows_take_their_own_image_grid():
    row0 = [T, S] + [I] * 4 + [T, S] + [I] * 16 + [T]
    row1 = [T, S] + [I] * 4 + [T]
    _, (t, h, w) = _positions([row0, row1], [(1, 4, 4), (1, 8, 8), (1, 4, 4)])
    assert t == [0, 1, 2, 2, 2, 2, 4]
    assert h == [0, 1, 2, 2, 3, 3, 4]
    assert w == [0, 1, 2, 3, 2, 3, 4]


def test_glm_ocr_places_each_video_frame_separately():
    row = [T, S, V, V, V, V, E, T, S, V, V, V, V, E, T]  # two frames, a timestamp between them
    (t, h, w), = _positions([row], video_grids=[(2, 4, 4)])
    assert t == [0, 1, 2, 2, 2, 2, 4, 5, 6, 7, 7, 7, 7, 9, 10]
    assert h == [0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7, 8, 8, 9, 10]
    assert w == [0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 7, 8, 9, 10]
