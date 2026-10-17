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

"""Checks the MLX compile merge replacements keep from upstream mlx-vlm: the
image-token / feature count check, and Idefics' padding-image filter.
Real MLX on macOS, the torch shim elsewhere."""

from __future__ import annotations

import types

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


@pytest.fixture(autouse=True)
def _shim_feature_size(monkeypatch):
    """The shim's arrays are torch tensors, whose `.size` is a method; the count check reads MLX's int."""
    import unsloth_zoo.mlx.compile as mc
    from mlx_simulation import mlx_is_simulated

    if mlx_is_simulated():
        check = mc._raise_on_feature_count_mismatch
        monkeypatch.setattr(mc, "_raise_on_feature_count_mismatch", lambda mask, rows, embeds, exact=True:
                            check(mask, types.SimpleNamespace(size=rows.numel()), embeds, exact))


IMG = 7


def test_qwen_family_merge_raises_when_tokens_and_features_differ():
    import mlx.core as mx
    from unsloth_zoo.mlx.compile import _merge_special_token_features_only

    ids = mx.array([[1, IMG, IMG, IMG, 2]])
    embeds = mx.zeros((1, 5, 4))
    with pytest.raises(ValueError, match="tokens=3, features=2"):
        _merge_special_token_features_only(IMG, None, mx.ones((2, 4)), embeds, ids)
    merged = _merge_special_token_features_only(IMG, None, mx.ones((3, 4)), embeds, ids)
    assert np.asarray(merged)[0, :, 0].tolist() == [0, 1, 1, 1, 0]


def _uneven_idefics_pixels():
    # HF's Idefics3 processor pads row 0 (one image) to the batch's two with a zero image.
    pixels = np.zeros((2, 2, 3, 4, 4), np.float32)
    pixels[0, 0], pixels[1, 0], pixels[1, 1] = 1, 3, 4  # flat slot 1 is the padding
    return pixels


def test_idefics_collator_records_which_images_are_real():
    from unsloth_zoo.mlx.utils import _prepare_vlm_batch_for_compile

    def prepare(pixels):
        return _prepare_vlm_batch_for_compile({
            "input_ids": np.array([[IMG, 0, 0], [IMG, IMG, 0]], np.int32),
            "attention_mask": np.ones((2, 3), np.int32),
            "pixel_values": pixels,
        }, {"model_type": "idefics3", "image_token_index": IMG}, phase="content")

    assert prepare(_uneven_idefics_pixels())["_unsloth_static_vlm_metadata"] == {
        "real_image_indices": (0, 2, 3)}
    assert "_unsloth_static_vlm_metadata" not in prepare(np.ones((2, 1, 3, 4, 4), np.float32))


@pytest.fixture
def idefics_model(monkeypatch):
    import mlx.core as mx
    import unsloth_zoo.mlx.compile as mc
    from mlx_simulation import mlx_is_simulated

    if mlx_is_simulated():
        import mlx_vlm.models.base as base

        monkeypatch.setattr(base, "InputEmbeddingsFeatures", types.SimpleNamespace, raising=False)
    towered = []

    def vision_model(pixels, patch_attention_mask=None, output_hidden_states=False):
        towered.append(np.asarray(pixels)[:, 0, 0, 0].tolist())
        return (pixels[:, :1, 0, :1],)   # one feature row per image: its pixel value

    native = lambda *_args, **_kwargs: pytest.fail("native method")
    model_cls = type("Model", (), {"_prepare_inputs_for_multimodal": native, "get_input_embeddings": native})
    module = types.SimpleNamespace(Model=model_cls)
    monkeypatch.setattr(mc, "_try_import_module",
                        lambda name: module if name == "mlx_vlm.models.idefics3.idefics3" else None)
    monkeypatch.setattr(mc, "_PATCHED_ARCHES", set())
    monkeypatch.setattr(mc, "_PATCH_BINDINGS", set())
    mc._install_idefics_family_compile_patches()

    model = model_cls()
    model.config = types.SimpleNamespace(image_token_index=IMG, vision_config=types.SimpleNamespace(patch_size=2))
    model.language_model = types.SimpleNamespace(embed_tokens=lambda ids: mx.zeros((*ids.shape, 1)))
    model.vision_model = vision_model
    model.connector = lambda features: features
    model.towered = towered
    return model


def test_idefics_tower_drops_padding_images_before_the_merge(idefics_model):
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import _apply_static_vlm_metadata

    ids = mx.array([[IMG, 0, 0], [IMG, IMG, 0]])
    pixels = mx.array(_uneven_idefics_pixels())
    kwargs = {}
    _apply_static_vlm_metadata(idefics_model, {"_unsloth_static_vlm_metadata": {
        "real_image_indices": (0, 2, 3)}}, kwargs)

    for call_kwargs in ({}, kwargs):   # read on the host, or handed over for compile
        out = np.asarray(idefics_model.get_input_embeddings(ids, pixels, **call_kwargs).inputs_embeds)
        assert out[..., 0].tolist() == [[1, 0, 0], [3, 4, 0]]
    assert kwargs == {"real_image_indices": (0, 2, 3)}
    assert idefics_model.towered == [[1, 3, 4], [1, 3, 4]]


def test_idefics_merge_raises_when_tokens_and_features_differ(idefics_model):
    import mlx.core as mx

    with pytest.raises(ValueError, match="tokens=3, features=4"):
        idefics_model._prepare_inputs_for_multimodal(
            mx.ones((4, 1, 1)), mx.zeros((2, 3, 1)), mx.array([[IMG, 0, 0], [IMG, IMG, 0]]))
