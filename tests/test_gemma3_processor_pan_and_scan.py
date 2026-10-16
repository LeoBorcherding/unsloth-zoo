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

"""Patched Gemma3Processor.__call__: pan-and-scan crop counts per image (not per row), image paths, and the
truncation check upstream runs after tokenizing."""
import pytest

pytest.importorskip("transformers")
import numpy as np
from PIL import Image

TINY = "trl-internal-testing/tiny-Gemma3ForConditionalGeneration"
PAN_AND_SCAN = dict(
    do_pan_and_scan = True, pan_and_scan_min_crop_size = 32,
    pan_and_scan_max_num_crops = 4, pan_and_scan_min_ratio_to_activate = 1.2,
)


@pytest.fixture
def proc():
    from transformers import AutoProcessor
    from unsloth_zoo.temporary_patches.gemma import patch_Gemma3Processor
    patch_Gemma3Processor()
    try:
        return AutoProcessor.from_pretrained(TINY)
    except OSError as e:  # Hub / network unavailable only
        pytest.skip(f"{TINY} not downloadable: {e}")


def _images_per_row(proc, out):
    return [int((np.array(r) == proc.image_token_id).sum()) // proc.image_seq_length for r in out["input_ids"]]


def test_group_num_crops_accepts_flat_and_grouped():
    from unsloth_zoo.temporary_patches.gemma import _group_num_crops
    rows = [["a", "b"], ["c"]]
    assert _group_num_crops([0, 4, 3], rows) == [[0, 4], [3]]       # transformers 5.x: one entry per image
    assert _group_num_crops([[0, 4], [3]], rows) == [[0, 4], [3]]   # transformers 4.5x: grouped per row
    assert _group_num_crops(2, rows) == [[2, 0], [0]]


def test_two_pan_and_scan_images_in_one_row(proc):
    sq, wide, tall = Image.new("RGB", (64, 64), 1), Image.new("RGB", (256, 64), 2), Image.new("RGB", (64, 256), 3)
    boi = proc.boi_token
    out = proc(images = [[sq, wide], [tall]], text = [f"a {boi} b {boi}", f"c {boi}"], padding = True, **PAN_AND_SCAN)
    alone = [
        len(proc(images = [[img]], text = [f"x {boi}"], **PAN_AND_SCAN)["pixel_values"])
        for img in (sq, wide, tall)
    ]
    # Every pixel_values row (original image or crop) needs its own image block in the text.
    assert sum(_images_per_row(proc, out)) == len(out["pixel_values"])
    assert _images_per_row(proc, out) == [alone[0] + alone[1], alone[2]]


def test_image_path(proc, tmp_path):
    path = tmp_path / "img.png"
    Image.new("RGB", (64, 64), 1).save(path)
    out = proc(images = [str(path)], text = f"x {proc.boi_token}")
    assert len(out["pixel_values"]) == 1


def test_text_passed_positionally_still_works(proc):
    out = proc("just some text")
    assert len(out["input_ids"]) == 1


def test_truncation_through_the_image_block_raises(proc):
    with pytest.raises(ValueError, match = "Mismatch in `image` token count"):
        proc(images = [Image.new("RGB", (64, 64), 1)], text = f"x {proc.boi_token}", truncation = True, max_length = 40)
