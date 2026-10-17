"""A merge that grew the vocab and then failed mid-shard must not leave its resized
tensors behind for the next _merge_and_overwrite_lora call in the same process (Unsloth
Studio's backend and notebooks retry exports in one process)."""
from collections import defaultdict

import pytest
import torch
from safetensors.torch import load_file, save_file

import unsloth_zoo.saving_utils as SU

EMBED = "model.embed_tokens.weight"
Q_PROJ = "model.layers.0.self_attn.q_proj.weight"


def _merge(tmp_path, filename, lora_weights):
    return SU._merge_and_overwrite_lora(
        save_directory = str(tmp_path),
        filename = filename,
        lora_weights = defaultdict(lambda: None, lora_weights),
        output_dtype = torch.float32,
        model_class_name = "LlamaForCausalLM",
    )


def test_a_failed_resized_merge_does_not_leak_into_the_next_shard(tmp_path, monkeypatch):
    torch.manual_seed(0)
    save_file({EMBED: torch.randn(4, 8), Q_PROJ: torch.randn(8, 8)}, str(tmp_path / "a.safetensors"))
    second = {"model.norm.weight": torch.randn(8)}
    save_file(second, str(tmp_path / "b.safetensors"))

    # modules_to_save embedding with two new rows: no longer fits its byte slot.
    grown = torch.nn.Embedding(6, 8)
    lora_weights = {"model.embed_tokens": SU.LoraStats(grown, None, None, None)}

    real_write = SU._write_tensor_direct_torch

    def write_then_fail(mm, header_metadata, length_of_header, key, W, dtype):
        if key == Q_PROJ:
            raise OSError("simulated mmap write error")
        return real_write(mm, header_metadata, length_of_header, key, W, dtype)

    monkeypatch.setattr(SU, "_write_tensor_direct_torch", write_then_fail)
    with pytest.raises(RuntimeError, match = "simulated mmap write error"):
        _merge(tmp_path, "a.safetensors", lora_weights)
    monkeypatch.setattr(SU, "_write_tensor_direct_torch", real_write)

    _merge(tmp_path, "b.safetensors", {})

    merged = load_file(str(tmp_path / "b.safetensors"))
    assert set(merged) == set(second)
    torch.testing.assert_close(merged["model.norm.weight"], second["model.norm.weight"])
    assert not getattr(SU._merge_and_overwrite_lora, "_resized", None)
