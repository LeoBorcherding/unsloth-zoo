"""gpt-oss 4-bit with the BnB experts opted out, on a float16 GPU without grouped_mm:
patch_gpt_oss_linearized binds a forward onto the stock 3D GptOssExperts, and that
forward has to train and match the stock math. The square case is gpt-oss-20b's own
shape (hidden_size == intermediate_size), where down_proj's orientation is ambiguous.
On transformers 4 the stock forward, which takes v4's scattered routing weights, stays."""
from types import SimpleNamespace

import pytest
import torch

import unsloth_zoo.temporary_patches.gpt_oss as G

modeling_gpt_oss = pytest.importorskip("transformers.models.gpt_oss.modeling_gpt_oss")
from transformers import GptOssConfig


def _opt_out_on_float16(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MODEL_NAME", "unsloth/gpt-oss-20b_load_in_4bit_")
    monkeypatch.setenv("UNSLOTH_GPT_OSS_BNB4BIT_DISABLE", "1")
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    monkeypatch.setattr(G, "select_moe_backend_for_interleaved_gate_up", lambda: "native_torch")


@pytest.fixture
def linearized_loop(monkeypatch):
    if not G._is_transformers_v5():
        pytest.skip("the stock 3D experts take (tokens, top_k) weights from transformers 5")
    _opt_out_on_float16(monkeypatch)
    experts_cls = modeling_gpt_oss.GptOssExperts
    stock_forward = experts_cls.forward
    assert stock_forward.__module__ == modeling_gpt_oss.__name__
    try:
        G.patch_gpt_oss_linearized()
        assert experts_cls.forward is not stock_forward
        yield stock_forward
    finally:
        experts_cls.forward = stock_forward


@pytest.mark.parametrize("hidden, intermediate", [(16, 12), (16, 16)])
def test_stock_experts_train_and_match_stock_forward(linearized_loop, hidden, intermediate):
    stock_forward = linearized_loop
    config = GptOssConfig(
        hidden_size=hidden, intermediate_size=intermediate, num_local_experts=4,
        num_experts_per_tok=2, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8,
    )
    torch.manual_seed(0)
    experts = modeling_gpt_oss.GptOssExperts(config).train()
    for parameter in experts.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    router = modeling_gpt_oss.GptOssTopKRouter(config)
    torch.nn.init.normal_(router.weight)
    x = torch.randn(5, hidden)
    with torch.no_grad():
        _, scores, indices = router(x)

    out = experts(x, indices, scores)
    out.sum().backward()
    grads = [p.grad.clone() for p in experts.parameters()]
    experts.zero_grad()

    expected = stock_forward(experts, x, indices, scores)
    expected.sum().backward()
    torch.testing.assert_close(out, expected)
    for got, want in zip(grads, (p.grad for p in experts.parameters())):
        torch.testing.assert_close(got, want)


def test_transformers_4_keeps_the_stock_forward(monkeypatch):
    _opt_out_on_float16(monkeypatch)
    monkeypatch.setattr(G, "_is_transformers_v5", lambda: False)
    experts_cls = modeling_gpt_oss.GptOssExperts
    stock_forward = experts_cls.forward
    try:
        G.patch_gpt_oss_linearized()
        assert experts_cls.forward is stock_forward
    finally:
        experts_cls.forward = stock_forward


class _PackedStub(torch.nn.Parameter):
    """Stands in for bitsandbytes' Params4bit (no bnb on the test host): load_in_4bit packs
    the stock 3D experts into an (N, 1) uint8 tensor and keeps the real shape in quant_state."""


def test_packed_4bit_experts_are_dequantized(linearized_loop, monkeypatch):
    import unsloth_zoo.temporary_patches.moe_utils as M
    stock_forward = linearized_loop
    config = GptOssConfig(
        hidden_size=16, intermediate_size=12, num_local_experts=4, num_experts_per_tok=2,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
    )
    torch.manual_seed(0)
    experts = modeling_gpt_oss.GptOssExperts(config)
    for parameter in experts.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    router = modeling_gpt_oss.GptOssTopKRouter(config)
    x = torch.randn(5, 16)
    with torch.no_grad():
        _, scores, indices = router(x)
        expected = stock_forward(experts, x, indices, scores)

    def pack(name):
        real = getattr(experts, name).detach()
        packed = _PackedStub(torch.zeros(real.numel() // 2, 1, dtype=torch.uint8), requires_grad=False)
        packed.quant_state = SimpleNamespace(weight=real)
        setattr(experts, name, packed)

    pack("gate_up_proj"); pack("down_proj")
    monkeypatch.setattr(M, "HAS_BNB", True)
    monkeypatch.setattr(M, "Params4bit", _PackedStub)
    monkeypatch.setattr(M, "_get_base_weight", lambda p, target_dtype=None, **_: p.quant_state.weight.to(target_dtype))
    with torch.no_grad():
        out = experts(x, indices, scores)
    torch.testing.assert_close(out, expected)
