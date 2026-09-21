"""Checkpoint-visible NF4 fused-forward groundwork (#842, folding #841)."""

from __future__ import annotations

import pytest

from soup_cli.utils.layer_stream_runtime import checkpoint_visible_nf4_linear

torch = pytest.importorskip("torch")
bnb_functional = pytest.importorskip("bitsandbytes.functional")


@pytest.mark.parametrize("nested", [False, True])
def test_fused_forward_and_input_gradient_match_dequant_linear(nested):
    torch.manual_seed(17)
    weight = torch.randn(16, 16)
    packed, state = bnb_functional.quantize_4bit(
        weight, quant_type="nf4", compress_statistics=nested
    )
    x = torch.randn(3, 16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)

    actual = checkpoint_visible_nf4_linear(x, packed, state)
    dense = bnb_functional.dequantize_4bit(packed, state).to(reference_x.dtype)
    expected = torch.nn.functional.linear(reference_x, dense)

    assert torch.equal(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert torch.equal(x.grad, reference_x.grad)


def _non_nested_state(absmax, template):
    return bnb_functional.QuantState(
        absmax=absmax,
        shape=template.shape,
        dtype=template.dtype,
        blocksize=template.blocksize,
        quant_type=template.quant_type,
    )


def test_save_for_backward_makes_pool_recycling_recompute_and_preserve_gradients():
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(23)
    weight0 = torch.randn(16, 16)
    weight1 = torch.randn(16, 16)
    packed0, state0 = bnb_functional.quantize_4bit(weight0, quant_type="nf4")
    packed1, state1 = bnb_functional.quantize_4bit(weight1, quant_type="nf4")

    shared_packed = packed0.clone()
    shared_absmax = state0.absmax.clone()
    shared_state = _non_nested_state(shared_absmax, state0)
    calls = {"n": 0}

    def streamed_body(value):
        calls["n"] += 1

        with torch.no_grad():
            shared_packed.copy_(packed0)
            shared_absmax.copy_(state0.absmax)
        return checkpoint_visible_nf4_linear(value, shared_packed, shared_state)

    x = torch.randn(3, 16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    output = checkpoint(streamed_body, x, use_reentrant=False)

    # Recycle the pool slot to another layer before backward.
    with torch.no_grad():
        shared_packed.copy_(packed1)
        shared_absmax.copy_(state1.absmax)

    output.square().sum().backward()
    dense0 = bnb_functional.dequantize_4bit(packed0, state0).to(reference_x.dtype)
    reference = torch.nn.functional.linear(reference_x, dense0)
    reference.square().sum().backward()

    assert calls["n"] == 2, "checkpoint did not recompute the custom NF4 op"
    assert torch.equal(x.grad, reference_x.grad)


def test_plain_ctx_control_is_detectably_wrong_after_pool_recycling():
    """Negative control: reproduces the #331 lifetime bug on a tiny CPU fixture."""
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(29)
    weight0 = torch.randn(16, 16)
    weight1 = torch.randn(16, 16)

    packed0, state0 = bnb_functional.quantize_4bit(weight0, quant_type="nf4")
    packed1, state1 = bnb_functional.quantize_4bit(weight1, quant_type="nf4")
    shared_packed = packed0.clone()
    shared_absmax = state0.absmax.clone()
    shared_state = _non_nested_state(shared_absmax, state0)

    class PlainCtx4Bit(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, packed):
            ctx.packed = packed
            ctx.state = shared_state
            return torch.ops.bitsandbytes.gemm_4bit.default(
                value,
                packed,
                shared_state.shape,
                shared_state.absmax,
                shared_state.blocksize,
                shared_state.quant_type,
            )

        @staticmethod
        def backward(ctx, grad_output):
            dense = bnb_functional.dequantize_4bit(ctx.packed, ctx.state).to(
                grad_output.dtype
            )
            return torch.matmul(grad_output, dense), None

    calls = {"n": 0}

    def broken_body(value):
        calls["n"] += 1
        with torch.no_grad():
            shared_packed.copy_(packed0)
            shared_absmax.copy_(state0.absmax)
        return PlainCtx4Bit.apply(value, shared_packed)

    x = torch.randn(3, 16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    output = checkpoint(broken_body, x, use_reentrant=False)
    with torch.no_grad():
        shared_packed.copy_(packed1)
        shared_absmax.copy_(state1.absmax)
    output.square().sum().backward()

    dense0 = bnb_functional.dequantize_4bit(packed0, state0).to(reference_x.dtype)
    reference = torch.nn.functional.linear(reference_x, dense0)
    reference.square().sum().backward()

    assert calls["n"] == 1, "negative control unexpectedly became checkpoint-visible"
    assert not torch.equal(x.grad, reference_x.grad)
    assert (x.grad - reference_x.grad).abs().max().item() > 1e-3


def test_production_capability_gate_keeps_cpu_on_the_old_path():
    from soup_cli.utils.layer_stream_runtime import _can_use_checkpoint_visible_nf4_gemm

    packed, state = bnb_functional.quantize_4bit(torch.randn(16, 16), quant_type="nf4")
    assert packed.device.type == "cpu"
    assert _can_use_checkpoint_visible_nf4_gemm(torch.randn(2, 16), state) is False
