"""#838: shared-X Fast-LoRA Q/K/V correctness tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _deps():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    return torch


def _randomise_b(model):
    torch = _deps()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(param, std=0.2)


def _make_model(targets, *, ranks=None, bias=True):
    _deps()
    import torch.nn as nn
    from peft import LoraConfig, inject_adapter_in_model

    class TinyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=bias)
            self.k_proj = nn.Linear(8, 4, bias=bias)

            self.v_proj = nn.Linear(8, 4, bias=bias)

        def forward(self, x):
            return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = TinyAttention()

        def forward(self, x):
            return self.attn(x)

    model = Model()
    inject_adapter_in_model(
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            target_modules=list(targets),
            rank_pattern=ranks or {},
        ),
        model,
    )
    _randomise_b(model)
    return model


def _adapter_grads(model):
    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if "lora_" in name and param.grad is not None
    }


class TestMath:
    @pytest.mark.parametrize(
        "ranks",
        [
            (2, 0, 3),
            (2, 2, 2),
        ],
    )
    def test_float64_gradcheck_gqa_shapes(self, ranks):
        torch = _deps()
        from soup_cli.utils.fast_lora_qkv import _qkv_function

        torch.manual_seed(3)
        x = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
        wq = torch.randn(8, 8, dtype=torch.float64)
        wk = torch.randn(4, 8, dtype=torch.float64)
        wv = torch.randn(4, 8, dtype=torch.float64)
        bq = torch.randn(8, dtype=torch.float64)
        bk = torch.randn(4, dtype=torch.float64)
        bv = torch.randn(4, dtype=torch.float64)

        def pair(rank, out):
            if rank == 0:
                return x.new_empty(0), x.new_empty(0)
            return (
                torch.randn(rank, 8, dtype=torch.float64, requires_grad=True),
                torch.randn(out, rank, dtype=torch.float64, requires_grad=True),
            )

        aq, bql = pair(ranks[0], 8)
        ak, bkl = pair(ranks[1], 4)
        av, bvl = pair(ranks[2], 4)
        fn = _qkv_function()

        variables = [x]
        for a, b in ((aq, bql), (ak, bkl), (av, bvl)):
            if a.numel():
                variables += [a, b]

        def call(*vals):
            it = iter(vals)
            x_ = next(it)
            pairs = []
            for a, b in ((aq, bql), (ak, bkl), (av, bvl)):
                if a.numel():
                    pairs.append((next(it), next(it)))
                else:
                    pairs.append((a, b))
            (aq_, bq_), (ak_, bk_), (av_, bv_) = pairs
            q, k, v = fn.apply(
                x_, wq, bq, wk, bk, wv, bv,
                aq_, bq_, ak_, bk_, av_, bv_,
                1.2, 0.8, 1.7, None, None, None,
            )
            return torch.cat((q, k, v), dim=-1)

        assert torch.autograd.gradcheck(
            call, tuple(variables), eps=1e-6, atol=1e-5, rtol=1e-4
        )


    @pytest.mark.parametrize(
        "targets",
        [
            ("q_proj", "v_proj"),
            ("q_proj", "k_proj", "v_proj"),
            ("q_proj",),
            ("v_proj",),
        ],
    )
    def test_fp32_forward_and_backward_match_peft(self, targets):
        torch = _deps()
        from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

        torch.manual_seed(11)
        model = _make_model(targets, bias=True)
        x0 = torch.randn(2, 5, 8, requires_grad=True)
        ref = model(x0)
        ref_loss = sum(part.square().mean() for part in ref)
        ref_loss.backward()
        ref_x = x0.grad.detach().clone()
        ref_grads = _adapter_grads(model)
        for param in model.parameters():
            param.grad = None

        x1 = x0.detach().clone().requires_grad_(True)
        assert patch_fast_lora_qkv(model) == 1
        got = model(x1)
        sum(part.square().mean() for part in got).backward()

        for got_part, ref_part in zip(got, ref):
            torch.testing.assert_close(got_part, ref_part)

        torch.testing.assert_close(x1.grad, ref_x)
        got_grads = _adapter_grads(model)
        assert got_grads.keys() == ref_grads.keys()
        for name, grad in got_grads.items():
            torch.testing.assert_close(grad, ref_grads[name], msg=name)

    def test_unequal_ranks_and_gqa_match_peft(self):
        torch = _deps()
        from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

        torch.manual_seed(19)
        model = _make_model(
            ("q_proj", "k_proj", "v_proj"),
            ranks={"q_proj": 2, "k_proj": 3, "v_proj": 4},
        )
        actual = tuple(
            model.attn.__getattr__(name).lora_A["default"].weight.shape[0]
            for name in ("q_proj", "k_proj", "v_proj")
        )
        assert actual == (2, 3, 4)

        x0 = torch.randn(9, 8, requires_grad=True)
        ref = model(x0)
        sum(part.sum() for part in ref).backward()
        ref_x = x0.grad.detach().clone()
        ref_grads = _adapter_grads(model)

        for param in model.parameters():
            param.grad = None

        x1 = x0.detach().clone().requires_grad_(True)
        assert patch_fast_lora_qkv(model) == 1
        got = model(x1)
        sum(part.sum() for part in got).backward()
        for a, b in zip(got, ref):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(x1.grad, ref_x)
        for name, grad in _adapter_grads(model).items():
            torch.testing.assert_close(grad, ref_grads[name], msg=name)


class TestCoordinator:
    def test_patch_is_idempotent_reversible_and_cache_is_drained(self):
        _deps()
        from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv, unpatch_fast_lora_qkv

        model = _make_model(("q_proj", "v_proj"))
        had_instance = {
            name: "forward" in getattr(model.attn, name).__dict__
            for name in ("q_proj", "k_proj", "v_proj")
        }
        assert patch_fast_lora_qkv(model) == 1
        assert patch_fast_lora_qkv(model) == 0
        model(_deps().randn(2, 8))
        assert getattr(model.attn, "_soup_fast_lora_qkv_cache", None) is None
        hits, grad_fns = getattr(model.attn, "_soup_fast_lora_qkv_last_cache_hits")
        assert hits == 2
        assert grad_fns == ("_FastLoraQKVBackward",) * 3
        assert unpatch_fast_lora_qkv(model) == 1
        assert unpatch_fast_lora_qkv(model) == 0
        for name in ("q_proj", "k_proj", "v_proj"):
            assert ("forward" in getattr(model.attn, name).__dict__) is had_instance[name]


    def test_checkpoint_non_reentrant_matches_plain(self):
        torch = _deps()
        from torch.utils.checkpoint import checkpoint

        from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

        torch.manual_seed(23)
        model = _make_model(("q_proj", "v_proj"))
        x0 = torch.randn(3, 8, requires_grad=True)

        def objective(x):
            q, k, v = model(x)
            return q.square().mean() + k.square().mean() + v.square().mean()

        ref = checkpoint(objective, x0, use_reentrant=False)
        ref.backward()
        ref_x = x0.grad.detach().clone()
        ref_grads = _adapter_grads(model)
        for param in model.parameters():
            param.grad = None

        x1 = x0.detach().clone().requires_grad_(True)
        assert patch_fast_lora_qkv(model) == 1
        got = checkpoint(objective, x1, use_reentrant=False)
        got.backward()

        torch.testing.assert_close(got, ref)
        torch.testing.assert_close(x1.grad, ref_x)
        for name, grad in _adapter_grads(model).items():
            torch.testing.assert_close(grad, ref_grads[name], msg=name)


    def test_mps_bfloat16_qv_default_is_finite(self):
        torch = _deps()
        if not torch.backends.mps.is_available():
            pytest.skip("Apple MPS is required")
        from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

        model = _make_model(("q_proj", "v_proj")).to(
            device="mps", dtype=torch.bfloat16
        )
        assert patch_fast_lora_qkv(model) == 1
        x = torch.randn(4, 8, device="mps", dtype=torch.bfloat16, requires_grad=True)
        q, k, v = model(x)
        (q.float().sum() + k.float().sum() + v.float().sum()).backward()
        assert all(torch.isfinite(part).all() for part in (q, k, v))
        assert torch.isfinite(x.grad).all()


class TestFastPathIsActuallyTaken:
    @pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
    @pytest.mark.parametrize("shape", [(4, 8), (2, 3, 8)], ids=["2d", "3d"])
    @pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"], ids=["fp32", "bf16"])
    def test_all_three_outputs_come_from_the_fused_kernel(
        self, bias, shape, dtype_name
    ):
        torch = _deps()
        from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

        dtype = getattr(torch, dtype_name)
        model = _make_model(("q_proj", "v_proj"), bias=bias).to(dtype=dtype)
        x = torch.randn(*shape, dtype=dtype, requires_grad=True)

        before = model(x)
        assert all(type(part.grad_fn).__name__ != "_FastLoraQKVBackward" for part in before)

        assert patch_fast_lora_qkv(model) == 1
        out = model(x)
        assert [type(part.grad_fn).__name__ for part in out] == [
            "_FastLoraQKVBackward",
            "_FastLoraQKVBackward",
            "_FastLoraQKVBackward",
        ]
        hits, grad_fns = getattr(model.attn, "_soup_fast_lora_qkv_last_cache_hits")
        assert hits == 2
        assert grad_fns == ("_FastLoraQKVBackward",) * 3

    def test_non_contiguous_three_dimensional_input_stays_on_fast_path(self):
        torch = _deps()
        from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

        model = _make_model(("q_proj", "v_proj"), bias=False)
        base = torch.randn(3, 2, 8, requires_grad=True)
        x = base.transpose(0, 1)
        assert not x.is_contiguous()
        ref = model(x)
        assert patch_fast_lora_qkv(model) == 1
        got = model(x)
        for actual, expected in zip(got, ref):
            torch.testing.assert_close(actual, expected)
        assert all(type(part.grad_fn).__name__ == "_FastLoraQKVBackward" for part in got)

    @pytest.mark.gpu
    def test_nf4_qkv_outputs_are_from_the_fused_kernel(self):
        torch = _deps()
        bnb = pytest.importorskip("bitsandbytes")
        import torch.nn as nn
        from peft import LoraConfig, inject_adapter_in_model

        from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = bnb.nn.Linear4bit(
                    8, 8, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4"
                )
                self.k_proj = bnb.nn.Linear4bit(
                    8, 4, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4"
                )
                self.v_proj = bnb.nn.Linear4bit(
                    8, 4, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4"
                )

            def forward(self, x):
                return self.q_proj(x), self.k_proj(x), self.v_proj(x)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()

            def forward(self, x):
                return self.attn(x)

        model = Model().to("cuda")
        inject_adapter_in_model(
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                target_modules=["q_proj", "v_proj"],
            ),
            model,
        )
        _randomise_b(model)
        assert model.attn.q_proj.get_base_layer().weight.quant_state is not None
        x = torch.randn(2, 3, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        assert patch_fast_lora_qkv(model) == 1
        out = model(x)
        assert all(type(part.grad_fn).__name__ == "_FastLoraQKVBackward" for part in out)
        hits, _ = getattr(model.attn, "_soup_fast_lora_qkv_last_cache_hits")
        assert hits == 2


class TestRealLlamaAttention:
    def test_real_llama_gqa_forward_backward_and_cache_hits(self):
        torch = _deps()
        from peft import LoraConfig, inject_adapter_in_model
        from transformers import LlamaConfig, LlamaForCausalLM

        from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_bias=False,
        )
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config)
        inject_adapter_in_model(
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                target_modules=["q_proj", "v_proj"],
            ),
            model,
        )
        _randomise_b(model)

        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        ref = model(input_ids=input_ids, use_cache=False).logits
        ref.square().mean().backward()
        ref_grads = _adapter_grads(model)
        for param in model.parameters():
            param.grad = None

        assert patch_fast_lora_qkv(model) == 1
        got = model(input_ids=input_ids, use_cache=False).logits
        got.square().mean().backward()
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)
        for name, grad in _adapter_grads(model).items():
            torch.testing.assert_close(grad, ref_grads[name], rtol=1e-5, atol=1e-6, msg=name)

        attn = model.model.layers[0].self_attn
        assert getattr(attn, "_soup_fast_lora_qkv_cache", None) is None
        hits, grad_fns = getattr(attn, "_soup_fast_lora_qkv_last_cache_hits")
        assert hits == 2
        assert grad_fns == ("_FastLoraQKVBackward",) * 3
