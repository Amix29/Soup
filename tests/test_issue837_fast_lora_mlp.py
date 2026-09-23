"""#837: fused Fast-LoRA SwiGLU MLP correctness tests."""

from __future__ import annotations

from types import SimpleNamespace

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


def _make_model(
    targets, *, dropout=0.0, act="silu", module_act=None, ranks=None, bias=True
):
    _deps()
    import torch.nn as nn
    import torch.nn.functional as functional
    from peft import LoraConfig, inject_adapter_in_model

    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(8, 12, bias=bias)
            self.up_proj = nn.Linear(8, 12, bias=bias)
            self.down_proj = nn.Linear(12, 8, bias=bias)
            activation = module_act or act
            self.act_fn = nn.SiLU() if activation in {"silu", "swish"} else nn.GELU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_act=act)
            self.mlp = TinyMLP()

        def forward(self, x):
            return self.mlp(x)

    model = Model()
    config = LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(targets),
        rank_pattern=ranks or {},
    )
    inject_adapter_in_model(config, model)
    _randomise_b(model)
    return model


def _adapter_grads(model):
    return {

        name: p.grad.detach().clone()
        for name, p in model.named_parameters()
        if "lora_" in name and p.grad is not None
    }


class TestMath:
    def test_float64_gradcheck_all_six_adapter_matrices_unequal_ranks(self):
        torch = _deps()
        from soup_cli.utils.fast_lora_mlp import _mlp_function

        torch.manual_seed(3)
        x = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
        wg = torch.randn(12, 8, dtype=torch.float64)
        wu = torch.randn(12, 8, dtype=torch.float64)
        wd = torch.randn(8, 12, dtype=torch.float64)
        bg = torch.randn(12, dtype=torch.float64)
        bu = torch.randn(12, dtype=torch.float64)
        bd = torch.randn(8, dtype=torch.float64)
        ag = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
        bgl = torch.randn(12, 2, dtype=torch.float64, requires_grad=True)
        au = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
        bul = torch.randn(12, 3, dtype=torch.float64, requires_grad=True)
        ad = torch.randn(4, 12, dtype=torch.float64, requires_grad=True)
        bdl = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
        fn = _mlp_function()

        def call(x_, ag_, bgl_, au_, bul_, ad_, bdl_):
            return fn.apply(
                x_, wg, bg, wu, bu, wd, bd,
                ag_, bgl_, au_, bul_, ad_, bdl_,
                1.7, 0.8, 2.1, None, None, None,
            )

        assert torch.autograd.gradcheck(
            call, (x, ag, bgl, au, bul, ad, bdl), eps=1e-6, atol=1e-5, rtol=1e-4
        )

    @pytest.mark.parametrize(
        "targets",
        [
            ("gate_proj", "up_proj", "down_proj"),
            ("gate_proj",),
            ("up_proj",),
            ("down_proj",),
            ("gate_proj", "down_proj"),
        ],
    )
    def test_fp32_forward_and_backward_match_peft_for_partial_adapters(self, targets):
        torch = _deps()
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        torch.manual_seed(11)
        model = _make_model(targets)
        x_ref = torch.randn(2, 5, 8, requires_grad=True)
        ref = model(x_ref)
        loss_ref = ref.square().mean()
        loss_ref.backward()
        ref_x = x_ref.grad.detach().clone()

        ref_grads = _adapter_grads(model)

        for p in model.parameters():
            p.grad = None
        x_fast = x_ref.detach().clone().requires_grad_(True)
        assert patch_fast_lora_mlp(model) == 1
        out = model(x_fast)
        out.square().mean().backward()

        torch.testing.assert_close(out, ref)
        torch.testing.assert_close(x_fast.grad, ref_x)
        got_grads = _adapter_grads(model)
        assert got_grads.keys() == ref_grads.keys()
        for name in ref_grads:
            torch.testing.assert_close(got_grads[name], ref_grads[name], msg=name)

    def test_unequal_rank_pattern_matches_peft(self):
        torch = _deps()
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        torch.manual_seed(17)
        model = _make_model(
            ("gate_proj", "up_proj", "down_proj"),
            ranks={"gate_proj": 2, "up_proj": 3, "down_proj": 4},
        )
        ranks = tuple(
            model.mlp.__getattr__(name).lora_A["default"].weight.shape[0]
            for name in ("gate_proj", "up_proj", "down_proj")
        )
        assert ranks == (2, 3, 4)

        x0 = torch.randn(7, 8, requires_grad=True)
        ref = model(x0)
        ref.sum().backward()
        ref_x = x0.grad.detach().clone()
        ref_grads = _adapter_grads(model)
        for p in model.parameters():
            p.grad = None

        x1 = x0.detach().clone().requires_grad_(True)
        assert patch_fast_lora_mlp(model) == 1
        got = model(x1)
        got.sum().backward()
        torch.testing.assert_close(got, ref)
        torch.testing.assert_close(x1.grad, ref_x)
        for name, grad in _adapter_grads(model).items():
            torch.testing.assert_close(grad, ref_grads[name], msg=name)


class TestPatching:
    def test_patch_is_idempotent_and_reversible(self):
        _deps()
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp, unpatch_fast_lora_mlp

        model = _make_model(("gate_proj", "up_proj", "down_proj"))
        assert patch_fast_lora_mlp(model) == 1
        assert patch_fast_lora_mlp(model) == 0
        assert unpatch_fast_lora_mlp(model) == 1
        assert unpatch_fast_lora_mlp(model) == 0

    def test_non_silu_activation_falls_back_without_patch(self):
        _deps()
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        model = _make_model(("gate_proj",), act="gelu")
        assert patch_fast_lora_mlp(model) == 0

    def test_nonzero_dropout_delegates_to_original_mlp(self):
        torch = _deps()
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        model = _make_model(("gate_proj", "up_proj", "down_proj"), dropout=0.25)
        model.train()
        calls = []
        original = model.mlp.forward

        def spy(x):
            calls.append(True)
            return original(x)

        model.mlp.forward = spy
        assert patch_fast_lora_mlp(model) == 1
        torch.manual_seed(9)
        model(torch.randn(2, 8))
        assert calls == [True]

    def test_saturation_is_finite_on_mps_bfloat16(self):
        torch = _deps()
        if not torch.backends.mps.is_available():
            pytest.skip("Apple MPS is required")
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        model = _make_model(("gate_proj", "up_proj", "down_proj")).to(
            device="mps", dtype=torch.bfloat16
        )
        with torch.no_grad():
            model.mlp.gate_proj.get_base_layer().weight.fill_(2.5)
        assert patch_fast_lora_mlp(model) == 1
        x = torch.ones(3, 8, device="mps", dtype=torch.bfloat16, requires_grad=True)
        y = model(x)
        y.float().sum().backward()
        assert torch.isfinite(y).all()
        assert torch.isfinite(x.grad).all()


class TestFastPathAndScope:
    @pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
    @pytest.mark.parametrize("shape", [(4, 8), (2, 5, 8)], ids=["2d", "3d"])
    @pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"], ids=["fp32", "bf16"])
    def test_kernel_is_taken_across_bias_rank_and_dtype(self, bias, shape, dtype_name):
        torch = _deps()
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        dtype = getattr(torch, dtype_name)
        model = _make_model(
            ("gate_proj", "up_proj", "down_proj"), bias=bias
        ).to(dtype=dtype)
        x = torch.randn(*shape, dtype=dtype, requires_grad=True)
        before = model(x)
        assert type(before.grad_fn).__name__ != "_FastLoraSwiGLUBackward"

        assert patch_fast_lora_mlp(model) == 1
        out = model(x)
        assert type(out.grad_fn).__name__ == "_FastLoraSwiGLUBackward"

    def test_module_gelu_is_refused_even_when_parent_config_says_silu(self):
        _deps()
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        model = _make_model(("gate_proj",), act="silu", module_act="gelu")
        assert patch_fast_lora_mlp(model) == 0

    def test_moe_expert_path_is_out_of_scope(self):
        _deps()
        import torch.nn as nn
        from peft import LoraConfig, inject_adapter_in_model

        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        class Expert(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(8, 12)
                self.up_proj = nn.Linear(8, 12)
                self.down_proj = nn.Linear(12, 8)
                self.act_fn = nn.SiLU()

            def forward(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_act="silu")
                self.experts = nn.ModuleList([Expert()])

        model = Model()
        inject_adapter_in_model(
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                target_modules=["gate_proj", "up_proj", "down_proj"],
            ),
            model,
        )
        assert patch_fast_lora_mlp(model) == 0

    @pytest.mark.gpu
    def test_nf4_kernel_is_taken(self):
        torch = _deps()
        bnb = pytest.importorskip("bitsandbytes")
        import torch.nn as nn
        from peft import LoraConfig, inject_adapter_in_model

        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        class TinyNF4MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = bnb.nn.Linear4bit(
                    8, 12, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4"
                )
                self.up_proj = bnb.nn.Linear4bit(
                    8, 12, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4"
                )
                self.down_proj = bnb.nn.Linear4bit(
                    12, 8, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4"
                )
                self.act_fn = nn.SiLU()

            def forward(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_act="silu")
                self.mlp = TinyNF4MLP()

            def forward(self, x):
                return self.mlp(x)

        model = Model().to("cuda")
        inject_adapter_in_model(
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                target_modules=["gate_proj", "up_proj", "down_proj"],
            ),
            model,
        )
        _randomise_b(model)
        assert model.mlp.gate_proj.get_base_layer().weight.quant_state is not None
        x = torch.randn(2, 3, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        before = model(x)
        assert type(before.grad_fn).__name__ != "_FastLoraSwiGLUBackward"
        assert patch_fast_lora_mlp(model) == 1
        out = model(x)
        assert type(out.grad_fn).__name__ == "_FastLoraSwiGLUBackward"


class TestRealLlamaAndSavedBytes:
    def test_real_llama_mlp_matches_peft(self):
        torch = _deps()
        from peft import LoraConfig, inject_adapter_in_model
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaMLP

        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = LlamaConfig(
                    hidden_size=8,
                    intermediate_size=12,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    hidden_act="silu",
                )
                self.mlp = LlamaMLP(self.config)

            def forward(self, x):
                return self.mlp(x)

        torch.manual_seed(31)
        model = Wrapper()
        inject_adapter_in_model(
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,

                bias="none",
                target_modules=["gate_proj", "up_proj", "down_proj"],
            ),
            model,
        )
        _randomise_b(model)
        x0 = torch.randn(2, 5, 8, requires_grad=True)
        ref = model(x0)
        ref.square().mean().backward()
        ref_x = x0.grad.detach().clone()
        ref_grads = _adapter_grads(model)
        for param in model.parameters():
            param.grad = None

        x1 = x0.detach().clone().requires_grad_(True)
        assert patch_fast_lora_mlp(model) == 1
        got = model(x1)
        got.square().mean().backward()
        torch.testing.assert_close(got, ref)
        torch.testing.assert_close(x1.grad, ref_x)
        for name, grad in _adapter_grads(model).items():
            torch.testing.assert_close(grad, ref_grads[name], msg=name)

    def test_saved_tensor_bytes_are_lower_than_peft(self):
        torch = _deps()
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        def saved_bytes(model):
            seen = set()
            total = 0

            def pack(tensor):
                nonlocal total
                key = (
                    tensor.untyped_storage().data_ptr(),
                    tensor.storage_offset(),
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                )
                if key not in seen:
                    seen.add(key)
                    total += tensor.numel() * tensor.element_size()
                return tensor

            x = torch.randn(2, 64, 8, requires_grad=True)
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                model(x).sum().backward()
            return total

        torch.manual_seed(0)
        model = _make_model(("gate_proj", "up_proj", "down_proj"))
        plain = saved_bytes(model)
        for param in model.parameters():
            param.grad = None
        assert patch_fast_lora_mlp(model) == 1
        fast = saved_bytes(model)

        assert 0 < fast < plain
        print(f"#837 saved bytes: peft={plain} fast={fast}")
