"""#796: SFT gradient-checkpointing tiers must produce different execution."""

from __future__ import annotations

import inspect

import pytest


def _tiny_llama():
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    return transformers.LlamaForCausalLM(config)


def test_medium_uses_transformers_every_n_layers_and_skips_half_the_blocks() -> None:
    from soup_cli.utils.gradient_ckpt import plan_gradient_checkpointing

    model = _tiny_llama()
    plan = plan_gradient_checkpointing(model, "medium", gpu_memory_gb=40)

    assert plan.granularity == "medium"
    assert plan.kwargs == {
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {
            "use_reentrant": False,
            "every_n_layers": 2,
        },
    }

    model.gradient_checkpointing_enable(
        every_n_layers=plan.kwargs["gradient_checkpointing_kwargs"]["every_n_layers"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    active = [layer.gradient_checkpointing for layer in model.model.layers]
    assert active == [True, False, True, False]


def test_full_and_medium_are_observably_different_on_a_tiny_model() -> None:
    from soup_cli.utils.gradient_ckpt import plan_gradient_checkpointing

    full_model = _tiny_llama()
    medium_model = _tiny_llama()
    full = plan_gradient_checkpointing(full_model, "full")
    medium = plan_gradient_checkpointing(medium_model, "medium")

    full_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=full.kwargs["gradient_checkpointing_kwargs"],
    )
    medium_kwargs = dict(medium.kwargs["gradient_checkpointing_kwargs"])
    every_n_layers = medium_kwargs.pop("every_n_layers")
    medium_model.gradient_checkpointing_enable(
        every_n_layers=every_n_layers,
        gradient_checkpointing_kwargs=medium_kwargs,
    )

    assert sum(layer.gradient_checkpointing for layer in full_model.model.layers) == 4
    assert sum(layer.gradient_checkpointing for layer in medium_model.model.layers) == 2


def test_selective_wraps_one_attention_module_per_block_without_hf_checkpointing() -> None:
    from soup_cli.utils.gradient_ckpt import plan_gradient_checkpointing

    model = _tiny_llama()
    plan = plan_gradient_checkpointing(model, "selective")

    assert plan.granularity == "selective"
    assert plan.kwargs == {"gradient_checkpointing": False}
    assert plan.hooked_modules == 4
    for layer in model.model.layers:
        assert layer.self_attn.forward.__name__ == "_checkpointed_forward"
        assert layer.self_attn.q_proj.forward.__name__ != "_checkpointed_forward"
        assert layer.self_attn.k_proj.forward.__name__ != "_checkpointed_forward"
        assert layer.self_attn.v_proj.forward.__name__ != "_checkpointed_forward"
        assert layer.self_attn.o_proj.forward.__name__ != "_checkpointed_forward"


def test_selective_tiny_model_completes_forward_and_backward() -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.utils.gradient_ckpt import plan_gradient_checkpointing

    model = _tiny_llama()
    plan = plan_gradient_checkpointing(model, "selective")
    input_ids = torch.tensor([[1, 2, 3, 4]])

    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()

    assert plan.hooked_modules == 4
    assert loss.isfinite().item() is True
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None


def test_selective_falls_back_truthfully_when_architecture_has_no_attention_child() -> None:
    from soup_cli.utils.gradient_ckpt import plan_gradient_checkpointing

    class Block:
        def named_children(self):
            return iter(())

    class Model:
        def named_modules(self):
            yield "model.layers.0", Block()

    plan = plan_gradient_checkpointing(Model(), "selective")

    assert plan.granularity == "full"
    assert plan.kwargs["gradient_checkpointing"] is True
    assert plan.hooked_modules == 0
    assert "full fallback" in plan.description


def test_sft_setup_consumes_the_truthful_plan() -> None:
    from soup_cli.trainer.sft import SFTTrainerWrapper

    source = inspect.getsource(SFTTrainerWrapper.setup)
    assert "plan_gradient_checkpointing(" in source
    assert "training_kwargs.update(ckpt_plan.kwargs)" in source
    assert "ckpt_plan.description" in source
