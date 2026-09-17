"""Regression tests for #794: advertised PEFT variants reach live trainer paths."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError


def _config(**lora):
    from soup_cli.config.schema import SoupConfig

    return SoupConfig(
        base="tiny-local-llama",
        task="sft",
        data={"train": "train.jsonl"},
        training={
            "quantization": "none",
            "lora": {
                "r": 8,
                "alpha": 16,
                "dropout": 0.0,
                "target_modules": ["q_proj", "v_proj"],
                **lora,
            },
        },
    )


def _model():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            pad_token_id=0,
        )
    ).to(torch.float32)


def _setup_sft(config, model):
    from soup_cli.trainer.sft import SFTTrainerWrapper

    wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
    wrapper.config = config
    wrapper.device = "cpu"
    wrapper._trust_remote_code = False
    wrapper.model = None
    wrapper.tokenizer = None
    tokenizer = SimpleNamespace(pad_token=None, eos_token="</s>", chat_template=None)
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model),
    ):
        wrapper._setup_transformers(config, config.training)
    return wrapper.model


def test_pissa_changes_base_weight_through_real_sft_setup() -> None:
    import torch

    model = _model()
    before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    wrapped = _setup_sft(_config(init_strategy="pissa"), model)
    q_proj = wrapped.base_model.model.model.layers[0].self_attn.q_proj

    assert wrapped.peft_config["default"].init_lora_weights == "pissa"
    assert not torch.equal(q_proj.base_layer.weight.detach(), before)


def test_vera_builds_vera_layers_not_lora_layers_through_real_sft_setup() -> None:
    wrapped = _setup_sft(_config(use_vera=True, r=16), _model())
    q_proj = wrapped.base_model.model.model.layers[0].self_attn.q_proj

    assert type(wrapped.peft_config["default"]).__name__ == "VeraConfig"
    assert type(q_proj).__module__.startswith("peft.tuners.vera")
    assert hasattr(q_proj, "vera_lambda_b")
    assert not hasattr(q_proj, "lora_A")


def test_loftq_reaches_shared_peft_constructor_with_config() -> None:
    from peft import LoraConfig

    from soup_cli.config.schema import LoraConfig as SoupLoraConfig
    from soup_cli.utils.peft_wiring import build_lora_config

    config = build_lora_config(
        SoupLoraConfig(init_strategy="loftq", loftq_iter=3, loftq_bits=8),
        target_modules=["q_proj"],
        task_type="CAUSAL_LM",
    )

    assert isinstance(config, LoraConfig)
    assert config.init_lora_weights == "loftq"
    assert config.loftq_config == {"loftq_bits": 8, "loftq_iter": 3}


@pytest.mark.parametrize("backend", ["mlx", "unsloth"])
@pytest.mark.parametrize("lora", [{"init_strategy": "pissa"}, {"use_vera": True}])
def test_unwired_backends_refuse_peft_variants(backend: str, lora: dict) -> None:
    from soup_cli.config.schema import SoupConfig

    with pytest.raises(ValidationError, match="requires backend='transformers'"):
        SoupConfig(
            base="tiny-local-llama",
            task="sft",
            backend=backend,
            data={"train": "train.jsonl"},
            training={"quantization": "none", "lora": lora},
        )


def test_loftq_refuses_prequantized_base() -> None:
    from soup_cli.config.schema import SoupConfig

    with pytest.raises(ValidationError, match="LoftQ quantizes the base model"):
        SoupConfig(
            base="tiny-local-llama",
            task="sft",
            data={"train": "train.jsonl"},
            training={"lora": {"init_strategy": "loftq"}},
        )


def test_pissa_refuses_prequantized_base() -> None:
    from soup_cli.config.schema import SoupConfig

    with pytest.raises(ValidationError, match="PiSSA computes an SVD"):
        SoupConfig(
            base="tiny-local-llama",
            task="sft",
            data={"train": "train.jsonl"},
            training={"lora": {"init_strategy": "pissa"}},
        )
