"""Native DoRA lifecycle on a tiny, local random Llama; no model downloads."""

import json
import math

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.llama import Model, ModelArgs  # noqa: E402
from mlx_lm.tuner.dora import DoRALinear  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from soup_cli.config.schema import SoupConfig  # noqa: E402
from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper  # noqa: E402


def _checkpoint(path, quantized=False):
    path.mkdir()
    mx.random.seed(7)
    config = dict(
        model_type="llama", hidden_size=64, num_hidden_layers=1, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=4, rms_norm_eps=1e-5,
        vocab_size=32, tie_word_embeddings=False,
    )
    model = Model(ModelArgs(**config))
    if quantized:
        nn.quantize(model, group_size=32, bits=4)
        config["quantization"] = {"group_size": 32, "bits": 4}
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (path / "config.json").write_text(json.dumps(config))
    vocab = {word: i for i, word in enumerate(
        ["<unk>", "<eos>", "<pad>", "user", "assistant", "alpha", "beta", "gamma"]
    )}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} "
        "<eos> {% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    )
    tokenizer.save_pretrained(path)
    return path


def _config(base, output, *, dora=True, quantized=False, targets="auto"):
    return SoupConfig(
        base=str(base), backend="mlx", output=str(output),
        data={"train": "unused.jsonl", "train_on_responses_only": False, "max_length": 64},
        training={
            "quantization": "4bit" if quantized else "none", "epochs": 1,
            "batch_size": 1, "gradient_accumulation_steps": 1, "lr": 0.005,
            "logging_steps": 1, "save_steps": 2, "scheduler": "constant",
            "warmup_ratio": 0, "weight_decay": 0,
            "lora": {"r": 4, "alpha": 8, "dropout": 0, "use_dora": dora,
                     "target_modules": targets},
        },
    )


def _rows():
    return {"train": [{"text": "alpha beta gamma alpha beta gamma"}] * 4}


def _selective_fixture(path, quantized):
    """Real local tokenizer: one BOS, explicit role headers, PAD=EOS, Unicode."""
    _checkpoint(path, quantized)
    words = ["<unk>", "<eos>", "<bos>", "user", "assistant", "alpha", "beta", "gamma",
             "system", "developer", "tool", "répété", "policy", "ignored"]
    raw = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="<unk>", bos_token="<bos>",
        eos_token="<eos>", pad_token="<eos>",
    )
    tokenizer.chat_template = (
        "{{ bos_token }} {% for message in messages %}"
        "{{ message['role'] }} {{ message['content'] }} <eos> {% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    tokenizer.save_pretrained(path)
    messages = [
        {"role": "system", "content": "policy", "train": True},
        {"role": "developer", "content": "ignored", "train": False},
        {"role": "user", "content": "répété", "train": True},
        {"role": "assistant", "content": "répété", "train": False},
        {"role": "tool", "content": "beta", "train": True},
        {"role": "assistant", "content": "alpha", "train": None},
        {"role": "assistant", "content": "gamma alpha"},
    ]
    # Hand-checked token positions, independent of every production label builder.
    ids = [2, 8, 12, 1, 9, 13, 1, 3, 11, 1, 4, 11, 1,
           10, 6, 1, 4, 5, 1, 4, 7, 5, 1]
    return messages, ids


@pytest.mark.parametrize("mode", ["train_field", "mask_history", "responses_only"])
@pytest.mark.parametrize("quantized", [False, True])
def test_native_selective_loss_gradients_public_training_and_reload(
    tmp_path, monkeypatch, mode, quantized,
):
    from mlx_lm.tuner.datasets import CacheDataset
    from typer.testing import CliRunner

    from soup_cli.cli import app
    from soup_cli.trainer.mlx_masking import MaskedChatDataset, masked_iterate_batches, masked_loss

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    base = tmp_path / "base"
    messages, ids = _selective_fixture(base, quantized)
    cfg = _config(base, tmp_path / "out", dora=False, quantized=quantized)
    cfg.data.format = "chatml"
    cfg.data.val_split = 0
    cfg.data.train_on_responses_only = mode != "train_field"
    cfg.data.mask_history = mode == "mask_history"
    cfg.data.train_on_messages_with_train_field = mode == "train_field"
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"messages": messages}) for _ in range(4)) + "\n")
    cfg.data.train = str(data)
    wrapper = MLXSFTTrainerWrapper(cfg)
    wrapper.setup({"train": [{"messages": messages}] * 4})
    wrapper._apply_lora(wrapper.model)
    wrapper.model.eval()
    dataset = MaskedChatDataset(
        [{"messages": messages}], wrapper.tokenizer,
        train_field=mode == "train_field", mask_history=mode == "mask_history", max_length=64,
    )
    tokens, mask = dataset.process(dataset[0])
    positions = {
        "train_field": [*range(4), 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        "mask_history": [20, 21, 22],
        "responses_only": [11, 12, 17, 18, 20, 21, 22],
    }[mode]
    assert tokens == ids
    assert mask == [int(i in positions) for i in range(len(ids))]
    batch, actual_mask = next(masked_iterate_batches(CacheDataset(dataset), 1, 64))
    assert batch.shape == actual_mask.shape
    assert actual_mask[0, len(ids):].sum().item() == 0
    assert mask[-1] == 1  # Real EOS stays supervised although PAD=EOS.

    expected_positions = [position - 1 for position in positions if position > 0]

    def oracle(model, batch, unused):
        logits = model(batch[:, :-1]).astype(mx.float32)
        chosen = logits[0, expected_positions]
        targets = mx.array([ids[position + 1] for position in expected_positions])
        losses = mx.logsumexp(chosen, axis=-1) - mx.take_along_axis(
            chosen, targets[:, None], axis=-1,
        ).squeeze(-1)
        return losses.mean(), mx.array(len(expected_positions))

    (actual, count), gradients = nn.value_and_grad(wrapper.model, masked_loss)(
        wrapper.model, batch, actual_mask,
    )
    (expected, _), reference = nn.value_and_grad(wrapper.model, oracle)(
        wrapper.model, batch, actual_mask,
    )
    assert count.item() == len(expected_positions)
    assert actual.item() == pytest.approx(expected.item(), abs=2e-6)
    reference = dict(tree_flatten(reference))
    assert any(mx.abs(value).sum().item() > 0 for _, value in tree_flatten(gradients))
    for name, value in tree_flatten(gradients):
        assert mx.allclose(value, reference[name], atol=2e-6, rtol=2e-5).item(), name

    # Perturb a target under controlled fixed logits, keeping causal context constant.
    logits = wrapper.model(batch[:, :-1]).astype(mx.float32)
    fixed = lambda inputs: logits  # noqa: E731
    original_loss = masked_loss(fixed, batch, actual_mask)[0].item()
    ignored = next(i for i in range(1, len(ids)) if i not in positions)
    changed = mx.array(batch)
    changed[0, ignored] = (ids[ignored] + 1) % 32
    assert masked_loss(fixed, changed, actual_mask)[0].item() == original_loss
    changed[0, positions[-2]] = (ids[positions[-2]] + 1) % 32
    assert abs(masked_loss(fixed, changed, actual_mask)[0].item() - original_loss) > 1e-5

    original, _ = load(str(base))
    frozen = dict(tree_flatten(original.parameters()))
    config_file = tmp_path / "config.json"
    config_file.write_text(cfg.model_dump_json())
    outcome = CliRunner().invoke(app, ["train", "--config", str(config_file), "--yes"])
    assert outcome.exit_code == 0, (outcome.output, repr(outcome.exception))
    restored, _ = load(str(base), adapter_path=str(tmp_path / "out"))
    restored.eval()
    trained = dict(tree_flatten(restored.parameters()))
    for name, value in frozen.items():
        parent, leaf = name.rsplit(".", 1)
        actual_name = f"{parent}.linear.{leaf}"
        assert mx.array_equal(trained.get(actual_name, trained.get(name)), value).item(), name
    weights = mx.load(str(tmp_path / "out" / "adapters.safetensors"))
    assert any(mx.abs(value).sum().item() > 0 for name, value in weights.items()
               if name.endswith("lora_b"))
    metadata = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
    assert metadata["response_token_mask"] is True
    assert metadata["mask_history"] is (mode == "mask_history")
    assert metadata["train_on_messages_with_train_field"] is (mode == "train_field")
    restored_weights = dict(tree_flatten(restored.parameters()))
    assert all(mx.array_equal(restored_weights[name], value).item()
               for name, value in weights.items())
    assert mx.abs(restored(mx.array([ids])) - original(mx.array([ids]))).max().item() > 1e-4
    again, _ = load(str(base), adapter_path=str(tmp_path / "out"))
    again.eval()
    assert mx.allclose(again(mx.array([ids])), restored(mx.array([ids])), atol=1e-5).item()


@pytest.mark.parametrize("quantized, checkpoint, accumulation", [
    (False, False, 1), (True, False, 1), (False, True, 2), (True, True, 2),
])
def test_native_training_reload_warm_start_and_fusion(
    tmp_path, quantized, checkpoint, accumulation, monkeypatch
):
    base = _checkpoint(tmp_path / "base", quantized)
    cfg = _config(base, tmp_path / "adapters", quantized=quantized)
    cfg.training.gradient_checkpointing = checkpoint
    cfg.training.gradient_accumulation_steps = accumulation
    wrapper = MLXSFTTrainerWrapper(cfg)
    wrapper.setup(_rows())
    original = dict(tree_flatten(wrapper.model.parameters()))
    initial = {}
    apply = wrapper._apply_lora

    def capture_initial(model):
        apply(model)
        initial.update(tree_flatten(model.trainable_parameters()))
        mx.eval(initial)

    monkeypatch.setattr(wrapper, "_apply_lora", capture_initial)
    result = wrapper.train()
    assert result["total_steps"] == 4
    assert math.isfinite(result["final_loss"])
    assert isinstance(wrapper.model.layers[0].self_attn.q_proj, DoRALinear)
    trainable = dict(tree_flatten(wrapper.model.trainable_parameters()))
    assert len(trainable) == 6
    assert {name.rsplit(".", 1)[-1] for name in trainable} == {"lora_a", "lora_b", "m"}
    assert all(not mx.array_equal(value, initial[name]).item()
               for name, value in trainable.items())
    from mlx_lm.tuner.trainer import default_loss

    (_, _), gradients = nn.value_and_grad(wrapper.model, default_loss)(
        wrapper.model, mx.array([[5, 6, 7, 5]]), mx.array([[0, 4]])
    )
    assert all(mx.isfinite(value).all().item() for _, value in tree_flatten(gradients))
    # Every frozen base tensor, including quantization scales/biases, is unchanged.
    trained = dict(tree_flatten(wrapper.model.parameters()))
    for name, value in original.items():
        parent, leaf = name.rsplit(".", 1)
        adapted_name = f"{parent}.linear.{leaf}"
        actual = trained[adapted_name] if adapted_name in trained else trained[name]
        assert mx.array_equal(actual, value).item(), name
    for projection in ("q_proj", "v_proj"):
        layer = getattr(wrapper.model.layers[0].self_attn, projection)
        assert not mx.array_equal(layer.lora_b, mx.zeros_like(layer.lora_b)).item()
        assert not mx.array_equal(
            layer.m, mx.linalg.norm(layer._dequantized_weight().astype(mx.float32), axis=1)
        ).item()

    tokens = mx.array([[5, 6, 7, 5]])
    wrapper.model.eval()
    expected = wrapper.model(tokens)
    metadata = json.loads((tmp_path / "adapters" / "adapter_config.json").read_text())
    assert metadata["fine_tune_type"] == "dora"
    assert metadata["lora_parameters"]["scale"] == 2
    restored, _ = load(str(base), adapter_path=str(tmp_path / "adapters"))
    restored.eval()
    assert isinstance(restored.layers[0].self_attn.q_proj, DoRALinear)
    assert mx.allclose(restored(tokens), expected, atol=1e-5, rtol=1e-5).item()

    resumed = MLXSFTTrainerWrapper(cfg)
    resumed.setup(_rows())
    resumed._apply_lora(resumed.model)
    resumed._load_checkpoint_weights(str(tmp_path / "adapters" / "adapters.safetensors"))
    resumed.model.eval()
    assert mx.allclose(resumed.model(tokens), expected, atol=1e-5, rtol=1e-5).item()

    # This is the operation used by mlx_lm.fuse --dequantize. Avoid requantization
    # error when asserting that the adapter's mathematical effect is preserved.
    restored.update_modules(tree_unflatten([
        (name, module.fuse(dequantize=True))
        for name, module in restored.named_modules() if isinstance(module, DoRALinear)
    ]))
    assert mx.allclose(restored(tokens), expected, atol=2e-5, rtol=2e-5).item()


@pytest.mark.parametrize("quantized", [False, True])
def test_public_cli_trains_a_reloadable_dora_adapter(tmp_path, quantized, monkeypatch):
    from typer.testing import CliRunner

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    base = _checkpoint(tmp_path / "base", quantized)
    cfg = _config(base, tmp_path / "out", quantized=quantized)
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"messages": [
        {"role": "user", "content": "alpha beta"},
        {"role": "assistant", "content": "gamma alpha beta"},
    ]}) for _ in range(4)) + "\n")
    cfg.data.train = str(data)
    cfg.data.format = "chatml"
    cfg.data.val_split = 0
    config = tmp_path / "soup.json"
    # The loader uses YAML, whose JSON subset keeps the serialized schema exact.
    config.write_text(cfg.model_dump_json())
    outcome = CliRunner().invoke(app, ["train", "--config", str(config), "--yes"])
    assert outcome.exit_code == 0, (outcome.output, repr(outcome.exception))
    restored, _ = load(str(base), adapter_path=str(tmp_path / "out"))
    assert isinstance(restored.layers[0].self_attn.q_proj, DoRALinear)
    assert mx.isfinite(restored(mx.array([[5, 6, 7]]))).all().item()
    resumed = CliRunner().invoke(
        app, ["train", "--config", str(config), "--yes", "--resume", "auto"]
    )
    assert resumed.exit_code == 0, (resumed.output, repr(resumed.exception))
    assert "loading checkpoint weights" in resumed.output
    reloaded, _ = load(str(base), adapter_path=str(tmp_path / "out"))
    assert isinstance(reloaded.layers[0].self_attn.q_proj, DoRALinear)
    assert mx.isfinite(reloaded(mx.array([[5, 6, 7]]))).all().item()


@pytest.mark.parametrize("change", ["lora", "rank", "targets", "missing", "extra"])
def test_dora_refuses_incompatible_warm_start_before_loading(tmp_path, change):
    base = _checkpoint(tmp_path / "base")
    cfg = _config(base, tmp_path / "out")
    target = MLXSFTTrainerWrapper(cfg)
    target.setup(_rows())
    target._apply_lora(target.model)
    before = dict(tree_flatten(target.model.trainable_parameters()))
    source_cfg = _config(base, tmp_path / "other", dora=change != "lora")
    if change == "rank":
        source_cfg.training.lora.r = 2
    if change == "targets":
        source_cfg.training.lora.target_modules = ["self_attn.k_proj"]
    source = MLXSFTTrainerWrapper(source_cfg)
    source.setup(_rows())
    source._apply_lora(source.model)
    weights = dict(tree_flatten(source.model.trainable_parameters()))
    if change == "missing":
        weights.pop(next(iter(weights)))
    if change == "extra":
        weights["unexpected.m"] = mx.ones((1,))
    checkpoint = tmp_path / "incompatible.safetensors"
    mx.save_safetensors(str(checkpoint), weights)
    with pytest.raises(ValueError, match="[Dd][Oo][Rr][Aa].*incompatible"):
        target._load_checkpoint_weights(str(checkpoint))
    after = dict(tree_flatten(target.model.trainable_parameters()))
    assert all(mx.array_equal(value, after[name]).item() for name, value in before.items())


@pytest.mark.parametrize("invalid", ["missing", "switch", "embedding", "8bit"])
def test_unsupported_targets_fail_before_any_conversion(tmp_path, invalid):
    from mlx_lm.models.switch_layers import SwitchLinear

    base = _checkpoint(tmp_path / "base")
    targets = ["self_attn.q_proj", "self_attn.v_proj"]
    if invalid == "missing":
        targets.append("self_attn.typo")
    wrapper = MLXSFTTrainerWrapper(_config(base, tmp_path / "out", targets=targets))
    wrapper.setup(_rows())
    attention = wrapper.model.layers[0].self_attn
    if invalid == "switch":
        attention.v_proj = SwitchLinear(64, 64, 2)
    elif invalid == "embedding":
        attention.v_proj = nn.Embedding(32, 64)
    elif invalid == "8bit":
        attention.v_proj = nn.QuantizedLinear.from_linear(attention.v_proj, group_size=32, bits=8)
    before = dict(tree_flatten(wrapper.model.trainable_parameters()))
    with pytest.raises(ValueError, match="MLX DoRA requires dense linear targets"):
        wrapper._apply_lora(wrapper.model)
    assert isinstance(attention.q_proj, nn.Linear)
    assert set(dict(tree_flatten(wrapper.model.trainable_parameters()))) == set(before)


def test_plain_lora_still_constructs_lora_layers(tmp_path):
    from mlx_lm.tuner.lora import LoRALinear

    base = _checkpoint(tmp_path / "base")
    wrapper = MLXSFTTrainerWrapper(_config(base, tmp_path / "out", dora=False))
    wrapper.setup(_rows())
    wrapper._apply_lora(wrapper.model)
    assert isinstance(wrapper.model.layers[0].self_attn.q_proj, LoRALinear)
    assert not any(name.endswith(".m")
                   for name, _ in tree_flatten(wrapper.model.trainable_parameters()))


@pytest.mark.parametrize("mode", ["train_field", "responses_only"])
@pytest.mark.parametrize("quantized", [False, True])
def test_native_real_preprocess_cache_loss_training_and_reload(
    tmp_path, monkeypatch, mode, quantized,
):
    from datasets import load_from_disk
    from mlx_lm.tuner.datasets import CacheDataset
    from typer.testing import CliRunner

    from soup_cli.cli import app
    from soup_cli.trainer.mlx_masking import MaskedChatDataset
    from soup_cli.trainer.mlx_pretokenized import labeled_iterate_batches, labeled_loss

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    base = tmp_path / "base"
    messages, ids = _selective_fixture(base, quantized)
    cfg = _config(base, tmp_path / "out", dora=False, quantized=quantized)
    cfg.data.format = "chatml"
    cfg.data.val_split = 0.0
    cfg.data.train_on_responses_only = mode != "train_field"
    cfg.data.train_on_messages_with_train_field = mode == "train_field"
    raw = [{"messages": messages}] * 4
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in raw) + "\n")
    cfg.data.train = str(data)
    config = tmp_path / "config.json"
    config.write_text(cfg.model_dump_json())
    cache = tmp_path / "cache"
    result = CliRunner().invoke(app, ["data", "preprocess", str(config),
                                      "--output", str(cache), "--yes"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    (cache,) = [path for path in cache.iterdir() if path.is_dir()]
    cached = list(load_from_disk(str(cache)))
    assert len(cached) == 4 and cached[0]["input_ids"] == ids
    uncached = MaskedChatDataset(raw, load(str(base))[1], train_field=mode == "train_field")
    raw_ids, raw_mask = uncached.process(raw[0])
    assert raw_ids == ids
    if mode == "train_field":
        assert cached[0]["labels"] == [token if keep else -100
                                       for token, keep in zip(raw_ids, raw_mask)]
    else:
        # The producer freezes the shared Transformers objective. Native raw
        # template heuristics may differ; the consumer must retain stored labels.
        from transformers import AutoTokenizer

        from soup_cli.data.loss_mask import build_assistant_only_labels

        shared = build_assistant_only_labels(
            messages, AutoTokenizer.from_pretrained(str(base)), max_length=64,
        )
        assert cached[0]["labels"] == shared["labels"]
    cfg.data.format = "pre_tokenized"
    cfg.data.tokenized_path = str(cache)
    wrapper = MLXSFTTrainerWrapper(cfg)
    wrapper.setup({"train": []})
    assert list(wrapper._dataset["train"]) == cached
    frozen = dict(tree_flatten(wrapper.model.parameters()))
    mx.eval(frozen)
    wrapper._apply_lora(wrapper.model)
    wrapper.model.eval()
    batch, labels = next(iter(labeled_iterate_batches(
        CacheDataset(wrapper._dataset["train"]), batch_size=2, max_seq_length=64,
    )))
    assert batch[0, :len(ids)].tolist() == ids
    assert labels[0, :len(ids)].tolist() == cached[0]["labels"]
    assert (labels[:, len(ids):] == -100).all().item()  # PAD=EOS cannot create targets.

    def oracle(model):
        logits = model(batch[:, :-1]).astype(mx.float32)
        selected = labels[:, 1:] != -100
        targets = mx.maximum(labels[:, 1:], 0)
        ce = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(
            logits, targets[..., None], axis=-1,
        ).squeeze(-1)
        return (ce * selected).sum() / selected.sum()

    actual, gradients = nn.value_and_grad(wrapper.model, labeled_loss)(
        wrapper.model, batch, labels,
    )
    expected, expected_gradients = nn.value_and_grad(wrapper.model, oracle)(wrapper.model)
    assert actual[1].item() == 2 * sum(value != -100 for value in cached[0]["labels"][1:])
    assert actual[0].item() == pytest.approx(expected.item(), abs=2e-6)
    expected_flat = dict(tree_flatten(expected_gradients))
    for name, value in tree_flatten(gradients):
        assert mx.allclose(value, expected_flat[name], atol=2e-6, rtol=2e-6).item(), name
    del wrapper

    observed = {}
    setup = MLXSFTTrainerWrapper.setup

    def no_tokenization(*args, **kwargs):
        raise AssertionError("cached training must not tokenize or render")

    def observe_setup(self, dataset):
        setup(self, dataset)
        observed["wrapper"] = self
        # Observe the real loaded native tokenizer; never replace model/trainer outputs.
        self.tokenizer.encode = no_tokenization
        self.tokenizer.apply_chat_template = no_tokenization

    monkeypatch.setattr(MLXSFTTrainerWrapper, "setup", observe_setup)
    config.write_text(cfg.model_dump_json())
    outcome = CliRunner().invoke(app, ["train", "--config", str(config), "--yes"])
    assert outcome.exit_code == 0, (outcome.output, repr(outcome.exception))
    trained_wrapper = observed["wrapper"]
    trained_wrapper.model.eval()
    restored, _ = load(str(base), adapter_path=str(tmp_path / "out"))
    restored.eval()
    trained = dict(tree_flatten(restored.parameters()))
    for name, value in frozen.items():
        parent, leaf = name.rsplit(".", 1)
        assert mx.array_equal(trained.get(f"{parent}.linear.{leaf}", trained.get(name)),
                              value).item(), name
    weights = mx.load(str(tmp_path / "out" / "adapters.safetensors"))
    assert any(mx.abs(value).sum().item() > 0 for name, value in weights.items()
               if name.endswith("lora_b"))
    assert mx.allclose(restored(batch), trained_wrapper.model(batch), atol=1e-5).item()
    assert mx.abs(restored(batch) - load(str(base))[0](batch)).max().item() > 1e-4
    records = [json.loads(line) for line in (tmp_path / "out" / "rewind.jsonl").read_text()
               .splitlines()]
    steps = [record for record in records if "row_tokens" in record]
    assert steps
    expected_count = sum(value != -100 for value in cached[0]["labels"][1:])
    assert all(all(value == expected_count for value in record["row_tokens"]) for record in steps)
    metadata = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
    assert metadata["pre_tokenized"] is True


def test_native_explicit_cached_targets_and_rewind_share_the_numerical_objective():
    from types import SimpleNamespace

    from soup_cli.trainer.mlx_pretokenized import PretokenizedDataset, labeled_loss
    from soup_cli.trainer.rewind_mlx import ROW_LOSS_KEY, ROW_TOKENS_KEY, make_rewind_loss

    class FixedTable(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.arange(64, dtype=mx.float32).reshape(8, 8) / 10

        def __call__(self, tokens):
            return self.weight[tokens]

    row = {"input_ids": [1, 3, 4, 2], "labels": [-100, -100, 6, 2]}
    data = PretokenizedDataset([row], vocab_size=8, max_length=4)
    tokens, targets = data.process(data[0])
    batch, labels = mx.array([tokens]), mx.array([targets])
    model = FixedTable()
    model.train()
    state = SimpleNamespace(failed=False, optimizer_state={}, mark_pending=lambda: None)
    recorded = make_rewind_loss(state, kind="labels")
    logits = model(batch[:, :-1])
    expected = ((mx.logsumexp(logits, axis=-1)[0, 1] - logits[0, 1, 6]) +
                (mx.logsumexp(logits, axis=-1)[0, 2] - logits[0, 2, 2])) / 2
    loss, count = labeled_loss(model, batch, labels)
    tracked, tracked_count = recorded(model, batch, labels)
    assert count.item() == tracked_count.item() == 2
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
    assert tracked.item() == pytest.approx(expected.item(), abs=1e-6)
    assert state.optimizer_state[ROW_LOSS_KEY].item() == pytest.approx(expected.item(), abs=1e-6)
    assert state.optimizer_state[ROW_TOKENS_KEY].item() == 2
    _, actual_grad = nn.value_and_grad(model, labeled_loss)(model, batch, labels)
    _, rewind_grad = nn.value_and_grad(model, recorded)(model, batch, labels)
    assert mx.allclose(actual_grad["weight"], rewind_grad["weight"], atol=1e-6).item()
    assert actual_grad["weight"][3, 6].item() < 0
    assert actual_grad["weight"][3, 4].item() > 0
