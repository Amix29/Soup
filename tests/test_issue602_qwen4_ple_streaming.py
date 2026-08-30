"""#602 — Qwen4-Exp PLE rows may stay on a read-only SSD mapping."""

from __future__ import annotations

import hashlib
import os
import re

import pytest


def _tiny_qwen4_config(*, ple: bool = True):
    from transformers import Qwen4ExpConfig, Qwen4ExpTextConfig

    text = Qwen4ExpTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts_per_tok=1,
        num_experts=2,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=64,
        hc_count=2,
        hc_lowrank=4,
        ple_layer_ids=[1] if ple else [],
        ple_embed_dim=16,
        heads_per_ngram=2,
        ngram_size=3,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=8,
        ple_conv_kernel_size=2,
        use_cache=False,
        indexer_n_heads=1,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
        eos_token_id=2,
    )
    return Qwen4ExpConfig(text_config=text.to_dict())


def _save_tiny_qwen4(path):
    import torch
    from transformers import AutoModelForCausalLM

    torch.manual_seed(602)
    model = AutoModelForCausalLM.from_config(
        _tiny_qwen4_config(), dtype=torch.float32
    )
    model.save_pretrained(path, safe_serialization=True)
    return model


def _ple_entry(weights_dir):
    from safetensors import safe_open

    found = []
    for name in sorted(os.listdir(weights_dir)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(weights_dir, name)
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():
                match = re.search(r"ngram_embedding\.shard_(\d+)\.weight$", key)
                if match:
                    tensor = handle.get_tensor(key)
                    found.append((int(match.group(1)), name, key, tensor))
                elif key.endswith(".ple.ple_embedding.ngram_embedding.weight"):
                    found.append((0, name, key, handle.get_tensor(key)))
    if found:
        _part, name, key, tensor = sorted(found)[0]
        return name, key, tensor
    raise AssertionError("tiny Qwen4 checkpoint has no PLE table")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.parametrize(
    ("torch_dtype", "safe_dtype"),
    [("float32", "F32"), ("float16", "F16"), ("bfloat16", "BF16")],
)
def test_sparse_reader_preserves_supported_dtypes(tmp_path, torch_dtype, safe_dtype):
    import torch
    from safetensors.torch import save_file

    from soup_cli.utils.qwen4_ple import SafeTensorRowReader

    weights = tmp_path / "weights"
    weights.mkdir()
    source = torch.arange(24, dtype=torch.float32).view(6, 4).to(
        getattr(torch, torch_dtype)
    )
    path = weights / "rows.safetensors"
    save_file({"rows": source}, path)
    reader = SafeTensorRowReader(
        str(weights),
        source_file=path.name,
        source_key="rows",
        expected_shape=tuple(source.shape),
        expected_dtype=safe_dtype,
    )

    ids = torch.tensor([5, 1, 5, 0])
    actual = reader.gather(ids)

    assert actual.dtype == source.dtype
    torch.testing.assert_close(actual, source[ids], rtol=0, atol=0)
    reader.close()


def test_sparse_reader_matches_resident_rows_without_mutating_source(tmp_path):
    import torch

    from soup_cli.utils.qwen4_ple import SafeTensorRowReader

    weights = tmp_path / "weights"
    weights.mkdir()
    _save_tiny_qwen4(weights)
    filename, key, resident = _ple_entry(weights)
    path = weights / filename
    before_hash = _sha256(path)
    before_stat = path.stat()

    reader = SafeTensorRowReader(
        str(weights),
        source_file=filename,
        source_key=key,
        expected_shape=tuple(resident.shape),
        expected_dtype="F32",
    )
    last = resident.shape[0] - 1
    row_ids = torch.tensor([[last, 0, last], [0, last, 0]])
    gathered = reader.gather(row_ids)

    torch.testing.assert_close(gathered, resident[row_ids], rtol=0, atol=0)
    with pytest.raises(TypeError):
        reader._mapping[reader.spec.start] = 0
    reader.close()
    assert reader.closed is True
    assert _sha256(path) == before_hash
    after_stat = path.stat()
    assert after_stat.st_size == before_stat.st_size
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns


def test_sparse_reader_rejects_escape_and_out_of_range_rows(tmp_path):
    import torch

    from soup_cli.utils.qwen4_ple import SafeTensorRowReader

    weights = tmp_path / "weights"
    weights.mkdir()
    _save_tiny_qwen4(weights)
    filename, key, resident = _ple_entry(weights)

    with pytest.raises(ValueError, match="filename"):
        SafeTensorRowReader(
            str(weights),
            source_file="../model.safetensors",
            source_key=key,
            expected_shape=tuple(resident.shape),
            expected_dtype="F32",
        )

    reader = SafeTensorRowReader(
        str(weights),
        source_file=filename,
        source_key=key,
        expected_shape=tuple(resident.shape),
        expected_dtype="F32",
    )
    with pytest.raises(IndexError, match="outside"):
        reader.gather(torch.tensor([resident.shape[0]]))
    reader.close()


def test_qwen4_sharder_keeps_ple_in_original_checkpoint(tmp_path):
    from safetensors import safe_open

    from soup_cli.utils.layer_shard import (
        QWEN4_PLE_WEIGHT_SUFFIX,
        layer_shard_path,
        read_shard_index,
        shard_checkpoint,
    )

    weights = tmp_path / "weights"
    shards = tmp_path / "shards"
    weights.mkdir()
    _save_tiny_qwen4(weights)

    index = shard_checkpoint(
        str(weights), str(shards), dtype="float32", arch="qwen4_exp"
    )

    assert index.external_mode == "qwen4_ple"
    assert len(index.external_tensors) == 1
    external_key = next(iter(index.external_tensors))
    assert external_key.endswith(QWEN4_PLE_WEIGHT_SUFFIX)
    assert len(index.external_tensors[external_key].parts) > 1
    assert external_key not in index.layer_keys
    reloaded = read_shard_index(str(shards))
    assert reloaded.external_tensors == index.external_tensors
    assert (
        shard_checkpoint(
            str(weights), str(shards), dtype="float32", arch="qwen4_exp"
        )
        == index
    )
    for layer_idx in range(index.n_layers):
        with safe_open(layer_shard_path(str(shards), layer_idx), framework="pt") as handle:
            assert all("ngram_embedding.weight" not in key for key in handle.keys())


def test_qwen4_sharder_rejects_omlx_oq_before_writing_cache(tmp_path):
    import torch
    from safetensors.torch import save_file

    from soup_cli.utils.layer_shard import shard_checkpoint

    weights = tmp_path / "weights"
    shards = tmp_path / "shards"
    weights.mkdir()
    save_file(
        {
            "language_model.model.layers.0.proj.weight": torch.ones(2, 2),
            "language_model.model.layers.0.proj.biases": torch.zeros(2),
        },
        weights / "model.safetensors",
    )

    with pytest.raises(ValueError, match="oMLX/oQ.*inference-only"):
        shard_checkpoint(
            str(weights), str(shards), dtype="float32", arch="qwen4_exp"
        )
    assert list(shards.iterdir()) == []


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("ngram_source", ["disk", "ram"])
def test_tiny_qwen4_ple_matches_resident_forward_loss_and_lora_gradients(
    tmp_path, device, ngram_source
):
    import torch
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights = tmp_path / "weights"
    shards = tmp_path / "shards"
    weights.mkdir()
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("needs an Apple Silicon MPS device")
    resident = _save_tiny_qwen4(weights).to(device).eval()
    index = shard_checkpoint(
        str(weights), str(shards), dtype="float32", arch="qwen4_exp"
    )
    streamed, runtime = build_streamed_model(
        model_id=str(weights),
        weights_dir=str(weights),
        shard_dir=str(shards),
        index=index,
        lora_config=LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "in_proj_qkv"],
            task_type=TaskType.CAUSAL_LM,
        ),
        device=device,
        dtype="float32",
        buffers=2,
        pin=False,
        tier="ram",
        ngram_source=ngram_source,
    )
    streamed.eval()
    batch = torch.tensor([[1, 3, 4, 5]], device=device)

    with torch.no_grad():
        expected = resident(input_ids=batch, labels=batch)
    actual = streamed(input_ids=batch, labels=batch)

    if device == "cpu":
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
        torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    else:
        # MPS may select a different reduction schedule once PEFT wraps a
        # projection, even with its zero-initialised B matrix. The measured
        # divergence on M4 Max is <= 1.49e-8 in float32; the CPU oracle above
        # remains bit-exact and guards the PLE row mapping itself.
        torch.testing.assert_close(actual.logits, expected.logits, rtol=3e-4, atol=2e-8)
        torch.testing.assert_close(actual.loss, expected.loss, rtol=3e-4, atol=2e-8)
    actual.loss.backward()
    trainable = [parameter for parameter in streamed.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert runtime.external_sources
    reader = runtime.external_sources[0]
    if ngram_source == "disk":
        assert len(reader.containers) == 1
        assert len({id(part._mapping) for part in reader.parts}) == 1
    runtime.close()
    assert reader.closed is True


def test_qwen4_streaming_gate_and_ngram_config():
    from types import SimpleNamespace

    from soup_cli.config.schema import TrainingConfig
    from soup_cli.utils.layer_stream import stream_arch_of

    assert stream_arch_of(SimpleNamespace(model_type="qwen4_exp")) == "qwen4_exp"
    assert (
        stream_arch_of(SimpleNamespace(model_type="qwen4_exp_text"))
        == "qwen4_exp"
    )
    assert TrainingConfig(stream_ngram_source="disk").stream_ngram_source == "disk"
    with pytest.raises(ValueError, match="stream_ngram_source"):
        TrainingConfig(stream_ngram_source="network")
