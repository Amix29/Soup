"""#602 — Qwen4-Exp PLE rows may stay on a read-only SSD mapping."""

from __future__ import annotations

import hashlib
import json
import os
import re

import pytest


def _tiny_qwen4_config(*, ple: bool = True):
    try:
        from transformers import Qwen4ExpConfig, Qwen4ExpTextConfig
    except ImportError:
        pytest.skip("installed Transformers release does not include qwen4_exp yet")

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


def test_ple_header_rejects_shape_range_mismatch_and_past_eof():
    from soup_cli.utils.qwen4_ple import _tensor_rows_from_header

    with pytest.raises(ValueError, match="byte range"):
        _tensor_rows_from_header(
            {"ple": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 4]}},
            data_start=32,
            file_size=64,
            source_key="ple",
        )
    with pytest.raises(ValueError, match="past end"):
        _tensor_rows_from_header(
            {"ple": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}},
            data_start=32,
            file_size=47,
            source_key="ple",
        )


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


@pytest.mark.parametrize(
    ("bits", "words", "scale", "bias", "expected"),
    [
        (
            4,
            [3437096703, 2291772091, 1146447479, 1122867],
            -2.066666603088379,
            31.0,
            [0.0, 0.0, 2.0666675567626953, 2.0666675567626953],
        ),
        (
            5,
            [1975416799, 978769862, 902792293, 3343013046, 4469268],
            -1.0,
            31.0,
            [0.0, 1.0, 2.0, 3.0],
        ),
        (
            6,
            [
                2011676543,
                3144664893,
                2251909030,
                375498526,
                2735620389,
                2164256,
            ],
            -0.4920634925365448,
            31.0,
            [0.0, 0.9841270446777344, 1.9682540893554688, 2.952381134033203],
        ),
        (
            8,
            [
                3874486271,
                3318666974,
                2779624893,
                2223805596,
                1667986299,
                1112167002,
                556347706,
                528409,
            ],
            -0.12156862765550613,
            31.0,
            [0.0, 0.9725494384765625, 1.945098876953125, 3.039215087890625],
        ),
    ],
)
def test_oq_affine_decoder_matches_mlx_vectors(bits, words, scale, bias, expected):
    import torch

    from soup_cli.utils.oq_affine import AffineQuantSpec, dequantize_affine

    actual = dequantize_affine(
        torch.tensor([words], dtype=torch.uint32),
        torch.tensor([[scale]]),
        torch.tensor([[bias]]),
        spec=AffineQuantSpec(bits=bits, group_size=32),
        dtype="float32",
    )

    torch.testing.assert_close(
        actual[0, :4], torch.tensor(expected), rtol=0, atol=0
    )


def test_qwen4_sharder_dequantizes_omlx_oq_without_copying_companions(tmp_path):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint

    weights = tmp_path / "weights"
    shards = tmp_path / "shards"
    weights.mkdir()
    save_file(
        {
            "language_model.model.layers.0.proj.weight": torch.tensor(
                [[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.uint32
            ),
            "language_model.model.layers.0.proj.scales": torch.ones(
                (2, 1), dtype=torch.bfloat16
            ),
            "language_model.model.layers.0.proj.biases": torch.zeros(
                (2, 1), dtype=torch.bfloat16
            ),
            "language_model.model.layers.0.conv1d.weight": torch.arange(
                24, dtype=torch.float32
            ).reshape(2, 12, 1),
            (
                "language_model.model.layers.0.ple.ple_embedding."
                "ngram_embedding.shards.0.weight"
            ): torch.tensor([[0, 1, 2, 3]] * 3, dtype=torch.uint32),
            (
                "language_model.model.layers.0.ple.ple_embedding."
                "ngram_embedding.shards.0.scales"
            ): torch.ones((3, 1), dtype=torch.bfloat16),
            (
                "language_model.model.layers.0.ple.ple_embedding."
                "ngram_embedding.shards.0.biases"
            ): torch.zeros((3, 1), dtype=torch.bfloat16),
            "vision_tower.ignored.weight": torch.ones(2, 2),
            "mtp.ignored.weight": torch.ones(2, 2),
        },
        weights / "model.safetensors",
    )
    (weights / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "bits": 4,
                    "group_size": 32,
                    "mode": "affine",
                }
            }
        ),
        encoding="utf-8",
    )

    index = shard_checkpoint(
        str(weights), str(shards), dtype="float32", arch="qwen4_exp"
    )

    assert index.n_layers == 1
    assert index.layer_keys == ("conv1d.weight", "proj.weight")
    assert len(index.external_tensors) == 1
    external = next(iter(index.external_tensors.values()))
    assert external.shape == (3, 32)
    assert external.bits == 4
    with safe_open(layer_shard_path(str(shards), 0), framework="pt") as handle:
        assert handle.keys() == ["conv1d.weight", "proj.weight"]
        actual = handle.get_tensor("proj.weight")
        conv = handle.get_tensor("conv1d.weight")
    assert actual.shape == (2, 32)
    assert conv.shape == (2, 1, 12)
    assert not any(key.endswith((".scales", ".biases")) for key in index.layer_keys)


def test_oq_ple_reader_dequantizes_only_selected_read_only_rows(tmp_path):
    import torch
    from safetensors.torch import save_file

    from soup_cli.utils.layer_shard import OQExternalTensorPart, OQExternalTensorSpec
    from soup_cli.utils.qwen4_ple import OQShardedSafeTensorRowReader

    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / "model.safetensors"
    bits = 5
    rows = []
    for offset in range(4):
        stream = sum(((value + offset) % 32) << (bits * value) for value in range(32))
        rows.append([(stream >> (32 * word)) & 0xFFFFFFFF for word in range(bits)])
    packed = torch.tensor(rows, dtype=torch.uint32)
    scales = torch.tensor([[0.5], [1.0], [1.5], [2.0]], dtype=torch.bfloat16)
    biases = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.bfloat16)
    save_file(
        {"ple.weight": packed, "ple.scales": scales, "ple.biases": biases},
        source,
    )
    part = OQExternalTensorPart(
        source_file=source.name,
        weight_key="ple.weight",
        scales_key="ple.scales",
        biases_key="ple.biases",
        packed_shape=tuple(packed.shape),
        stats_shape=tuple(scales.shape),
        packed_dtype="U32",
        stats_dtype="BF16",
    )
    spec = OQExternalTensorSpec(
        parts=(part,),
        shape=(4, 32),
        dtype="float32",
        bits=bits,
        group_size=32,
        mode="affine",
    )
    before = _sha256(source)
    reader = OQShardedSafeTensorRowReader(str(weights), spec)
    ids = torch.tensor([[3, 1, 3]])

    actual = reader.gather(ids)

    quantized = torch.tensor(
        [[(value + offset) % 32 for value in range(32)] for offset in (3, 1, 3)],
        dtype=torch.float32,
    )
    expected = quantized * scales[[3, 1, 3]].float() + biases[[3, 1, 3]].float()
    torch.testing.assert_close(actual.squeeze(0), expected, rtol=0, atol=0)
    assert reader.nbytes == packed.numel() * 4 + (scales.numel() + biases.numel()) * 2
    reader.close()
    assert _sha256(source) == before


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
        # projection, even with its zero-initialised B matrix. The MPS gate uses
        # a narrow numerical tolerance; the CPU oracle above remains bit-exact
        # and guards the PLE row mapping itself.
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


def test_qwen4_streaming_refuses_unsupported_task_by_name():
    from soup_cli.trainer.stream_setup import _validate_qwen4_streaming_mode

    with pytest.raises(ValueError, match="task='sft'"):
        _validate_qwen4_streaming_mode(
            arch="qwen4_exp", task="dpo", quant="none"
        )


def test_qwen4_streaming_refuses_quantized_base_by_name():
    from soup_cli.trainer.stream_setup import _validate_qwen4_streaming_mode

    with pytest.raises(ValueError, match="quantization='none'"):
        _validate_qwen4_streaming_mode(
            arch="qwen4_exp", task="sft", quant="nf4"
        )


def test_qwen4_ple_disk_streaming_refuses_non_ssd_by_name():
    from soup_cli.trainer.stream_setup import _validate_qwen4_ngram_disk

    with pytest.raises(ValueError, match="needs an SSD or NVMe"):
        _validate_qwen4_ngram_disk(disk_kind="hdd", weights_dir="/slow/model")


def test_qwen4_stream_ngram_source_warns_when_checkpoint_has_no_ple():
    from soup_cli.trainer.stream_setup import _warn_if_ngram_source_unused

    messages = []
    _warn_if_ngram_source_unused(
        arch="qwen4_exp", requested="disk", ngram_bytes=0, notify=messages.append
    )

    assert len(messages) == 1
    assert "has no effect" in messages[0]


def test_external_ple_descriptor_reapplies_a_production_sized_element_cap():
    from soup_cli.utils.layer_shard import ExternalTensorPart, ExternalTensorSpec

    oversized = ExternalTensorPart(
        source_file="model.safetensors",
        source_key="ple.weight",
        shape=(2**36 + 1, 1),
        dtype="F32",
    )

    with pytest.raises(ValueError, match="element cap"):
        ExternalTensorSpec(parts=(oversized,), shape=oversized.shape, dtype="F32")
