"""Load-bearing contracts for the #1112 Llasa/XCodec2 review fixes."""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import get_args

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_transformers_xcodec2_feature_extractor_contract_is_networkless():
    pytest.importorskip("torchaudio")
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")

    from transformers import Xcodec2FeatureExtractor, Xcodec2Model

    extractor = Xcodec2FeatureExtractor()
    audio = np.zeros(1280, dtype=np.float32)
    batch = extractor(audio=audio, sampling_rate=16_000, return_tensors="pt")

    encode_params = set(inspect.signature(Xcodec2Model.encode).parameters)
    assert batch
    assert set(batch.keys()) <= encode_params


def test_tts_docs_preencoded_example_loads_through_real_schema():
    from soup_cli.config.loader import load_config_from_string

    text = (ROOT / "docs" / "training.md").read_text(encoding="utf-8")
    marker = "**Pre-encoded chat (live for codec-string families).**"
    start = text.index(marker)
    fence = text.index("```yaml", start) + len("```yaml")
    end = text.index("```", fence)
    yaml = text[fence:end].strip()

    cfg = load_config_from_string(yaml)
    assert cfg.task == "tts"
    assert cfg.data.format == "chatml"
    assert cfg.training.lora.r == 16
    assert cfg.training.lora.alpha == 32


def test_every_source_data_format_literal_is_schema_valid():
    from soup_cli.config.schema import DataConfig

    allowed = set(get_args(DataConfig.model_fields["format"].annotation))
    pattern = re.compile(
        r"data\.format\s*(?:=|:)\s*[\"\'`]?([a-z][a-z0-9_-]*)",
        re.IGNORECASE,
    )
    found: dict[str, set[str]] = {}

    for path in (ROOT / "src" / "soup_cli").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        values = {match.group(1) for match in pattern.finditer(source)}
        if values:
            found[str(path.relative_to(ROOT))] = values

    assert found, "ratchet scanned no data.format literals"
    invalid = {
        value
        for values in found.values()
        for value in values
        if value not in allowed
    }
    assert not invalid, f"source names refused data.format values: {sorted(invalid)}"


def test_clear_xcodec2_cache_releases_the_requested_device(monkeypatch):
    from soup_cli.utils import tts_codec

    marker = object()
    monkeypatch.setattr(tts_codec, "_XCODEC2_CACHE", {"cpu": marker, "cuda": object()})

    tts_codec.clear_xcodec2_cache("cpu")

    assert "cpu" not in tts_codec._XCODEC2_CACHE
    assert "cuda" in tts_codec._XCODEC2_CACHE
