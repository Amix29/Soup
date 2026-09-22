"""Contract tests for the published STEP 13 CPU fixture harness (#379)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HARNESS = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "harness"
    / "fixture_window_cpu.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location("fixture_window_cpu", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recorded_fixture_shapes_and_m_grid_are_pinned():
    harness = _load_harness()
    assert harness.FIXTURE_SHAPES == ((64, 64), (256, 64))
    assert harness.M_VALUES == (8, 16, 32, 64, 128, 256, 512)


def test_one_row_reports_the_path_and_three_pairwise_differences():
    pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    harness = _load_harness()

    row = harness.measure_row(64, 64, 8, seed=3)
    assert row.out_features == 64
    assert row.in_features == 64
    assert row.m == 8
    assert isinstance(row.inference_packed_for_cpu, bool)
    assert row.training_packed_for_cpu is False
    assert row.inference_vs_variant2_max_abs >= 0.0
    assert row.training_vs_variant2_max_abs >= 0.0
    assert row.inference_vs_training_max_abs >= 0.0


def test_probe_row_count_is_shape_times_m_grid(monkeypatch):
    harness = _load_harness()

    def fake_measure(out_features, in_features, m):
        return harness.Row(
            out_features=out_features,
            in_features=in_features,
            m=m,
            inference_packed_for_cpu=False,
            training_packed_for_cpu=False,
            inference_vs_variant2_max_abs=0.0,
            training_vs_variant2_max_abs=0.0,
            inference_vs_training_max_abs=0.0,
        )

    monkeypatch.setattr(harness, "measure_row", fake_measure)
    result = harness.run_probe(m_values=(8, 64), shapes=((64, 64), (256, 64)))
    assert len(result["rows"]) == 4
    assert result["packed_inference_rows"] == 0


def test_harness_source_has_no_network_or_model_download():
    source = HARNESS.read_text(encoding="utf-8")
    forbidden = ("from_pretrained(", "load_dataset(", "requests.", "hf_hub_download")
    assert not any(token in source for token in forbidden)
