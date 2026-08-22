"""Temporary GitHub-hosted Apple Silicon probe for issue #99."""

from __future__ import annotations

import os
import platform

import pytest

from soup_cli.utils.gpu_monitor import PowermetricsStatus, query_powermetrics


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") != "true"
    or platform.system() != "Darwin"
    or platform.machine().lower() not in {"arm64", "aarch64"},
    reason="live probe only runs on GitHub-hosted Apple Silicon",
)
def test_live_powermetrics_snapshot():
    """Require one real, parsed GPU utilization or power sample."""
    result = query_powermetrics(0.25)
    print(f"live_powermetrics_result={result!r}")
    assert result.status is PowermetricsStatus.OK, result
    assert len(result.samples) == 1
    sample = result.samples[0]
    assert sample.util_gpu_pct is not None or sample.power_w is not None, sample
