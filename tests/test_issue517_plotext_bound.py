"""Regression coverage for the Plotext 6 API break reported in #517."""

import re
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


def _plotext_requirement() -> Requirement:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^\s*"(plotext[^"]+)"', text, flags=re.MULTILINE)
    assert match, "pyproject.toml has no Plotext dependency"
    return Requirement(match.group(1))


def test_declared_plotext_range_keeps_the_supported_module_api() -> None:
    specifier = _plotext_requirement().specifier

    assert Version("5.2.0") in specifier
    assert Version("5.3.2") in specifier
    assert Version("6.0.0") not in specifier
