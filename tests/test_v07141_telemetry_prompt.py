import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch


def test_handle_telemetry_consent_env(monkeypatch):
    from soup_cli.cli import _handle_telemetry_consent

    monkeypatch.setenv("SOUP_TELEMETRY", "1")
    # Returns early
    _handle_telemetry_consent()


def test_handle_telemetry_consent_exists(tmp_path, monkeypatch):
    from soup_cli.cli import _handle_telemetry_consent

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    consent = tmp_path / ".soup" / "telemetry_consent"
    consent.parent.mkdir(parents=True)
    consent.touch()
    _handle_telemetry_consent()


def test_handle_telemetry_consent_not_tty(tmp_path, monkeypatch):
    from soup_cli.cli import _handle_telemetry_consent

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.delenv("SOUP_TELEMETRY", raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    _handle_telemetry_consent()


def test_handle_telemetry_consent_agree(tmp_path, monkeypatch):
    from soup_cli.cli import _handle_telemetry_consent

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.delenv("SOUP_TELEMETRY", raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    with patch("rich.prompt.Confirm.ask", return_value=True):
        _handle_telemetry_consent()

    consent = tmp_path / ".soup" / "telemetry_consent"
    assert consent.read_text() == "1"
    assert os.environ["SOUP_TELEMETRY"] == "1"


def test_handle_telemetry_consent_exception(tmp_path, monkeypatch):
    from soup_cli.cli import _handle_telemetry_consent

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.delenv("SOUP_TELEMETRY", raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    with patch("rich.prompt.Confirm.ask", side_effect=Exception("mock")):
        _handle_telemetry_consent()

    consent = tmp_path / ".soup" / "telemetry_consent"
    assert consent.read_text() == "0"
    assert os.environ["SOUP_TELEMETRY"] == "0"


def test_emit_telemetry_disabled():
    from soup_cli.cli import _emit_telemetry
    import soup_cli.cli as cli

    cli._telemetry_disabled = True
    _emit_telemetry(["soup"], 1.0)
    cli._telemetry_disabled = False


def test_emit_telemetry_no_consent(tmp_path, monkeypatch):
    from soup_cli.cli import _emit_telemetry

    monkeypatch.setattr("soup_cli.utils.trackers.is_telemetry_enabled", lambda: False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    _emit_telemetry(["soup"], 1.0)


def test_emit_telemetry_consent_0(tmp_path, monkeypatch):
    from soup_cli.cli import _emit_telemetry

    monkeypatch.setattr("soup_cli.utils.trackers.is_telemetry_enabled", lambda: False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    consent = tmp_path / ".soup" / "telemetry_consent"
    consent.parent.mkdir(parents=True)
    consent.write_text("0")
    _emit_telemetry(["soup"], 1.0)


def test_emit_telemetry_exception(tmp_path, monkeypatch):
    from soup_cli.cli import _emit_telemetry

    monkeypatch.setattr("soup_cli.utils.trackers.is_telemetry_enabled", lambda: False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    with patch("pathlib.Path.exists", side_effect=Exception("mock")):
        _emit_telemetry(["soup"], 1.0)


def test_distinct_id_creation_and_reuse(tmp_path, monkeypatch):
    from soup_cli.utils.trackers import get_or_create_distinct_id

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    # First call generates
    id1 = get_or_create_distinct_id()
    assert len(id1) == 36

    # Second call reuses
    id2 = get_or_create_distinct_id()
    assert id1 == id2


def test_distinct_id_invalid_length(tmp_path, monkeypatch):
    from soup_cli.utils.trackers import get_or_create_distinct_id

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    id_file = tmp_path / ".soup" / "distinct_id"
    id_file.parent.mkdir(parents=True)
    id_file.write_text("invalid")

    # Generates a new one
    id1 = get_or_create_distinct_id()
    assert len(id1) == 36
    assert id1 != "invalid"


def test_distinct_id_exception_swallowed(tmp_path, monkeypatch):
    from soup_cli.utils.trackers import get_or_create_distinct_id

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    with patch("pathlib.Path.exists", side_effect=Exception("mock")):
        id1 = get_or_create_distinct_id()
        assert len(id1) == 36
