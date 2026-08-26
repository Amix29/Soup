"""Regression tests for #318: silent, environment-only telemetry opt-in."""

from __future__ import annotations

import json
import uuid
from unittest.mock import patch

from typer.testing import CliRunner


def test_emit_telemetry_disabled_never_builds_or_sends(monkeypatch):
    from soup_cli.cli import _emit_telemetry

    monkeypatch.delenv("SOUP_TELEMETRY", raising=False)
    monkeypatch.setattr(
        "soup_cli.utils.trackers.build_telemetry_payload",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("payload built")),
    )
    monkeypatch.setattr(
        "soup_cli.utils.trackers.send_telemetry_payload",
        lambda _payload: (_ for _ in ()).throw(AssertionError("telemetry sent")),
    )

    _emit_telemetry(["soup", "train"], 1.0)


def test_emit_telemetry_enabled_builds_and_sends(monkeypatch):
    from soup_cli.cli import _emit_telemetry

    monkeypatch.setenv("SOUP_TELEMETRY", "1")
    sent = []
    monkeypatch.setattr(
        "soup_cli.utils.trackers.build_telemetry_payload",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(
        "soup_cli.utils.trackers.send_telemetry_payload", sent.append
    )

    _emit_telemetry(["soup", "train", "--config", "private.yaml"], 1.25)

    assert len(sent) == 1
    assert sent[0]["command"] == "train"
    assert sent[0]["duration_seconds"] == 1.25
    assert "private.yaml" not in str(sent[0])


def test_cli_has_no_prompt_or_telemetry_override_flag(monkeypatch):
    from soup_cli.cli import app

    monkeypatch.delenv("SOUP_TELEMETRY", raising=False)
    with patch("rich.prompt.Confirm.ask", side_effect=AssertionError("prompted")):
        result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--no-telemetry" not in result.output


def test_send_boundary_disabled_never_opens_network(monkeypatch):
    from soup_cli.utils.trackers import send_telemetry_payload

    monkeypatch.delenv("SOUP_TELEMETRY", raising=False)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network opened")),
    )

    assert send_telemetry_payload({"command": "train"}) is False


def test_send_boundary_enabled_uses_stdlib_and_allowlisted_payload(monkeypatch):
    from soup_cli.utils.trackers import send_telemetry_payload

    monkeypatch.setenv("SOUP_TELEMETRY", "1")
    monkeypatch.setattr(
        "soup_cli.utils.trackers._telemetry_endpoint_is_safe", lambda _endpoint: True
    )
    captured = {}

    class Response:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def getcode(self):
            return self.status

    def open_request(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    payload = {
        "command": "train",
        "soup_version": "1.0",
        "python": "3.12",
        "os": "Darwin",
        "arch": "arm64",
        "duration_seconds": 1.0,
        "distinct_id": str(uuid.uuid4()),
    }

    assert send_telemetry_payload(payload, api_key="public-key") is True
    body = json.loads(captured["request"].data)
    assert captured["timeout"] == 1.0
    assert body["event"] == "train"
    assert set(body["properties"]) == set(payload) - {"command"}
    assert "httpx" not in captured["request"].headers.get("Content-type", "").lower()


def test_distinct_id_creation_reuse_and_invalid_replacement(tmp_path, monkeypatch):
    from soup_cli.utils.trackers import get_or_create_distinct_id

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    first = get_or_create_distinct_id()
    assert uuid.UUID(first)
    assert get_or_create_distinct_id() == first

    id_file = tmp_path / ".soup" / "telemetry_id"
    id_file.write_text("not-a-uuid", encoding="utf-8")
    replacement = get_or_create_distinct_id()
    assert uuid.UUID(replacement)
    assert replacement != "not-a-uuid"


def test_distinct_id_filesystem_failure_is_swallowed(tmp_path, monkeypatch):
    from soup_cli.utils.trackers import get_or_create_distinct_id

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    with patch("pathlib.Path.exists", side_effect=OSError("unavailable")):
        value = get_or_create_distinct_id()

    assert uuid.UUID(value)
