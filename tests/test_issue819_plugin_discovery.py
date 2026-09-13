"""Regression tests for issue #819: make the plugin system reachable."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from soup_cli import plugins as plugins_pkg


class _HookPlugin:
    def pre_train(self, _context):
        return None


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch, tmp_path):
    state_path = tmp_path / "plugins.json"
    monkeypatch.setenv("SOUP_PLUGIN_STATE_PATH", str(state_path))
    monkeypatch.setattr(plugins_pkg, "_iter_plugin_entry_points", lambda: ())
    plugins_pkg.clear_plugins()
    yield state_path
    plugins_pkg.clear_plugins()


def _install_fake_entry_point(monkeypatch, name: str = "hello") -> None:
    module_name = f"issue819_{name.replace('-', '_')}"
    module = ModuleType(module_name)

    def register() -> None:
        plugins_pkg.register_plugin(
            name=name,
            version="1.0.0",
            plugin=_HookPlugin(),
            description="entry-point plugin",
        )

    module.register = register
    monkeypatch.setitem(sys.modules, module_name, module)
    entry_point = importlib.metadata.EntryPoint(
        name=name,
        value=f"{module_name}:register",
        group="soup_cli.plugins",
    )
    monkeypatch.setattr(
        plugins_pkg, "_iter_plugin_entry_points", lambda: (entry_point,)
    )


def test_plugins_cli_discovers_entry_point_without_manual_load(monkeypatch):
    from soup_cli.commands import plugins as plugins_cli

    _install_fake_entry_point(monkeypatch)
    result = CliRunner().invoke(plugins_cli.app, ["list"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "hello" in result.output
    assert "disabled" in result.output


def test_attach_callback_discovers_enabled_entry_point(monkeypatch, _isolated_registry):
    pytest.importorskip("transformers")
    from soup_cli.utils.peft_wiring import attach_plugin_callback

    _install_fake_entry_point(monkeypatch)
    _isolated_registry.write_text(
        json.dumps({"version": 1, "enabled": {"hello": True}}),
        encoding="utf-8",
    )
    trainer = MagicMock()

    assert attach_plugin_callback(trainer) is True
    trainer.add_callback.assert_called_once()


def test_disable_survives_a_fresh_process(tmp_path):
    package = tmp_path / "fake_soup_plugin.py"
    package.write_text(
        "from soup_cli.plugins import register_plugin\n"
        "class Plugin:\n"
        "    def pre_train(self, context):\n"
        "        return None\n"
        "def register():\n"
        "    register_plugin(name='hello', version='1.0.0', plugin=Plugin())\n",
        encoding="utf-8",
    )
    dist_info = tmp_path / "fake_soup_plugin-1.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: fake-soup-plugin\nVersion: 1.0.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[soup_cli.plugins]\nhello = fake_soup_plugin:register\n",
        encoding="utf-8",
    )
    state_path = tmp_path / "state" / "plugins.json"
    repo_src = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join((str(tmp_path), str(repo_src))),
            "SOUP_PLUGIN_STATE_PATH": str(state_path),
            "SOUP_NO_AUDIT_LOG": "1",
            "SOUP_TELEMETRY": "0",
        }
    )

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "soup_cli", "plugins", *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

    enabled = run("enable", "hello")
    assert enabled.returncode == 0, (enabled.stdout, enabled.stderr)
    disabled = run("disable", "hello")
    assert disabled.returncode == 0, (disabled.stdout, disabled.stderr)

    fresh = run("list")
    assert fresh.returncode == 0, (fresh.stdout, fresh.stderr)
    assert "hello" in fresh.stdout
    assert "disabled" in fresh.stdout
    assert json.loads(state_path.read_text(encoding="utf-8"))["enabled"]["hello"] is False


def test_install_stub_fails_instead_of_claiming_success():
    from soup_cli.commands import plugins as plugins_cli

    result = CliRunner().invoke(plugins_cli.app, ["install", "anything"])

    assert result.exit_code != 0
    assert "does not install" in result.output


def test_plugin_resources_are_visible_in_the_cli():
    from soup_cli.commands import plugins as plugins_cli

    plugins_pkg.register_plugin(
        name="resources",
        version="1.0.0",
        plugin=_HookPlugin(),
        templates=["my-template"],
        model_groups=["my-models"],
    )

    result = CliRunner().invoke(plugins_cli.app, ["list"])

    assert result.exit_code == 0
    assert "my-template" in result.output
    assert "my-models" in result.output


def test_partially_failing_entry_point_cannot_leave_plugin_enabled(monkeypatch):
    class BrokenEntryPoint:
        name = "partial"

        @staticmethod
        def load():
            def register_then_fail():
                plugins_pkg.register_plugin(
                    name="partial",
                    version="1.0.0",
                    plugin=_HookPlugin(),
                )
                raise RuntimeError("broken registrar")

            return register_then_fail

    monkeypatch.setattr(
        plugins_pkg, "_iter_plugin_entry_points", lambda: (BrokenEntryPoint(),)
    )

    plugins_pkg.load_plugins()

    assert plugins_pkg.get_plugin("partial") is not None
    assert plugins_pkg.is_enabled("partial") is False


def test_enable_state_is_published_with_atomic_replace(monkeypatch, _isolated_registry):
    plugins_pkg.register_plugin(
        name="atomic", version="1.0.0", plugin=_HookPlugin()
    )
    real_replace = os.replace
    replacements: list[tuple[str, str]] = []

    def recording_replace(source: str, destination: str) -> None:
        replacements.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(plugins_pkg.os, "replace", recording_replace)

    plugins_pkg.enable_plugin("atomic")

    assert len(replacements) == 1
    source, destination = replacements[0]
    assert source != destination
    assert destination == str(_isolated_registry)
