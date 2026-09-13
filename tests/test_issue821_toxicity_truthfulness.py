"""Regression coverage for issue #821's toxicity-heuristic claims."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from soup_cli.cli import app


def test_abusive_language_scores_above_benign_technical_language() -> None:
    from soup_cli.utils.data_score import score_toxicity

    abusive = (
        "You are a worthless idiot and I hope you suffer.",
        "I will find where you live and hurt your family.",
        "Go back to your country; nobody wants your kind here.",
    )
    benign = (
        "heart attack symptoms",
        "Why does my Python thread die?",
        "How do I kill a zombie process in Linux?",
    )

    assert min(map(score_toxicity, abusive)) > max(map(score_toxicity, benign))


def test_magpie_default_quality_keeps_benign_process_management_text() -> None:
    from soup_cli.utils.magpie import default_quality_fn

    assert default_quality_fn(
        "How do I kill a hung process?",
        "Run kill -9 on its PID.",
    )


def test_toxicity_command_names_and_serialises_the_actual_heuristic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(
        json.dumps({"text": "You are a worthless idiot."}) + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    help_result = runner.invoke(app, ["data", "toxicity", "--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "violence-keyword heuristic" in help_result.output.lower()

    result = runner.invoke(
        app,
        ["data", "toxicity", "--input", str(input_path), "--output", "flagged.jsonl"],
    )
    assert result.exit_code == 0, result.output
    assert "violence-keyword heuristic" in result.output.lower()
    row = json.loads((tmp_path / "flagged.jsonl").read_text(encoding="utf-8"))
    assert row["_violence_keyword_score"] > 0.0
    assert "_toxicity" not in row


def test_docs_do_not_claim_unshipped_data_classifiers() -> None:
    docs = Path("docs/data.md").read_text(encoding="utf-8")

    assert "Llama-Guard-3-1B variant + FineWeb-Edu classifier ship" not in docs
    assert "not a toxicity classifier" in docs
