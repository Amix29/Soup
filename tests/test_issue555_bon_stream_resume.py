"""Regression tests for #555: resumable and bounded two-phase Best-of-N."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

_FINGERPRINT = "0" * 64


def _args(tmp_path):
    return [
        "best-of-n",
        "--provider",
        "ollama",
        "--model",
        "sampler-model",
        "--prompts",
        str(tmp_path / "prompts.jsonl"),
        "--n",
        "2",
        "--export-candidates",
        str(tmp_path / "candidates.jsonl"),
    ]


def _local_args(tmp_path, model_path, *extra):
    return [
        "best-of-n",
        "--base",
        str(model_path),
        "--prompts",
        str(tmp_path / "prompts.jsonl"),
        "--n",
        "2",
        "--seed",
        "23",
        "--export-candidates",
        str(tmp_path / "candidates.jsonl"),
        *extra,
    ]


def test_late_candidate_failure_resumes_without_replaying_completed_groups(
    tmp_path, monkeypatch
):
    from soup_cli.commands.data import app
    from soup_cli.utils.best_of_n_artifact import load_candidate_artifact

    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompts.jsonl").write_text(
        ''.join(json.dumps({"prompt": prompt}) + "\n" for prompt in ("q1", "q2", "q3")),
        encoding="utf-8",
    )
    first_calls = []

    def failing_factory(*_args, **_kwargs):
        def generate(prompt):
            first_calls.append(prompt)
            if prompt == "q2":
                raise RuntimeError("simulated sampler interruption")
            return f"first-{prompt}-{len(first_calls)}"

        return generate

    monkeypatch.setattr("soup_cli.utils.magpie.make_magpie_generate_fn", failing_factory)
    failed = CliRunner().invoke(app, _args(tmp_path))
    assert failed.exit_code == 1
    assert "Completed groups: 1/3" in failed.output
    assert not (tmp_path / "candidates.jsonl").exists()
    checkpoint = tmp_path / "candidates.jsonl.checkpoint.jsonl"
    assert len(checkpoint.read_bytes().splitlines()) == 2

    resumed_calls = []

    def resumed_factory(*_args, **_kwargs):
        def generate(prompt):
            resumed_calls.append(prompt)
            return f"resumed-{prompt}-{len(resumed_calls)}"

        return generate

    monkeypatch.setattr("soup_cli.utils.magpie.make_magpie_generate_fn", resumed_factory)
    resumed = CliRunner().invoke(app, [*_args(tmp_path), "--resume"])
    assert resumed.exit_code == 0, (resumed.output, repr(resumed.exception))
    assert resumed_calls == ["q2", "q2", "q3", "q3"]
    groups, sampler, _digest = load_candidate_artifact(str(tmp_path / "candidates.jsonl"))
    assert [group["prompt"] for group in groups] == ["q1", "q2", "q3"]
    assert sampler["model"] == "sampler-model"


def test_resume_rejects_changed_prompt_contract_before_sampling(tmp_path, monkeypatch):
    from soup_cli.commands.data import app

    monkeypatch.chdir(tmp_path)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt":"original"}\n', encoding="utf-8")
    monkeypatch.setattr(
        "soup_cli.utils.magpie.make_magpie_generate_fn",
        lambda *_args, **_kwargs: lambda _prompt: "candidate",
    )
    first = CliRunner().invoke(app, _args(tmp_path))
    assert first.exit_code == 0, (first.output, repr(first.exception))
    prompts.write_text('{"prompt":"changed"}\n', encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        "soup_cli.utils.magpie.make_magpie_generate_fn",
        lambda *_args, **_kwargs: lambda prompt: calls.append(prompt) or "unexpected",
    )
    resumed = CliRunner().invoke(app, [*_args(tmp_path), "--resume"])
    assert resumed.exit_code == 1
    assert "does not match this run" in resumed.output
    assert calls == []


def test_resume_rejects_changed_prompt_source_line_before_sampling(tmp_path, monkeypatch):
    from soup_cli.commands.data import app

    monkeypatch.chdir(tmp_path)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt":"same"}\n', encoding="utf-8")
    monkeypatch.setattr(
        "soup_cli.utils.magpie.make_magpie_generate_fn",
        lambda *_args, **_kwargs: lambda _prompt: "candidate",
    )
    first = CliRunner().invoke(app, _args(tmp_path))
    assert first.exit_code == 0, (first.output, repr(first.exception))

    prompts.write_text('\n{"prompt":"same"}\n', encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        "soup_cli.utils.magpie.make_magpie_generate_fn",
        lambda *_args, **_kwargs: lambda prompt: calls.append(prompt) or "unexpected",
    )
    resumed = CliRunner().invoke(app, [*_args(tmp_path), "--resume"])

    assert resumed.exit_code == 1
    assert "does not match this run" in resumed.output
    assert calls == []


def test_offline_large_artifacts_bypass_whole_file_materializers(tmp_path, monkeypatch):
    from soup_cli.commands.data import app
    from soup_cli.utils.best_of_n_artifact import (
        build_candidate_group,
        candidate_artifact_header,
        stable_json_line,
        verify_offline_manifest,
    )

    monkeypatch.chdir(tmp_path)
    sampler = {
        "kind": "provider",
        "provider": "ollama",
        "model": "sampler-model",
        "endpoint_fingerprint": _FINGERPRINT,
        "n": 2,
        "temperature": 1.0,
        "max_new_tokens": 256,
    }
    count = 1500
    candidate_path = tmp_path / "large-candidates.jsonl"
    judgment_path = tmp_path / "large-judgments.jsonl"
    with candidate_path.open("wb") as candidate_file, judgment_path.open(
        "wb"
    ) as judgment_file:
        candidate_file.write(stable_json_line(candidate_artifact_header(count, sampler)))
        for index in range(count):
            group = build_candidate_group(
                f"prompt-{index}",
                index,
                [f"loser-{index}", f"winner-{index}"],
                sampler,
                source_line=index + 1,
            )
            candidate_file.write(stable_json_line(group))
            judgment_file.write(
                stable_json_line(
                    {
                        "prompt_id": group["prompt_id"],
                        "group_digest": group["group_digest"],
                        "winner_idx": 1,
                        "scores": [0.0, 1.0],
                        "verifier": {"name": "Codex", "version": "offline-v1"},
                    }
                )
            )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("whole-artifact materializer must not be used")

    monkeypatch.setattr("soup_cli.utils.best_of_n_artifact.load_candidate_artifact", forbidden)
    monkeypatch.setattr("soup_cli.utils.best_of_n_artifact.load_judgments", forbidden)
    monkeypatch.setattr("soup_cli.utils.best_of_n_artifact.materialize_rows", forbidden)
    monkeypatch.setattr("soup_cli.utils.best_of_n_artifact.stable_jsonl", forbidden)
    from soup_cli.utils.best_of_n_stream import OfflineArtifactIndex

    real_iter_rows = OfflineArtifactIndex.iter_rows
    iteration_count = {"value": 0}

    def counted_iter_rows(self):
        iteration_count["value"] += 1
        yield from real_iter_rows(self)

    monkeypatch.setattr(OfflineArtifactIndex, "iter_rows", counted_iter_rows)
    sft = tmp_path / "sft.jsonl"
    dpo = tmp_path / "dpo.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "best-of-n",
            "--candidate-artifact",
            str(candidate_path),
            "--judgments",
            str(judgment_path),
            "--output",
            str(sft),
            "--emit-pairs",
            str(dpo),
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert sum(1 for _ in sft.open(encoding="utf-8")) == count
    assert sum(1 for _ in dpo.open(encoding="utf-8")) == count
    assert iteration_count["value"] == 1
    verify_offline_manifest(
        str(tmp_path / "sft.jsonl.manifest.json"),
        sft_path=str(sft),
        dpo_path=str(dpo),
    )


@pytest.mark.parametrize("mutation", ["reordered", "truncated"])
def test_streaming_candidate_structure_failures_never_commit_manifest(
    tmp_path, monkeypatch, mutation
):
    from soup_cli.commands.data import app
    from soup_cli.utils.best_of_n_artifact import (
        build_candidate_group,
        candidate_artifact_header,
        stable_json_line,
    )

    monkeypatch.chdir(tmp_path)
    sampler = {
        "kind": "provider",
        "provider": "ollama",
        "model": "sampler-model",
        "endpoint_fingerprint": _FINGERPRINT,
        "n": 2,
        "temperature": 1.0,
        "max_new_tokens": 256,
    }
    groups = [
        build_candidate_group(
            f"q{index}", index, ["a", "b"], sampler, source_line=index + 1
        )
        for index in range(2)
    ]
    records = [candidate_artifact_header(2, sampler), *groups]
    if mutation == "reordered":
        records[1:] = reversed(records[1:])
    else:
        records.pop()
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_bytes(b"".join(stable_json_line(row) for row in records))
    judgments = tmp_path / "judgments.jsonl"
    judgments.write_text("", encoding="utf-8")
    output = tmp_path / "sft.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "best-of-n",
            "--candidate-artifact",
            str(candidate_path),
            "--judgments",
            str(judgments),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 2
    assert not output.exists()
    assert not (tmp_path / "sft.jsonl.manifest.json").exists()


def test_resume_discards_an_uncommitted_candidate_tail(tmp_path, monkeypatch):
    from soup_cli.utils.best_of_n_artifact import build_candidate_group
    from soup_cli.utils.best_of_n_stream import (
        append_candidate_group,
        prepare_candidate_checkpoint,
    )

    monkeypatch.chdir(tmp_path)
    sampler = {
        "kind": "provider",
        "provider": "ollama",
        "model": "sampler-model",
        "endpoint_fingerprint": _FINGERPRINT,
        "n": 2,
        "temperature": 1.0,
        "max_new_tokens": 256,
    }
    prompts = ["q1", "q2"]
    checkpoint = tmp_path / "checkpoint.jsonl"
    assert prepare_candidate_checkpoint(
        str(checkpoint), prompts, sampler, resume=False
    ) == 0
    append_candidate_group(
        str(checkpoint),
        build_candidate_group("q1", 0, ["a", "b"], sampler, source_line=1),
    )
    with checkpoint.open("ab") as handle:
        handle.write(b'{"prompt_index":1,"prompt":')

    completed = prepare_candidate_checkpoint(
        str(checkpoint), prompts, sampler, resume=True
    )

    assert completed == 1
    assert checkpoint.read_bytes().endswith(b"\n")


def test_resume_rejects_provider_endpoint_drift(tmp_path, monkeypatch):
    from soup_cli.commands.data import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompts.jsonl").write_text('{"prompt":"q"}\n', encoding="utf-8")
    monkeypatch.setattr(
        "soup_cli.utils.magpie.make_magpie_generate_fn",
        lambda *_args, **_kwargs: lambda _prompt: "candidate",
    )
    first = CliRunner().invoke(app, _args(tmp_path))
    assert first.exit_code == 0, (first.output, repr(first.exception))

    changed = [*_args(tmp_path), "--resume", "--base-url", "http://localhost:11435"]
    resumed = CliRunner().invoke(app, changed)

    assert resumed.exit_code == 1
    assert "does not match this run" in resumed.output


def test_resume_rejects_a_different_private_local_model(tmp_path, monkeypatch):
    from soup_cli.commands.data import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompts.jsonl").write_text('{"prompt":"q"}\n', encoding="utf-8")
    model_a = tmp_path / "model-a"
    model_b = tmp_path / "model-b"
    model_a.mkdir()
    model_b.mkdir()
    monkeypatch.setattr(
        "soup_cli.utils.trust_remote.model_requires_trust_remote_code",
        lambda _model: False,
    )
    monkeypatch.setattr(
        "soup_cli.commands.data._load_bon_model", lambda *_args, **_kwargs: (object(), object())
    )
    monkeypatch.setattr(
        "soup_cli.utils.best_of_n.sample_candidates",
        lambda *_args, n, **_kwargs: [f"candidate-{i}" for i in range(n)],
    )

    first = CliRunner().invoke(app, _local_args(tmp_path, model_a))
    assert first.exit_code == 0, (first.output, repr(first.exception))
    resumed = CliRunner().invoke(
        app, _local_args(tmp_path, model_b, "--resume")
    )

    assert resumed.exit_code == 1
    assert "does not match this run" in resumed.output
    artifact_text = (tmp_path / "candidates.jsonl").read_text(encoding="utf-8")
    assert str(model_a) not in artifact_text
    assert str(model_b) not in artifact_text


def test_resume_rejects_replaced_content_at_the_same_local_model_path(
    tmp_path, monkeypatch
):
    from soup_cli.commands.data import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompts.jsonl").write_text('{"prompt":"q"}\n', encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "weights.bin"
    weights.write_bytes(b"first-content")
    load_calls = []
    monkeypatch.setattr(
        "soup_cli.utils.trust_remote.model_requires_trust_remote_code",
        lambda _model: False,
    )
    monkeypatch.setattr(
        "soup_cli.commands.data._load_bon_model",
        lambda *_args, **_kwargs: (
            load_calls.append(True) or (object(), object())
        ),
    )
    monkeypatch.setattr(
        "soup_cli.utils.best_of_n.sample_candidates",
        lambda *_args, n, **_kwargs: [f"candidate-{index}" for index in range(n)],
    )

    first = CliRunner().invoke(app, _local_args(tmp_path, model))
    assert first.exit_code == 0, (first.output, repr(first.exception))
    assert load_calls == [True]

    weights.write_bytes(b"other-content")
    load_calls.clear()
    resumed = CliRunner().invoke(app, _local_args(tmp_path, model, "--resume"))

    assert resumed.exit_code == 1
    assert "does not match this run" in resumed.output
    assert load_calls == []


def test_local_resume_reuses_the_same_prompt_seed(tmp_path, monkeypatch):
    import torch

    from soup_cli.commands.data import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompts.jsonl").write_text(
        '{"prompt":"first"}\n{"prompt":"second"}\n', encoding="utf-8"
    )
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(
        "soup_cli.utils.trust_remote.model_requires_trust_remote_code",
        lambda _model: False,
    )
    monkeypatch.setattr(
        "soup_cli.commands.data._load_bon_model", lambda *_args, **_kwargs: (object(), object())
    )
    sampled = []
    fail_second = {"value": True}

    def sample_candidates(_model, _tokenizer, prompt, *, n, **_kwargs):
        values = [f"{torch.rand(1).item():.12f}" for _ in range(n)]
        sampled.append((prompt, values))
        if prompt == "second" and fail_second["value"]:
            raise RuntimeError("simulated interruption")
        return values

    monkeypatch.setattr(
        "soup_cli.utils.best_of_n.sample_candidates", sample_candidates
    )
    first = CliRunner().invoke(app, _local_args(tmp_path, model))
    assert first.exit_code == 1, (first.output, repr(first.exception))
    interrupted = sampled[-1][1]

    fail_second["value"] = False
    resumed = CliRunner().invoke(app, _local_args(tmp_path, model, "--resume"))

    assert resumed.exit_code == 0, (resumed.output, repr(resumed.exception))
    assert sampled[-1] == ("second", interrupted)
