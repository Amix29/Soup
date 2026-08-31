"""Teacher-only input fingerprint verification for AutoDistill Milestone B1."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from soup_cli.autodistill.contract import ArtifactCorruptionError, AutoDistillPlan
from soup_cli.autodistill.fingerprints import (
    verify_dataset_fingerprint,
    verify_teacher_fingerprint,
    verify_tokenizer_fingerprint,
)

FIXTURES = Path(__file__).parent / "fixtures" / "autodistill" / "v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _local_plan(tmp_path: Path) -> tuple[AutoDistillPlan, Path, Path, Path, str]:
    teacher_root = tmp_path / "teacher"
    tokenizer_root = tmp_path / "tokenizer"
    dataset_root = tmp_path / "dataset"
    teacher_root.mkdir()
    tokenizer_root.mkdir()
    dataset_root.mkdir()
    config = b'{"model_type":"fixture"}\n'
    weights = b"teacher-weights"
    tokenizer = b'{"version":"1.0"}\n'
    dataset = b'{ "answer": 2, "prompt": "1+1" }\r\n'
    normalized = b'{"answer":2,"prompt":"1+1"}\n'
    template = "{{ messages }}"
    (teacher_root / "config.json").write_bytes(config)
    (teacher_root / "model.safetensors").write_bytes(weights)
    (tokenizer_root / "tokenizer.json").write_bytes(tokenizer)
    (dataset_root / "prompts.jsonl").write_bytes(dataset)

    payload = json.loads((FIXTURES / "plan.json").read_text(encoding="utf-8"))
    payload["teacher"]["config_sha256"] = _sha(config)
    payload["teacher"]["weights"] = [
        {
            "path": "model.safetensors",
            "bytes": len(weights),
            "sha256": _sha(weights),
        }
    ]
    payload["tokenizer"]["files"] = [
        {
            "path": "tokenizer.json",
            "bytes": len(tokenizer),
            "sha256": _sha(tokenizer),
        }
    ]
    payload["tokenizer"]["chat_template_sha256"] = _sha(template.encode())
    payload["dataset"]["source_files"] = [
        {
            "path": "prompts.jsonl",
            "bytes": len(dataset),
            "sha256": _sha(dataset),
        }
    ]
    payload["dataset"]["normalized_sha256"] = _sha(normalized)
    payload["dataset"]["rows"] = 1
    return (
        AutoDistillPlan.model_validate(payload),
        teacher_root,
        tokenizer_root,
        dataset_root,
        template,
    )


def test_capture_inputs_verify_without_accepting_or_loading_a_student(tmp_path):
    plan, teacher_root, tokenizer_root, dataset_root, template = _local_plan(tmp_path)

    verify_teacher_fingerprint(plan, teacher_root=teacher_root)
    verify_tokenizer_fingerprint(
        plan,
        tokenizer_root=tokenizer_root,
        chat_template=template,
        renderer=plan.tokenizer.renderer,
    )
    verify_dataset_fingerprint(plan, dataset_root=dataset_root)

    assert "student" not in inspect.signature(verify_teacher_fingerprint).parameters


def test_teacher_fingerprint_fails_closed_after_one_changed_byte(tmp_path):
    plan, teacher_root, _, _, _ = _local_plan(tmp_path)
    (teacher_root / "model.safetensors").write_bytes(b"teacher-weightX")

    with pytest.raises(ArtifactCorruptionError, match="sha256 mismatch"):
        verify_teacher_fingerprint(plan, teacher_root=teacher_root)


def test_tokenizer_requires_exact_files_template_and_renderer(tmp_path):
    plan, _, tokenizer_root, _, template = _local_plan(tmp_path)
    with pytest.raises(ArtifactCorruptionError, match="chat template"):
        verify_tokenizer_fingerprint(
            plan,
            tokenizer_root=tokenizer_root,
            chat_template=template + " ",
            renderer=plan.tokenizer.renderer,
        )
    with pytest.raises(ArtifactCorruptionError, match="renderer"):
        verify_tokenizer_fingerprint(
            plan,
            tokenizer_root=tokenizer_root,
            chat_template=template,
            renderer="different-runtime",
        )


def test_dataset_verifies_source_bytes_normalized_hash_and_rows(tmp_path):
    plan, _, _, dataset_root, _ = _local_plan(tmp_path)
    verify_dataset_fingerprint(plan, dataset_root=dataset_root)
    (dataset_root / "prompts.jsonl").write_text(
        '{"answer":3,"prompt":"1+1"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ArtifactCorruptionError, match="byte count mismatch|sha256 mismatch"):
        verify_dataset_fingerprint(plan, dataset_root=dataset_root)


def test_fingerprint_verifier_rejects_symlinked_files(tmp_path):
    plan, teacher_root, _, _, _ = _local_plan(tmp_path)
    weights = teacher_root / "model.safetensors"
    real_weights = teacher_root / "real-weights"
    weights.rename(real_weights)
    weights.symlink_to(real_weights)

    with pytest.raises(ArtifactCorruptionError, match="is missing"):
        verify_teacher_fingerprint(plan, teacher_root=teacher_root)
