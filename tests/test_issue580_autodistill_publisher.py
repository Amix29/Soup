"""Transactional shard publication tests for AutoDistill Milestone B1."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from soup_cli.autodistill.capture import (
    TeacherExpertExample,
    capture_teacher_expert_trajectory,
)
from soup_cli.autodistill.contract import (
    ArtifactCorruptionError,
    AutoDistillPlan,
    CaptureToken,
    ShardManifest,
    canonical_sha256,
)
from soup_cli.autodistill.publisher import CaptureShardPublisher

FIXTURES = Path(__file__).parent / "fixtures" / "autodistill" / "v1"


def _plan() -> AutoDistillPlan:
    return AutoDistillPlan.model_validate_json((FIXTURES / "plan.json").read_bytes())


def _captures(plan: AutoDistillPlan):
    logits = [0.0] * plan.capture.vocab_size
    logits[10] = 4.0
    logits[11] = 3.0
    return capture_teacher_expert_trajectory(
        example=TeacherExpertExample(
            example_id="example-1",
            prompt_token_ids=(1, 2),
            target_token_ids=(10, 12),
        ),
        teacher_logits_by_position=(logits, logits),
        vocab_size=plan.capture.vocab_size,
        probability_policy=plan.probability_policy,
        max_sequence_length=plan.capture.max_sequence_length,
        truncation=plan.capture.truncation,
    )


def _publisher(root: Path, plan: AutoDistillPlan) -> CaptureShardPublisher:
    return CaptureShardPublisher(
        root=root,
        plan=plan,
        shard_id="shard-0001",
        transaction_id="transaction-0001",
    )


def _read_manifest(path: Path) -> ShardManifest:
    return ShardManifest.model_validate_json(path.read_bytes())


def test_transaction_resumes_every_committed_state_and_publishes_manifest_last(tmp_path):
    plan = _plan()
    rows = _captures(plan)
    root = tmp_path / "capture-cache"
    transaction = root / ".transactions" / "transaction-0001"
    available = root / "shards" / "shard-0001"

    staging = _publisher(root, plan).publish(rows, stop_after="staging")
    assert staging.state == "staging"
    assert transaction.is_dir()
    assert not available.exists()

    complete = _publisher(root, plan).publish(rows, stop_after="complete")
    assert complete.state == "complete"
    assert complete.previous_manifest_sha256 == canonical_sha256(staging)
    assert not available.exists()

    verified = _publisher(root, plan).publish(rows, stop_after="verified")
    assert verified.state == "verified"
    assert verified.previous_manifest_sha256 == canonical_sha256(complete)
    assert not available.exists()

    final = _publisher(root, plan).publish(rows)
    assert final.state == "available"
    assert final.previous_manifest_sha256 == canonical_sha256(verified)
    assert not transaction.exists()
    assert available.is_dir()
    assert (available / "capture.jsonl").read_bytes().count(b"\n") == 2
    assert _read_manifest(available / "manifest.available.json") == final

    reused = _publisher(root, plan).publish(rows)
    assert reused == final


def test_corrupt_payload_is_quarantined_and_never_exposed_as_available(tmp_path):
    plan = _plan()
    rows = _captures(plan)
    root = tmp_path / "capture-cache"
    publisher = _publisher(root, plan)
    publisher.publish(rows, stop_after="complete")
    payload = root / ".transactions" / "transaction-0001" / "capture.jsonl"
    payload.write_bytes(payload.read_bytes().replace(b'"example-1"', b'"example-X"', 1))

    with pytest.raises(ArtifactCorruptionError, match="corrupt or mismatched"):
        _publisher(root, plan).publish(rows)

    quarantine = root / "quarantine" / "shard-0001.transaction-0001"
    assert quarantine.is_dir()
    assert (quarantine / "manifest.quarantined.json").is_file()
    assert not (root / "shards" / "shard-0001").exists()
    assert not (root / ".transactions" / "transaction-0001").exists()


def test_resume_refuses_changed_rows_instead_of_mixing_transactions(tmp_path):
    plan = _plan()
    rows = _captures(plan)
    root = tmp_path / "capture-cache"
    _publisher(root, plan).publish(rows, stop_after="staging")
    changed = tuple(
        row.model_copy(update={"example_id": "example-2"})
        for row in rows
    )

    with pytest.raises(ArtifactCorruptionError, match="corrupt or mismatched"):
        _publisher(root, plan).publish(changed)

    assert (root / "quarantine" / "shard-0001.transaction-0001").is_dir()


def test_publisher_rejects_duplicate_reordered_and_wrong_policy_rows(tmp_path):
    plan = _plan()
    rows = _captures(plan)
    root = tmp_path / "capture-cache"
    reordered = (rows[1], rows[0])
    with pytest.raises(ValueError, match="positions must be contiguous"):
        _publisher(root, plan).publish(reordered)

    wrong_temperature_payload = rows[0].model_dump(by_alias=True)
    wrong_temperature_payload["temperature"] = 2.0
    selected_mass = sum(
        math.exp(value)
        for value in wrong_temperature_payload["teacher_log_probabilities"]
    )
    wrong_temperature_payload["tail_mass"] = 1.0 - selected_mass
    wrong_temperature = CaptureToken.model_validate(wrong_temperature_payload)
    with pytest.raises(ValueError, match="temperature does not match plan"):
        _publisher(root, plan).publish((wrong_temperature,))


def test_plan_file_binds_publication_root_before_any_transaction(tmp_path):
    plan = _plan()
    rows = _captures(plan)
    root = tmp_path / "capture-cache"
    _publisher(root, plan).publish(rows, stop_after="staging")
    payload = json.loads((FIXTURES / "plan.json").read_text(encoding="utf-8"))
    payload["run_id"] = "different-run"
    changed_plan = AutoDistillPlan.model_validate(payload)

    other = CaptureShardPublisher(
        root=root,
        plan=changed_plan,
        shard_id="shard-0002",
        transaction_id="transaction-0002",
    )
    with pytest.raises(ArtifactCorruptionError, match="different plan"):
        other.publish(_captures(changed_plan), stop_after="staging")
