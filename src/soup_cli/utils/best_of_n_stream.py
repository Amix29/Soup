"""Bounded-memory persistence for two-phase Best-of-N workflows."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from soup_cli.utils import best_of_n_artifact as artifact
from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

_CHECKPOINT_SCHEMA = "soup.best_of_n.candidate_checkpoint.v1"


def _write_all(fd: int, data: bytes) -> None:
    """Write one checkpoint record completely before it can be fsynced."""
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("candidate checkpoint write made no progress")
        view = view[written:]


def _run_digest(prompts: list[str], sampler: dict) -> str:
    payload = {"prompts": prompts, "sampler": sampler}
    data = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _parse_line(raw: bytes, field: str, line_number: int) -> dict:
    try:
        row = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} has invalid UTF-8 JSON on line {line_number}") from exc
    if not isinstance(row, dict):
        raise ValueError(f"{field} line {line_number} must be an object")
    return row


@contextmanager
def _regular_binary_lines(path: str, field: str) -> Iterator[Iterator[tuple[int, bytes]]]:
    enforce_under_cwd_and_no_symlink(path, field)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{field} could not be opened safely: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"{field} must be a regular file")
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            yield enumerate(handle, 1)
    finally:
        if fd >= 0:
            os.close(fd)


def _checkpoint_header(prompts: list[str], sampler: dict) -> dict:
    return {
        "_best_of_n_checkpoint": {
            "schema": _CHECKPOINT_SCHEMA,
            "run_digest": _run_digest(prompts, sampler),
            "prompt_count": len(prompts),
            "sampler": sampler,
        }
    }


def _discard_incomplete_checkpoint_tail(path: str) -> None:
    """Discard a final record that never reached its newline commit boundary."""
    enforce_under_cwd_and_no_symlink(path, "--checkpoint path")
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("--checkpoint path must be a regular file")
        size = os.lseek(fd, 0, os.SEEK_END)
        if size == 0:
            return
        os.lseek(fd, size - 1, os.SEEK_SET)
        if os.read(fd, 1) == b"\n":
            return

        cursor = size
        truncate_at = 0
        while cursor:
            chunk_start = max(0, cursor - 8192)
            os.lseek(fd, chunk_start, os.SEEK_SET)
            chunk = os.read(fd, cursor - chunk_start)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                truncate_at = chunk_start + newline + 1
                break
            cursor = chunk_start
        os.ftruncate(fd, truncate_at)
        os.fsync(fd)
    finally:
        os.close(fd)


def prepare_candidate_checkpoint(
    path: str, prompts: list[str], sampler: dict, *, resume: bool
) -> int:
    """Create or authenticate a checkpoint and return completed group count."""
    sampler = artifact.validate_sampler_spec(sampler)
    enforce_under_cwd_and_no_symlink(path, "--checkpoint path")
    expected_header = _checkpoint_header(prompts, sampler)
    exists = os.path.lexists(path)
    if not exists:
        if resume:
            raise ValueError("--resume requires an existing candidate checkpoint")
        parent = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(parent, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        fd = os.open(path, flags, 0o600)
        try:
            _write_all(fd, artifact.stable_json_line(expected_header))
            os.fsync(fd)
        finally:
            os.close(fd)
        return 0
    if not resume:
        raise ValueError("candidate checkpoint already exists; pass --resume to continue")

    _discard_incomplete_checkpoint_tail(path)
    completed = 0
    saw_header = False
    with _regular_binary_lines(path, "--checkpoint path") as lines:
        for line_number, raw in lines:
            if not raw.strip():
                continue
            row = _parse_line(raw, "candidate checkpoint", line_number)
            if not saw_header:
                if row != expected_header:
                    raise ValueError("candidate checkpoint does not match this run")
                saw_header = True
                continue
            if completed >= len(prompts):
                raise ValueError("candidate checkpoint has too many groups")
            artifact.validate_candidate_group(
                row, completed, sampler, expected_prompt=prompts[completed]
            )
            completed += 1
    if not saw_header:
        raise ValueError("candidate checkpoint header is missing")
    return completed


def append_candidate_group(path: str, group: dict) -> None:
    """Durably append one complete candidate group."""
    enforce_under_cwd_and_no_symlink(path, "--checkpoint path")
    flags = (
        os.O_WRONLY
        | os.O_APPEND
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("--checkpoint path must be a regular file")
        _write_all(fd, artifact.stable_json_line(group))
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_candidate_checkpoint(
    checkpoint: str,
    output: str,
    prompts: list[str],
    sampler: dict,
) -> None:
    """Stream a complete checkpoint into one atomically published artifact."""
    enforce_under_cwd_and_no_symlink(output, "--export-candidates path")
    parent = os.path.dirname(os.path.abspath(output)) or "."
    os.makedirs(parent, exist_ok=True)
    fd, staging = tempfile.mkstemp(prefix=".soup-bon-candidates.", dir=parent)
    try:
        with os.fdopen(fd, "wb") as target:
            fd = -1
            target.write(
                artifact.stable_json_line(
                    artifact.candidate_artifact_header(len(prompts), sampler)
                )
            )
            completed = 0
            saw_header = False
            with _regular_binary_lines(checkpoint, "--checkpoint path") as lines:
                for line_number, raw in lines:
                    if not raw.strip():
                        continue
                    row = _parse_line(raw, "candidate checkpoint", line_number)
                    if not saw_header:
                        if row != _checkpoint_header(prompts, sampler):
                            raise ValueError("candidate checkpoint does not match this run")
                        saw_header = True
                        continue
                    if completed >= len(prompts):
                        raise ValueError("candidate checkpoint has too many groups")
                    artifact.validate_candidate_group(
                        row, completed, sampler, expected_prompt=prompts[completed]
                    )
                    target.write(artifact.stable_json_line(row))
                    completed += 1
            if not saw_header or completed != len(prompts):
                raise ValueError("candidate checkpoint is incomplete")
            target.flush()
            os.fsync(target.fileno())
        enforce_under_cwd_and_no_symlink(output, "--export-candidates path")
        os.replace(staging, output)
    finally:
        if fd >= 0:
            os.close(fd)
        if os.path.exists(staging):
            os.unlink(staging)


@dataclass
class OfflineArtifactIndex:
    """Disk-backed authenticated mapping of groups to judgments."""

    connection: sqlite3.Connection
    sampler: dict
    candidate_sha256: str
    judgments_sha256: str
    group_count: int

    def iter_rows(self) -> Iterator[tuple[dict, dict | None]]:
        query = """
            SELECT groups.position, groups.payload, judgments.payload
            FROM groups JOIN judgments USING (prompt_id)
            ORDER BY groups.position
        """
        for position, group_raw, judgment_raw in self.connection.execute(query):
            group = json.loads(group_raw)
            judgment = artifact.validate_judgment(
                json.loads(judgment_raw), group, position
            )
            yield artifact.materialize_group(
                group,
                judgment,
                sampler=self.sampler,
                candidate_artifact_sha256=self.candidate_sha256,
                judgments_sha256=self.judgments_sha256,
            )

    def validate_all(self) -> None:
        count = sum(1 for _ in self.iter_rows())
        if count != self.group_count:
            raise ValueError("judgments must cover every candidate group exactly once")


def _index_candidates(connection: sqlite3.Connection, path: str) -> tuple[dict, str, int]:
    digest = hashlib.sha256()
    header = None
    sampler = None
    count = 0
    with _regular_binary_lines(path, "--candidate-artifact path") as lines:
        for line_number, raw in lines:
            digest.update(raw)
            if not raw.strip():
                continue
            row = _parse_line(raw, "candidate artifact", line_number)
            if header is None:
                header = row.get("_best_of_n_candidates")
                if not isinstance(header, dict) or header.get("schema") != (
                    "soup.best_of_n.candidates.v1"
                ):
                    raise ValueError("candidate artifact header or schema is invalid")
                sampler = artifact.validate_sampler_spec(header.get("sampler"))
                continue
            assert sampler is not None
            prompt_id = artifact.validate_candidate_group(row, count, sampler)
            try:
                connection.execute(
                    "INSERT INTO groups(position, prompt_id, payload) VALUES (?, ?, ?)",
                    (count, prompt_id, artifact.stable_json_line(row).decode("utf-8")),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"candidate group {count} has an invalid prompt id") from exc
            count += 1
    if header is None or sampler is None:
        raise ValueError("candidate artifact is empty")
    if not count:
        raise ValueError("candidate artifact contains no prompt groups")
    if header.get("prompt_count") != count:
        raise ValueError("candidate artifact header does not match its groups")
    return sampler, digest.hexdigest(), count


def _index_judgments(connection: sqlite3.Connection, path: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    with _regular_binary_lines(path, "--judgments path") as lines:
        for line_number, raw in lines:
            digest.update(raw)
            if not raw.strip():
                continue
            row = _parse_line(raw, "judgments", line_number)
            prompt_id = row.get("prompt_id")
            if not isinstance(prompt_id, str):
                raise ValueError("judgments contain a missing or duplicate prompt_id")
            try:
                connection.execute(
                    "INSERT INTO judgments(prompt_id, payload) VALUES (?, ?)",
                    (prompt_id, artifact.stable_json_line(row).decode("utf-8")),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    "judgments contain a missing or duplicate prompt_id"
                ) from exc
            count += 1
    return digest.hexdigest(), count


@contextmanager
def index_offline_artifacts(
    candidate_path: str, judgments_path: str
) -> Iterator[OfflineArtifactIndex]:
    """Authenticate large offline artifacts through a temporary disk index."""
    with tempfile.TemporaryDirectory(prefix=".soup-bon-index.", dir=os.getcwd()) as temp:
        database = os.path.join(temp, "index.sqlite3")
        connection = sqlite3.connect(database)
        try:
            connection.execute(
                "CREATE TABLE groups(position INTEGER PRIMARY KEY, "
                "prompt_id TEXT NOT NULL UNIQUE, payload TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE judgments(prompt_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            sampler, candidate_sha, group_count = _index_candidates(
                connection, candidate_path
            )
            judgments_sha, judgment_count = _index_judgments(connection, judgments_path)
            connection.commit()
            matched = connection.execute(
                "SELECT COUNT(*) FROM groups JOIN judgments USING (prompt_id)"
            ).fetchone()[0]
            if judgment_count != group_count or matched != group_count:
                raise ValueError("judgments must cover every candidate group exactly once")
            yield OfflineArtifactIndex(
                connection=connection,
                sampler=sampler,
                candidate_sha256=candidate_sha,
                judgments_sha256=judgments_sha,
                group_count=group_count,
            )
        finally:
            connection.close()


@dataclass
class StagedDatasets:
    sft_temp: str
    dpo_temp: str
    sft_sha256: str
    dpo_sha256: str
    sft_count: int
    dpo_count: int

    def cleanup(self) -> None:
        for path in (self.sft_temp, self.dpo_temp):
            if path and os.path.exists(path):
                os.unlink(path)


def _staging_file(output: str, field: str) -> tuple[int, str]:
    enforce_under_cwd_and_no_symlink(output, field)
    parent = os.path.dirname(os.path.abspath(output)) or "."
    os.makedirs(parent, exist_ok=True)
    return tempfile.mkstemp(prefix=".soup-bon-output.", dir=parent)


def stage_offline_datasets(
    index: OfflineArtifactIndex, sft_path: str, dpo_path: str
) -> StagedDatasets:
    """Stream authenticated rows to unpublished files and compute exact digests."""
    sft_fd, sft_temp = _staging_file(sft_path, "--output path")
    dpo_fd = -1
    dpo_temp = ""
    sft_hash = hashlib.sha256()
    dpo_hash = hashlib.sha256()
    sft_count = 0
    dpo_count = 0
    try:
        if dpo_path:
            dpo_fd, dpo_temp = _staging_file(dpo_path, "--emit-pairs path")
        with os.fdopen(sft_fd, "wb") as sft_handle:
            sft_fd = -1
            dpo_handle = os.fdopen(dpo_fd, "wb") if dpo_fd >= 0 else None
            if dpo_handle is not None:
                dpo_fd = -1
            try:
                for sft, dpo in index.iter_rows():
                    sft_line = artifact.stable_json_line(sft)
                    sft_handle.write(sft_line)
                    sft_hash.update(sft_line)
                    sft_count += 1
                    if dpo_handle is not None and dpo is not None:
                        dpo_line = artifact.stable_json_line(dpo)
                        dpo_handle.write(dpo_line)
                        dpo_hash.update(dpo_line)
                        dpo_count += 1
                sft_handle.flush()
                os.fsync(sft_handle.fileno())
                if dpo_handle is not None:
                    dpo_handle.flush()
                    os.fsync(dpo_handle.fileno())
            finally:
                if dpo_handle is not None:
                    dpo_handle.close()
        return StagedDatasets(
            sft_temp=sft_temp,
            dpo_temp=dpo_temp,
            sft_sha256=sft_hash.hexdigest(),
            dpo_sha256=dpo_hash.hexdigest(),
            sft_count=sft_count,
            dpo_count=dpo_count,
        )
    except Exception:
        for path in (sft_temp, dpo_temp):
            if path and os.path.exists(path):
                os.unlink(path)
        raise
    finally:
        if sft_fd >= 0:
            os.close(sft_fd)
        if dpo_fd >= 0:
            os.close(dpo_fd)


def publish_staged_datasets(
    staged: StagedDatasets, sft_path: str, dpo_path: str
) -> None:
    """Replace the requested outputs with already validated staging files."""
    enforce_under_cwd_and_no_symlink(sft_path, "--output path")
    os.replace(staged.sft_temp, sft_path)
    staged.sft_temp = ""
    if dpo_path:
        enforce_under_cwd_and_no_symlink(dpo_path, "--emit-pairs path")
        os.replace(staged.dpo_temp, dpo_path)
        staged.dpo_temp = ""
