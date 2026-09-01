"""Run a tiny, reproducible B1 teacher-capture smoke on a local MLX checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

from rich.console import Console

from soup_cli.autodistill.contract import (
    AutoDistillPlan,
    FileDigest,
    build_plan_estimate,
    canonical_json_bytes,
    canonicalize_jsonl_bytes,
)
from soup_cli.autodistill.mlx_worker import run_mlx_teacher_capture_process

console = Console()


def _digest(path: Path, root: Path) -> FileDigest:
    payload = path.read_bytes()
    return FileDigest(
        path=path.relative_to(root).as_posix(),
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _quantization(config: dict[str, object]) -> str:
    active = {
        key: config[key]
        for key in ("quantization", "quantization_config")
        if config.get(key) is not None
    }
    if not active:
        return "none"
    digest = hashlib.sha256(canonical_json_bytes(active)).hexdigest()
    return f"config-sha256:{digest}"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--target", default=" Paris.")
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    model_root = Path(os.path.realpath(arguments.model_root))
    output_root = Path(os.path.realpath(arguments.output_root))
    if output_root.exists():
        raise FileExistsError("output root must not already exist")
    output_root.mkdir(parents=True)
    dataset_root = output_root / "dataset"
    publication_root = output_root / "publication"
    dataset_root.mkdir()

    config_path = model_root / "config.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if not isinstance(config, dict):
        raise ValueError("config.json must contain an object")
    dtype = str(config.get("torch_dtype", "")).removeprefix("torch.")
    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("checkpoint torch_dtype is not supported by the B1 plan")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
    prompt_ids = tuple(tokenizer.encode(arguments.prompt, add_special_tokens=False))
    full_ids = tuple(
        tokenizer.encode(arguments.prompt + arguments.target, add_special_tokens=False)
    )
    if not prompt_ids or full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt tokenization is not a prefix of prompt plus target")
    target_ids = full_ids[len(prompt_ids) :]
    if not target_ids:
        raise ValueError("target must produce at least one token")
    row = {
        "schema": "soup.autodistill.tokenized-teacher-example.v1",
        "example_id": "hardware-smoke-1",
        "prompt_token_ids": list(prompt_ids),
        "target_token_ids": list(target_ids),
    }
    dataset_bytes = canonical_json_bytes(row) + b"\n"
    dataset_path = dataset_root / "prompts.jsonl"
    dataset_path.write_bytes(dataset_bytes)

    weight_paths = sorted(model_root.glob("*.safetensors"))
    if not weight_paths:
        raise ValueError("checkpoint has no root-level safetensors weights")
    tokenizer_names = (
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    tokenizer_paths = [
        model_root / name for name in tokenizer_names if (model_root / name).is_file()
    ]
    if not tokenizer_paths:
        raise ValueError("checkpoint has no recognized tokenizer files")
    teacher = {
        "model_id": arguments.model_id,
        "revision": arguments.revision,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "weights": [
            item.model_dump(mode="json")
            for item in (_digest(path, model_root) for path in weight_paths)
        ],
    }
    tokenizer_fingerprint = {
        "tokenizer_id": arguments.model_id,
        "revision": arguments.revision,
        "vocab_size": int(config["vocab_size"]),
        "files": [
            item.model_dump(mode="json")
            for item in (_digest(path, model_root) for path in tokenizer_paths)
        ],
        "chat_template_sha256": hashlib.sha256(
            (tokenizer.chat_template or "").encode("utf-8")
        ).hexdigest(),
        "renderer": f"mlx-lm@{version('mlx-lm')}",
    }
    estimate = build_plan_estimate(
        token_count=len(target_ids),
        vocab_size=int(config["vocab_size"]),
        top_k=arguments.top_k,
        max_forced_tokens_per_position=2,
        token_id_bytes=4,
        log_probability_bytes=4,
        tail_mass_bytes=8,
        entropy_bytes=8,
    )
    plan = AutoDistillPlan.model_validate(
        {
            "schema": "soup.autodistill.plan.v1",
            "run_id": "mlx-hardware-smoke-1",
            "capture_boundary": "same_tokenizer",
            "teacher": teacher,
            "student": teacher,
            "tokenizer": tokenizer_fingerprint,
            "dataset": {
                "normalization": "soup-jsonl-c14n-v1",
                "normalized_sha256": hashlib.sha256(
                    canonicalize_jsonl_bytes(dataset_bytes)
                ).hexdigest(),
                "rows": 1,
                "source_files": [_digest(dataset_path, dataset_root).model_dump(mode="json")],
            },
            "capture": {
                "planned_token_count": len(target_ids),
                "vocab_size": int(config["vocab_size"]),
                "max_forced_tokens_per_position": 2,
                "backend": "mlx",
                "backend_version": version("mlx-lm"),
                "dtype": dtype,
                "quantization": _quantization(config),
                "max_sequence_length": int(config.get("max_position_embeddings", 2048)),
                "truncation": "none",
            },
            "probability_policy": {
                "name": "topk_union_forced_tail.v1",
                "top_k": arguments.top_k,
                "forced_token_sources": ["target", "student_sample"],
                "token_id_bytes": 4,
                "log_probability_bytes": 4,
                "tail_mass_bytes": 8,
                "entropy_bytes": 8,
                "temperature": 1.0,
                "renormalize_selected": False,
            },
            "consumption_policy": {
                "teacher_expert_replay": "explicit",
                "student_rollout_replay": "forbidden",
                "reservation_recovery": "release_if_checkpoint_absent",
                "commit_requires_checkpoint_sha256": True,
            },
            "throughput_profile": None,
            "estimate": estimate.model_dump(mode="json"),
        }
    )
    plan_path = output_root / "plan.json"
    plan_path.write_bytes(canonical_json_bytes(plan) + b"\n")

    started = time.monotonic()
    result = run_mlx_teacher_capture_process(
        plan=plan,
        teacher_root=model_root,
        tokenizer_root=model_root,
        dataset_root=dataset_root,
        publication_root=publication_root,
        shard_id="shard-0001",
        transaction_id="transaction-0001",
        python_executable=os.fspath(Path(os.sys.executable)),
        timeout_seconds=300,
    )
    elapsed = time.monotonic() - started
    console.print(
        {
            "status": "PASS",
            "elapsed_seconds": round(elapsed, 3),
            "prompt_token_ids": prompt_ids,
            "target_token_ids": target_ids,
            "plan": os.fspath(plan_path),
            "publication": os.fspath(publication_root),
            "result": result.model_dump(mode="json", by_alias=True),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
