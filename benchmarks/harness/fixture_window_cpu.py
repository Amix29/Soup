#!/usr/bin/env python3
"""STEP 13 CPU NF4 fixture-window probe (#379).

Reconstructs the recorded CPU-side question from gate-h100-validation.md:
is the CI-fixture divergence a size effect, or does it appear only when
bitsandbytes enters its CPU inference packing path?

No model download is required.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass

M_VALUES = (8, 16, 32, 64, 128, 256, 512)
FIXTURE_SHAPES = ((64, 64), (256, 64))


@dataclass(frozen=True)
class Row:
    out_features: int
    in_features: int
    m: int
    inference_packed_for_cpu: bool
    training_packed_for_cpu: bool
    inference_vs_variant2_max_abs: float
    training_vs_variant2_max_abs: float
    inference_vs_training_max_abs: float


def _quantized_linear(weight, *, compute_dtype):
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional

    packed, state = functional.quantize_4bit(
        weight, blocksize=64, compress_statistics=False, quant_type="nf4"
    )
    layer = bnb.nn.Linear4bit(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        compute_dtype=compute_dtype,
        compress_statistics=False,
        quant_type="nf4",
        device="cpu",
    )
    layer.weight = bnb.nn.Params4bit(
        data=packed,
        requires_grad=False,
        quant_state=state,
        blocksize=state.blocksize,
        compress_statistics=False,
        quant_type=state.quant_type,
        bnb_quantized=True,
    )
    return layer, packed, state


def measure_row(out_features: int, in_features: int, m: int, *, seed: int = 17) -> Row:
    import bitsandbytes.functional as functional
    import torch

    generator = torch.Generator().manual_seed(seed + out_features + m)
    weight = torch.randn((out_features, in_features), generator=generator, dtype=torch.float32)
    x = torch.randn((m, in_features), generator=generator, dtype=torch.float32)

    reference_layer, packed, state = _quantized_linear(weight, compute_dtype=torch.float32)
    del reference_layer
    dense = functional.dequantize_4bit(packed, state).to(x.dtype)
    variant2 = torch.nn.functional.linear(x, dense)


    training_layer, _, _ = _quantized_linear(weight, compute_dtype=torch.float32)
    training_layer.train()
    training_x = x.clone().requires_grad_(True)
    training = training_layer(training_x).detach()
    training_packed = bool(
        getattr(training_layer.weight.quant_state, "packing_format_for_cpu", False)
    )

    inference_layer, _, _ = _quantized_linear(weight, compute_dtype=torch.float32)
    inference_layer.eval()
    with torch.no_grad():
        inference = inference_layer(x)
    inference_packed = bool(
        getattr(inference_layer.weight.quant_state, "packing_format_for_cpu", False)
    )

    return Row(
        out_features=out_features,
        in_features=in_features,
        m=m,
        inference_packed_for_cpu=inference_packed,
        training_packed_for_cpu=training_packed,
        inference_vs_variant2_max_abs=float((inference - variant2).abs().max()),
        training_vs_variant2_max_abs=float((training - variant2).abs().max()),
        inference_vs_training_max_abs=float((inference - training).abs().max()),
    )


def run_probe(*, m_values=M_VALUES, shapes=FIXTURE_SHAPES) -> dict:
    # Keep metadata imports soft so the dependency-light contract tests can
    # monkeypatch measure_row without requiring bitsandbytes on the host.
    try:
        import bitsandbytes
        import bitsandbytes.functional as functional
    except ImportError:
        bitsandbytes = None
        functional = None
    try:
        import torch
    except ImportError:
        torch = None

    rows = [
        measure_row(out_features, in_features, m)
        for out_features, in_features in shapes
        for m in m_values
    ]
    return {
        "torch": getattr(torch, "__version__", None),
        "bitsandbytes": getattr(bitsandbytes, "__version__", None),
        "platform": platform.platform(),
        "has_avx512bf16": bool(functional.has_avx512bf16())
        if functional is not None
        else False,
        "rows": [asdict(row) for row in rows],
        "packed_inference_rows": sum(row.inference_packed_for_cpu for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="optional JSON output path")
    parser.add_argument(
        "--require-packed-inference",
        action="store_true",
        help="exit non-zero unless the CPU inference repack path actually ran",
    )
    args = parser.parse_args()

    result = run_probe()
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(payload + "\n", encoding="utf-8")

    if args.require_packed_inference and result["packed_inference_rows"] == 0:
        print(
            "CPU inference packing did not run on this host; "
            "no inference-path attribution can be made.",
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
