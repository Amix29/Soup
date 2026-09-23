#!/usr/bin/env python3
"""Per-layer CPU NF4 reconstruction for #379 / STEP 13.

This is NOT a replay of the full-model tiny-Llama logits numbers recorded in
``tests/test_v07202.py`` (1.948e-03 at 8 tokens, ~1.972e-03 from 32..256 on the
H100 host CPU, and 2.509e-03 on CI). Those numbers compare a streamed two-layer
model with a resident model.

Instead this harness reconstructs, one ``Linear4bit`` at a time, the narrower
question assigned to the historical ``cpu_mode_probe.py``:
does bitsandbytes' CPU *inference* packing path create a numerical difference
that is absent from the training-style path?

Requirements / attribution boundary
-----------------------------------
- AVX512-BF16 is required for the mechanism this gate is intended to
  demonstrate. Without it, the default invocation FAILS; use ``--survey`` for a
  capability-only report.
- On hosts where the optional ``kernels`` package is installed, bitsandbytes
  0.50.x may fetch/use ``kernels-community/quantization-bitsandbytes`` on the
  packed inference path. ``packing_format_for_cpu`` proves that the packed CPU
  inference path ran; it does NOT identify which lower-level kernel executed.
  This harness therefore makes no native-kernel attribution and no "no
  downloads" promise.
- The "variant 2" arm imports Soup's shipped ``install_dequant_forward``
  unmodified. NF4 uses ``compress_statistics=True`` (double quant), matching
  layer-stream shards.

Default invocation is a mechanism gate:
1. at least one inference row must enter CPU packing;
2. the training arm must stay unpacked and equal Soup variant 2 EXACTLY;
3. every packed inference row must differ from variant 2.

Use ``--survey`` to print the same JSON while always exiting 0.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from dataclasses import asdict, dataclass

M_VALUES = (8, 16, 32, 64, 128, 256)
FIXTURE_SHAPES = ((64, 64), (256, 64))

EXIT_OK = 0
EXIT_MECHANISM_ABSENT = 2
EXIT_CONTROL_FAILED = 3
EXIT_EFFECT_ABSENT = 4


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
        weight,
        blocksize=64,
        compress_statistics=True,
        quant_type="nf4",
    )
    layer = bnb.nn.Linear4bit(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        compute_dtype=compute_dtype,
        compress_statistics=True,
        quant_type="nf4",
        device="cpu",
    )
    layer.weight = bnb.nn.Params4bit(
        data=packed,
        requires_grad=False,
        quant_state=state,
        blocksize=state.blocksize,
        compress_statistics=True,
        quant_type=state.quant_type,
        bnb_quantized=True,
    )
    return layer


def measure_row(out_features: int, in_features: int, m: int, *, seed: int = 17) -> Row:
    import torch

    from soup_cli.utils.layer_stream_runtime import install_dequant_forward

    generator = torch.Generator().manual_seed(seed + out_features + m)
    weight = torch.randn(
        (out_features, in_features),
        generator=generator,
        dtype=torch.float32,
    )
    x = torch.randn(
        (m, in_features),
        generator=generator,
        dtype=torch.float32,
    )

    variant2_layer = _quantized_linear(weight, compute_dtype=torch.float32)
    patched = install_dequant_forward(variant2_layer)
    if patched != 1:
        raise RuntimeError(
            f"install_dequant_forward patched {patched} layers, expected 1"
        )
    variant2_layer.train()
    variant2_x = x.clone().requires_grad_(True)
    variant2 = variant2_layer(variant2_x).detach()

    training_layer = _quantized_linear(weight, compute_dtype=torch.float32)
    training_layer.train()
    training_x = x.clone().requires_grad_(True)
    training = training_layer(training_x).detach()
    training_packed = bool(
        getattr(
            training_layer.weight.quant_state,
            "packing_format_for_cpu",
            False,
        )
    )

    inference_layer = _quantized_linear(weight, compute_dtype=torch.float32)
    inference_layer.eval()
    with torch.no_grad():
        inference = inference_layer(x)
    inference_packed = bool(
        getattr(
            inference_layer.weight.quant_state,
            "packing_format_for_cpu",
            False,
        )
    )

    return Row(
        out_features=out_features,
        in_features=in_features,
        m=m,
        inference_packed_for_cpu=inference_packed,
        training_packed_for_cpu=training_packed,
        inference_vs_variant2_max_abs=float(
            (inference - variant2).abs().max()
        ),
        training_vs_variant2_max_abs=float(
            (training - variant2).abs().max()
        ),
        inference_vs_training_max_abs=float(
            (inference - training).abs().max()
        ),
    )


def _evaluate_rows(rows: list[Row]) -> dict:
    packed_rows = [row for row in rows if row.inference_packed_for_cpu]

    if not packed_rows:
        return {
            "verdict": "mechanism_absent",
            "exit_code": EXIT_MECHANISM_ABSENT,
            "reason": (
                "CPU inference packing did not run; this host cannot demonstrate "
                "the cpu_mode_probe mechanism."
            ),
            "attribution": "none",
        }

    if any(row.training_packed_for_cpu for row in rows) or any(
        row.training_vs_variant2_max_abs != 0.0 for row in rows
    ):
        return {
            "verdict": "control_failed",
            "exit_code": EXIT_CONTROL_FAILED,
            "reason": (
                "training-style Linear4bit did not remain an exact unpacked "
                "control against Soup install_dequant_forward."
            ),
            "attribution": "invalid-control",
        }

    if any(
        row.inference_vs_variant2_max_abs == 0.0
        for row in packed_rows
    ):
        return {
            "verdict": "packed_effect_absent",
            "exit_code": EXIT_EFFECT_ABSENT,
            "reason": (
                "the packed inference path ran, but at least one packed row did "
                "not differ from Soup variant 2."
            ),
            "attribution": "packed-path-detected-no-universal-divergence",
        }

    return {
        "verdict": "mechanism_reproduced",
        "exit_code": EXIT_OK,
        "reason": (
            "packed inference differs on every packed row while the training "
            "control is exactly equal to Soup variant 2."
        ),
        "attribution": (
            "bitsandbytes packed CPU inference path; exact lower-level kernel "
            "not instrumented"
        ),
    }


def run_probe(*, m_values=M_VALUES, shapes=FIXTURE_SHAPES) -> dict:
    import bitsandbytes
    import bitsandbytes.functional as functional
    import torch

    rows = [
        measure_row(out_features, in_features, m)
        for out_features, in_features in shapes
        for m in m_values
    ]
    verdict = _evaluate_rows(rows)
    return {
        "torch": torch.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "platform": platform.platform(),
        "has_avx512bf16": bool(functional.has_avx512bf16()),
        "kernels_package_installed": (
            importlib.util.find_spec("kernels") is not None
        ),
        "rows": [asdict(row) for row in rows],
        "packed_inference_rows": sum(
            row.inference_packed_for_cpu for row in rows
        ),
        "packed_training_rows": sum(
            row.training_packed_for_cpu for row in rows
        ),
        "training_control_exact": all(
            row.training_vs_variant2_max_abs == 0.0 for row in rows
        ),
        **verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="optional JSON output path")
    parser.add_argument(
        "--survey",
        action="store_true",
        help=(
            "capability-only survey: print verdict JSON but exit 0 even when "
            "the mechanism is absent or its controls fail"
        ),
    )
    args = parser.parse_args(argv)

    result = run_probe()
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(
            payload + "\n",
            encoding="utf-8",
        )

    if args.survey:
        return EXIT_OK

    code = int(result["exit_code"])
    if code != EXIT_OK:
        print(str(result["reason"]), file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
