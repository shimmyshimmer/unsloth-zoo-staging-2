# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""PR 1189 on a real mlx-community checkpoint: same tokens, fewer gather_qmm calls.

Two models, one at a time (the runner has 7 GB of RAM):
  * a dense control, where the scope must find nothing and change nothing;
  * a real quantized MoE, where the scope must fuse at least one block, keep
    greedy output byte-identical, and cut gather_qmm calls per forward.

A run that reports zero fused blocks on the MoE model is a FAILURE, not a pass:
equal output with the fusion never entered proves nothing.
"""

import argparse
import json
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate

from unsloth_zoo.mlx.inference import (
    _MOE_GATE_UP_CLASSES, _PackedMoEGateUp, fused_moe_gate_up,
)

PROMPT = "List three prime numbers."
MAX_TOKENS = 24


def _greedy(model, tokenizer):
    return generate(model, tokenizer, prompt = PROMPT, max_tokens = MAX_TOKENS, verbose = False)


def _count_gather_qmm(fn):
    """Run `fn` with mx.gather_qmm counted."""
    calls = [0]
    original = mx.gather_qmm

    def counting(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    mx.gather_qmm = counting
    try:
        result = fn()
    finally:
        mx.gather_qmm = original
    return result, calls[0]


def _fused_module_count(model):
    fused_classes = set(_MOE_GATE_UP_CLASSES.values())
    return sum(1 for _, module in model.named_modules() if type(module) in fused_classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required = True)
    parser.add_argument("--revision", default = None)
    parser.add_argument("--expect-moe", action = "store_true",
                        help = "fail unless at least one block actually fuses")
    args = parser.parse_args()

    kwargs = {"revision": args.revision} if args.revision else {}
    model, tokenizer = load(args.model, **kwargs)

    native_text, native_calls = _count_gather_qmm(lambda: _greedy(model, tokenizer))
    with fused_moe_gate_up(model) as scoped:
        fused_blocks = _fused_module_count(scoped)
        fused_text, fused_calls = _count_gather_qmm(lambda: _greedy(scoped, tokenizer))
    restored = _fused_module_count(model)
    leftover = sum(1 for _, m in model.named_modules()
                   if isinstance(getattr(m, "_unsloth_moe_gate_up", None), _PackedMoEGateUp))

    report = {
        "model": args.model,
        "fused_blocks": fused_blocks,
        "still_fused_after_exit": restored,
        "packed_attrs_after_exit": leftover,
        "gather_qmm_native": native_calls,
        "gather_qmm_fused": fused_calls,
        "text_identical": native_text == fused_text,
        "native_text": native_text,
        "fused_text": fused_text,
    }
    print(json.dumps(report, indent = 2))

    failures = []
    if not report["text_identical"]:
        failures.append("greedy output changed under the fusion scope")
    if restored:
        failures.append(f"{restored} modules left fused after the scope exited")
    if leftover:
        failures.append(f"{leftover} modules kept _unsloth_moe_gate_up after exit")
    if args.expect_moe:
        if fused_blocks == 0:
            failures.append("no MoE block was eligible; the fusion never ran")
        elif fused_calls >= native_calls:
            failures.append(
                f"gather_qmm calls did not drop ({native_calls} -> {fused_calls})")
    else:
        if fused_blocks:
            failures.append(f"dense control fused {fused_blocks} blocks")
        if fused_calls != native_calls:
            failures.append(
                f"dense control changed gather_qmm calls ({native_calls} -> {fused_calls})")

    for failure in failures:
        print(f"::error::{failure}")
    print("RESULT_1189:", "FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
