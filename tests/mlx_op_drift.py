# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""Which op is not reproducible on Metal?

The layerwise bisect put the first bit-difference between two identical
Gemma 4 E2B forwards at decoder layer 0, with a max absolute difference of
about 1.06 -- far too large to be float non-associativity, and upstream of
every KV-shared layer. So the answer is not architectural, it is an op.

This walks down from the layer to the op, each step run twice on the same
input in the same process:

  1. embed_tokens        -- a gather, no arithmetic
  2. layer 0 q_proj      -- one quantized matmul on real weights
  3. layer 0 self_attn   -- the attention block
  4. layer 0 whole       -- the decoder layer, the known-bad unit
  5. mx.quantized_matmul -- the bare op on synthetic operands
  6. mx.matmul           -- the same shape unquantized, as the control that
                            says whether quantization is the variable

The first step to report `differs` is the finding. If step 6 is stable and
step 5 is not, that is a quantized-matmul bug and belongs upstream in mlx
with no mlx-vlm or unsloth in the report at all.

Run: python tests/mlx_op_drift.py [--model REPO] [--rounds N]
"""

import argparse
import sys

DEFAULT_MODEL = "mlx-community/gemma-4-e2b-it-4bit"
PROMPT_IDS = [[2, 106, 1645, 108, 3689, 236743, 236764, 1902, 236764]]


def compare(name, fn, rounds):
    """Run fn() `rounds` times on identical input, report whether it moves."""
    import mlx.core as mx
    import numpy as np

    outs = []
    for _ in range(rounds):
        y = fn()
        mx.eval(y)
        outs.append(np.asarray(y.astype(mx.float32)))

    worst = 0.0
    for other in outs[1:]:
        d = np.abs(outs[0] - other)
        worst = max(worst, float(d.max()) if d.size else 0.0)
    finite = all(bool(np.isfinite(o).all()) for o in outs)
    verdict = "differs" if worst > 0.0 else "stable"
    print(f"  {name:<24} {verdict:<8} max|a-b|={worst:.6g} "
          f"finite={finite} shape={tuple(outs[0].shape)}", flush=True)
    return worst > 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rounds", type=int, default=4)
    args = ap.parse_args()

    import mlx.core as mx
    import mlx_vlm
    from mlx_vlm import load

    print(f"=== op drift: mlx {mx.__version__}, "
          f"mlx-vlm {getattr(mlx_vlm, '__version__', '?')}, {args.model} ===",
          flush=True)

    model, _ = load(args.model)
    lm = model.language_model
    inner = getattr(lm, "model", lm)
    layer0 = inner.layers[0]

    ids = mx.array(PROMPT_IDS)
    h = inner.embed_tokens(ids)
    mx.eval(h)
    print(f"embed dtype={h.dtype} shape={tuple(h.shape)}", flush=True)

    q_proj = layer0.self_attn.q_proj
    print(f"q_proj type={type(q_proj).__name__} "
          f"quantized={hasattr(q_proj, 'scales')}", flush=True)

    moved = []
    if compare("1_embed_tokens", lambda: inner.embed_tokens(ids), args.rounds):
        moved.append("embed_tokens")
    if compare("2_layer0_q_proj", lambda: q_proj(h), args.rounds):
        moved.append("q_proj")

    hn = layer0.input_layernorm(h)
    mx.eval(hn)

    def attn():
        out = layer0.self_attn(hn, None, None)
        return out[0] if isinstance(out, tuple) else out

    if compare("3_layer0_self_attn", attn, args.rounds):
        moved.append("self_attn")

    def whole():
        out = layer0(h, None, None)
        return out[0] if isinstance(out, tuple) else out

    if compare("4_layer0_decoder", whole, args.rounds):
        moved.append("decoder_layer")

    # Synthetic operands at the same shape, so a difference here is the op and
    # not anything the checkpoint brought with it.
    mx.random.seed(0)
    x = mx.random.normal((1, h.shape[1], h.shape[2])).astype(h.dtype)
    w = mx.random.normal((h.shape[2], h.shape[2])).astype(h.dtype)
    mx.eval(x, w)
    wq, scales, biases = mx.quantize(w, group_size=64, bits=4)
    mx.eval(wq, scales, biases)

    if compare("5_quantized_matmul",
               lambda: mx.quantized_matmul(
                   x, wq, scales, biases, transpose=True,
                   group_size=64, bits=4),
               args.rounds):
        moved.append("quantized_matmul")
    if compare("6_matmul_unquantized",
               lambda: x @ w.T, args.rounds):
        moved.append("matmul")

    print("", flush=True)
    if not moved:
        print("VERDICT every op stable; the drift is not in this set",
              flush=True)
    else:
        print(f"VERDICT unstable: {', '.join(moved)}", flush=True)
        if "quantized_matmul" in moved and "matmul" not in moved:
            print("quantized_matmul moves where the unquantized matmul does "
                  "not, so quantization is the variable and this is an mlx "
                  "bug reportable without mlx-vlm in the picture", flush=True)
        elif "matmul" in moved:
            print("even the unquantized matmul moves, so this is broader "
                  "than quantization", flush=True)
        elif "embed_tokens" in moved:
            print("a gather with no arithmetic moves, which points at memory "
                  "rather than at any kernel's math", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
