"""MLX QAT accuracy + speed on Apple Silicon: plain LoRA vs QAT, through the merged_4bit save.

    python jobs/mlx_qat_bench.py [--model M] [--steps N] [--arms lora,qat,qat_dense] [--out f.json]

Per arm, same seed / data / LoRA init: held-out wikitext-2 CE before training, after training
(in memory), and after save_pretrained_merged(merged_4bit) + reload; compiled step time, tokens/s,
peak memory. `qat_dense` = the PR's original forward (dense GEMM over the fake-quantized weight).
Prints `JOB_RESULT {json}`; exit 0 unless an arm crashed.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import tempfile
import time
import traceback


def _wikitext(tokenizer, split, n_rows, seq):
    from datasets import load_dataset
    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "".join(r["text"] for r in rows if r["text"].strip())
    ids = tokenizer.encode(text)
    return [ids[i * seq:(i + 1) * seq] for i in range(min(n_rows, len(ids) // seq))]


def _dense_qat_call(self, x):
    import mlx.core as mx
    from unsloth_zoo.mlx import qat as Q
    base = self.linear
    g, b, m = Q._quantization_grid(base)
    w = mx.dequantize(base.weight, base.scales, base.biases, group_size=g, bits=b, mode=m)
    merged = w + ((self.scale * self.lora_b.T) @ self.lora_a.T).astype(w.dtype)
    p, s, bi = mx.quantize(merged, group_size=g, bits=b, mode=m)
    fake = mx.dequantize(p, s, bi, group_size=g, bits=b, mode=m)
    y = x @ (merged + mx.stop_gradient(fake - merged)).T
    return y + base.bias if "bias" in base else y


def run_arm(arm, a, train, evals):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from unsloth_zoo.mlx import qat as Q
    from unsloth_zoo.mlx.loader import FastMLXModel

    def ce(model, batch):
        x = mx.array(batch)
        logits = model(x[:, :-1]).astype(mx.float32)
        return nn.losses.cross_entropy(logits, x[:, 1:]).mean()

    def eval_ce(model):
        vals = [float(ce(model, evals[i:i + a.batch])) for i in range(0, len(evals), a.batch)]
        return statistics.fmean(vals)

    original_call = Q._qat_call
    if arm == "qat_dense":
        Q._qat_call = _dense_qat_call
    try:
        mx.random.seed(a.seed)
        model, tok = FastMLXModel.from_pretrained(a.model, max_seq_length=a.seq)
        before = eval_ce(model)
        kw = {"qat_scheme": "auto"} if arm.startswith("qat") else {}
        mx.random.seed(a.seed)
        model = FastMLXModel.get_peft_model(
            model, r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0,
            use_gradient_checkpointing="mlx", random_state=a.seed, **kw)
        opt = optim.Adam(learning_rate=a.lr)
        loss_and_grad = nn.value_and_grad(model, ce)
        state = [model.state, opt.state]

        def _step(batch):
            loss, grads = loss_and_grad(model, batch)
            opt.update(model, grads)
            return loss
        step = mx.compile(_step, inputs=state, outputs=state)

        mx.reset_peak_memory()
        times, losses = [], []
        for i in range(a.steps):
            batch = mx.array(train[(i * a.batch) % (len(train) - a.batch):][:a.batch])
            t = time.perf_counter()
            loss = step(batch)
            mx.eval(state, loss)
            times.append(time.perf_counter() - t)
            losses.append(float(loss))
        warm = times[min(5, len(times) - 1):]
        step_s = statistics.median(warm)
        peak = mx.get_peak_memory() / 2**30
        trained = eval_ce(model)
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained_merged(d, tok, save_method="merged_4bit")
            del model
            gc.collect()
            reloaded, _ = FastMLXModel.from_pretrained(d, max_seq_length=a.seq)
            saved = eval_ce(reloaded)
        return {
            "arm": arm, "eval_ce_base": before, "eval_ce_trained": trained,
            "eval_ce_saved": saved, "save_drift": saved - trained,
            "kept_fraction": (before - saved) / (before - trained) if before != trained else None,
            "median_step_s": step_s, "tokens_per_s": a.batch * (a.seq - 1) / step_s,
            "peak_mem_gb": peak, "first_step_s": times[0],
            "train_loss_first_last": [losses[0], losses[-1]],
        }
    finally:
        Q._qat_call = original_call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=257)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--seeds", default=None, help="comma list; overrides --seed")
    ap.add_argument("--eval-rows", type=int, default=64)
    ap.add_argument("--arms", default="lora,qat,qat_dense")
    ap.add_argument("--out", default="mlx_qat_bench.json")
    ap.add_argument("--tiny", action="store_true", help="ignored (staging_ci)")
    a = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    _, tok = load(a.model)
    train = _wikitext(tok, "train", a.steps * a.batch + a.batch, a.seq)
    evals = _wikitext(tok, "test", a.eval_rows, a.seq)
    res = {"job": "mlx_qat_bench", "model": a.model, "config": vars(a),
           "device": str(mx.default_device()), "arms": {}, "errors": {}}
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else [a.seed]
    for seed in seeds:
        a.seed = seed
        for arm in a.arms.split(","):
            key = f"{arm}@{seed}"
            try:
                res["arms"][key] = run_arm(arm, a, train, evals)
                print("ARM", key, json.dumps(res["arms"][key]), flush=True)
            except Exception as e:  # noqa: BLE001 - record and continue with the other arms
                res["errors"][key] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
                print("ARM_ERROR", key, res["errors"][key], flush=True)
            gc.collect()
            mx.clear_cache()
    summary = {}
    for arm in a.arms.split(","):
        rows = [v for k, v in res["arms"].items() if k.startswith(arm + "@")]
        if rows:
            summary[arm] = {m: (statistics.fmean(r[m] for r in rows),
                                statistics.pstdev(r[m] for r in rows))
                            for m in ("eval_ce_trained", "eval_ce_saved", "save_drift",
                                      "median_step_s", "peak_mem_gb")}
    res["summary"] = summary
    print("SUMMARY", json.dumps(summary), flush=True)
    res["passed"] = not res["errors"]
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("JOB_RESULT " + json.dumps({k: res[k] for k in ("job", "model", "passed", "summary")}))
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
