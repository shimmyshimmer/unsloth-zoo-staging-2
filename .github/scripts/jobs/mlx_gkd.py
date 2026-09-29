"""MLX off-policy GKD (unsloth-zoo#965) on Apple Silicon: LoRA student distilled from a frozen
teacher through zoo MLXTrainer, against a cross-entropy arm on the same data.

    python jobs/mlx_gkd.py              # SmolLM-135M-Instruct-4bit <- SmolLM-360M-Instruct-4bit, 10 steps

Arms (fresh student each): CE (compile), GKD eager, GKD compiled (+ eval).
Checks: steps, finite loss, bounded grad norm, LoRA changed, KL(teacher || student) falls,
teacher weights untouched, compiled == eager within Metal reduction noise, eval reports a loss
and no perplexity, GKD loss on real logits matches a float64 reference.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

DEFAULT_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
TEACHER = "mlx-community/SmolLM-360M-Instruct-4bit"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PAIRS = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Add 17 and 25.", "17 + 25 = 42."),
    ("Name a primary colour.", "Red is a primary colour."),
    ("Translate 'thank you' to Spanish.", "'Thank you' in Spanish is 'gracias'."),
    ("What does LoRA stand for?", "LoRA stands for Low-Rank Adaptation."),
    ("Give a synonym for 'quick'.", "A synonym for 'quick' is 'fast'."),
    ("How many days are in a leap year?", "A leap year has 366 days."),
    ("What is 9 squared?", "9 squared is 81."),
]


def _peak_gb():
    import mlx.core as mx
    return round(float(mx.get_peak_memory()) / 1024**3, 4)


def _fingerprint(params):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    return sum(float(mx.sum(mx.abs(v.astype(mx.float32))).item()) for _, v in tree_flatten(params))


def _logits(model, inputs):
    out = model(inputs)
    return getattr(out, "logits", out)


def _rows(tokenizer):
    rows = []
    for q, a in PAIRS:
        msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False)
        except Exception:
            text = f"### Question: {q}\n### Answer: {a}"
        rows.append({"text": text})
    return rows


def _kl_to_teacher(student, teacher, tokenizer, rows):
    """Token-mean forward KL(teacher || student) in float32 over the probe rows."""
    import mlx.core as mx
    import mlx.nn as nn
    total, n = 0.0, 0
    for r in rows:
        ids = mx.array([tokenizer.encode(r["text"])[:-1]], dtype=mx.int32)
        s = nn.log_softmax(_logits(student, ids).astype(mx.float32), axis=-1)
        t = nn.log_softmax(_logits(teacher, ids).astype(mx.float32), axis=-1)
        kl = (mx.exp(t) * (t - s)).sum(-1)
        total += float(kl.sum().item())
        n += int(kl.size)
    return total / max(n, 1)


def _reference_jsd(student_logits, teacher_logits, mask, beta):
    """float64 numpy generalized JSD (TRL GKDTrainer convention)."""
    import numpy as np
    s = student_logits.astype(np.float64)
    t = teacher_logits.astype(np.float64)
    ls = s - np.logaddexp.reduce(s, -1, keepdims=True)
    lt = t - np.logaddexp.reduce(t, -1, keepdims=True)
    m = np.logaddexp(ls + np.log(1 - beta), lt + np.log(beta))
    per = beta * (np.exp(lt) * (lt - m)) + (1 - beta) * (np.exp(ls) * (ls - m))
    per = per.sum(-1)
    return float((per * mask).sum() / max(mask.sum(), 1))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = C.base_parser("mlx_gkd", DEFAULT_MODEL, DEFAULT_MODEL, default_steps=10)
    p.add_argument("--teacher", default=TEACHER)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--rank", type=int, default=8)
    a = C.resolve_args(p, argv)
    if not any(x == "--lr" or x.startswith("--lr=") for x in argv):
        a.lr = 5e-4
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    with C.JobRecorder(a, backend_hint=None) as rec:
        rec.summary(teacher=a.teacher, beta=a.beta)
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            rec.skip(f"MLX needs Apple Silicon, this is {platform.system()}-{platform.machine()}")
        import unsloth  # noqa: F401  must precede mlx_lm / transformers
        is_mlx = bool(getattr(unsloth, "_IS_MLX", False))
        rec.set_backend("unsloth-mlx" if is_mlx else "unsloth-no-mlx")
        if not rec.check("mlx_active", is_mlx, f"unsloth._IS_MLX={is_mlx} on {C.detect_device()}"):
            return
        rec.backend_check()

        import dataclasses
        import mlx.core as mx
        import numpy as np
        from mlx_lm import load
        from unsloth import FastLanguageModel
        from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
        from unsloth_zoo.mlx.distill import build_gkd_loss_fn

        teacher, _ = load(a.teacher)
        teacher.freeze()
        teacher_fp = _fingerprint(teacher.parameters())
        has_report = "report_grad_norm" in {f.name for f in dataclasses.fields(MLXTrainingConfig)}

        def run_arm(name, gkd, compile_, with_eval=False):
            mx.random.seed(a.seed)
            model, tok = FastLanguageModel.from_pretrained(
                a.model, max_seq_length=a.max_seq_length, text_only=True, load_in_4bit=True,
                random_state=a.seed, token=os.environ.get("HF_TOKEN") or None)
            mx.random.seed(a.seed)
            model = FastLanguageModel.get_peft_model(
                model, r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, target_modules=TARGETS,
                use_gradient_checkpointing=False, random_state=a.seed)
            rows = _rows(tok)
            kl_before = _kl_to_teacher(model, teacher, tok, rows)
            before = _fingerprint(model.trainable_parameters())
            cfg = dict(max_steps=a.max_steps, learning_rate=a.lr, warmup_steps=0, logging_steps=1,
                       seed=a.seed, use_cce=False, compile=compile_, gradient_checkpointing=False,
                       output_dir=str(Path(a.out).with_suffix("")) + f"_{name}", save_steps=0,
                       eval_steps=(max(a.max_steps // 2, 1) if with_eval else 0),
                       per_device_train_batch_size=2, gradient_accumulation_steps=1,
                       lr_scheduler_type="linear", optim="adamw", weight_decay=0.0, max_grad_norm=1.0,
                       max_seq_length=a.max_seq_length, dataset_text_field="text")
            if has_report:
                cfg["report_grad_norm"] = True
            if gkd:
                cfg.update(teacher_model_name_or_path=a.teacher, gkd_beta=a.beta)
            trainer = MLXTrainer(model=model, tokenizer=tok, train_dataset=rows * 4,
                                 eval_dataset=(rows if with_eval else None),
                                 args=MLXTrainingConfig(**cfg))
            trainer.save_model = lambda *_a, **_k: None
            steps, last = [], {"t": 0.0}

            def on_step(step, total, loss, lr, tok_s, peak_gb, elapsed, num_tokens, grad_norm=None, *_, **__):
                dt = float(elapsed) - last["t"] if elapsed is not None else None
                last["t"] = float(elapsed) if elapsed is not None else last["t"]
                steps.append(dict(step=int(step), loss=float(loss),
                                  grad_norm=None if grad_norm is None else float(grad_norm),
                                  time_ms=round(dt * 1000, 2) if dt else None,
                                  tokens_per_s=float(tok_s) if tok_s is not None else None,
                                  peak_mem_gb=float(peak_gb) if peak_gb is not None else None))
                print(f"  [{name}] step {step}/{total} loss={loss:.4f} tok/s={tok_s:.0f} peak={peak_gb:.2f}GB",
                      flush=True)

            trainer.add_step_callback(on_step)
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            trainer.train()
            out = dict(name=name, steps=steps, train_s=round(time.perf_counter() - t0, 2),
                       peak_gb=_peak_gb(), kl_before=kl_before,
                       kl_after=_kl_to_teacher(model, teacher, tok, rows),
                       lora_changed=_fingerprint(model.trainable_parameters()) != before,
                       eval=dict(getattr(trainer, "_last_eval_metrics", None) or {}))
            timed = [s["time_ms"] for s in steps[2:] if s["time_ms"]]  # skip compile warmup
            out["median_step_ms"] = sorted(timed)[len(timed) // 2] if timed else None
            return out, model, tok

        ce, _, _ = run_arm("ce", gkd=False, compile_=True)
        eager, _, _ = run_arm("gkd_eager", gkd=True, compile_=False)
        comp, model, tok = run_arm("gkd_compiled", gkd=True, compile_=True, with_eval=True)
        for s in comp["steps"]:
            rec.step(**s)
        arms = {x["name"]: {k: v for k, v in x.items() if k != "steps"} | {"losses": [round(s["loss"], 5) for s in x["steps"]]}
                for x in (ce, eager, comp)}
        rec.summary(arms=arms)
        for x in (ce, eager, comp):
            print(f"  {x['name']:13s} median_step_ms={x['median_step_ms']} peak_gb={x['peak_gb']} "
                  f"KL(teacher||student) {x['kl_before']:.4f} -> {x['kl_after']:.4f}", flush=True)

        losses = [s["loss"] for s in comp["steps"]]
        rec.check("completed_steps", len(losses) == a.max_steps, f"{len(losses)} logged")
        rec.check("finite_loss", bool(losses) and all(math.isfinite(v) and 0 <= v < 50 for v in losses), losses)
        gns = [s["grad_norm"] for s in comp["steps"] if s["grad_norm"] is not None]
        if gns:
            rec.check("grad_norm_bounded", all(math.isfinite(g) and 0 < g < 1e3 for g in gns), gns)
        rec.check("adapter_changed", comp["lora_changed"], "LoRA fingerprint moved")
        rec.check("kl_to_teacher_falls", comp["kl_after"] < comp["kl_before"],
                  f"{comp['kl_before']:.5f} -> {comp['kl_after']:.5f}")
        rec.check("gkd_moves_kl_more_than_ce",
                  (comp["kl_before"] - comp["kl_after"]) > (ce["kl_before"] - ce["kl_after"]),
                  f"GKD {comp['kl_before'] - comp['kl_after']:.5f} vs CE {ce['kl_before'] - ce['kl_after']:.5f}")
        rec.check("teacher_frozen", _fingerprint(teacher.parameters()) == teacher_fp, "teacher weights unchanged")
        worst = max(abs(x - y) / max(abs(y), 1e-6) for x, y in zip(losses, [s["loss"] for s in eager["steps"]]))
        rec.check("compiled_matches_eager", worst < 2e-2, f"worst relative loss gap {worst:.2e}")
        ev = comp["eval"]
        rec.check("eval_loss_no_perplexity", "eval_loss" in ev and "eval_perplexity" not in ev, ev)

        # GKD loss on real Metal logits vs a float64 reference.
        from types import SimpleNamespace
        ids = tok.encode(_rows(tok)[0]["text"])
        batch = mx.array([ids], dtype=mx.int32)
        lengths = mx.array([[0, len(ids)]])
        fn = build_gkd_loss_fn(teacher, SimpleNamespace(gkd_beta=a.beta, max_seq_length=len(ids),
                                                         gkd_chunk_size=16), 1, 1, 0, 1 << 50)
        got, ntoks = fn(model, batch, lengths, None)
        s_logits = np.array(_logits(model, batch[:, :-1]).astype(mx.float32))
        t_logits = np.array(_logits(teacher, batch[:, :-1]).astype(mx.float32))
        ref = _reference_jsd(s_logits, t_logits, np.ones(s_logits.shape[:2]), a.beta)
        rel = abs(float(got) - ref) / max(abs(ref), 1e-9)
        rec.summary(fp64_reference=ref, gkd_loss=float(got), ntoks=int(ntoks))
        rec.check("loss_matches_fp64", rel < 1e-3, f"mlx {float(got):.6f} vs fp64 {ref:.6f} (rel {rel:.2e})")
        rec.summary(peak_mem_gb=max(x["peak_gb"] for x in (ce, eager, comp)))


if __name__ == "__main__":
    main()
