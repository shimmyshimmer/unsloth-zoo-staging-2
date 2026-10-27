"""MLX GRPO with torch reward models on Apple Silicon (unsloth-zoo#2107).

A 3-step MLXGRPOTrainer run whose reward_funcs mix a torch sequence-classification model
(object, tokenizer given), the same model by local path (tokenizer loaded from it) and a plain
callable. Every reward-model call is recorded and re-scored with TRL 0.24's lines on an
independent CPU copy of the weights. Run per `--rm-device` (cpu, mps).
"""

from __future__ import annotations

import copy
import math
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

DEFAULT_MODEL = TINY_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
RM_REPO = "trl-internal-testing/tiny-Qwen3ForSequenceClassification"
PROMPTS = ["What is 2+2?", "Name a primary colour.", "Say hello in French.", "What is the capital of Italy?"]


def _trl_scores(model, tokenizer, prompts, completions):
    """TRL GRPOTrainer._calculate_rewards for an nn.Module reward func (trl/data_utils
    apply_chat_template on {"messages": p + c} = tokenizer.apply_chat_template(m, tools=None, tokenize=False))."""
    import torch
    if isinstance(completions[0], list):
        texts = [tokenizer.apply_chat_template(p + c, tools=None, tokenize=False) for p, c in zip(prompts, completions)]
    else:
        texts = [p + c for p, c in zip(prompts, completions)]
    inputs = tokenizer(text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
    with torch.inference_mode():
        return model(**inputs).logits[:, 0].float().tolist()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = C.base_parser("mlx_grpo_rm", DEFAULT_MODEL, TINY_MODEL, default_steps=3)
    p.add_argument("--rm-device", default="cpu,mps", help="comma list of torch devices for the reward model")
    a = C.resolve_args(p, argv)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    with C.JobRecorder(a, backend_hint=None) as rec:
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            rec.skip(f"MLX needs Apple Silicon, this is {platform.system()}-{platform.machine()}")
        import unsloth  # noqa: F401  must precede mlx_lm / transformers
        is_mlx = bool(getattr(unsloth, "_IS_MLX", False))
        rec.set_backend("unsloth-mlx" if is_mlx else "unsloth-no-mlx")
        if not rec.check("mlx_active", is_mlx, f"unsloth._IS_MLX={is_mlx}"):
            return
        rec.backend_check()

        import mlx.core as mx
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from unsloth import FastLanguageModel
        import unsloth_zoo.mlx.grpo as grpo
        from unsloth_zoo.mlx.trainer import MLXGRPOConfig, MLXGRPOTrainer
        rec.summary(zoo_grpo_file=grpo.__file__, torch=torch.__version__, mps=torch.backends.mps.is_available())

        out_dir = Path(Path(a.out).with_suffix("")).resolve()
        # Same weights under two names: rm_obj is passed as a model object, rm_path by its path.
        rm_dir, path_dir = out_dir / "rm_obj", out_dir / "rm_path"
        torch.manual_seed(0)
        rm = AutoModelForSequenceClassification.from_pretrained(RM_REPO, num_labels=1, ignore_mismatched_sizes=True)
        torch.nn.init.normal_(rm.score.weight, std=1.0)
        for d in (rm_dir, path_dir):
            rm.save_pretrained(d)
            AutoTokenizer.from_pretrained(RM_REPO).save_pretrained(d)

        calls = []
        real_call = grpo._RewardModel.__call__

        def recording_call(self, prompts, completions, **kw):
            out = real_call(self, prompts, completions, **kw)
            calls.append((self.__name__, str(next(self.model.parameters()).device),
                          copy.deepcopy(prompts), copy.deepcopy(completions), list(out)))
            return out

        grpo._RewardModel.__call__ = recording_call
        oracle_model = AutoModelForSequenceClassification.from_pretrained(rm_dir).eval()
        oracle_tok = AutoTokenizer.from_pretrained(rm_dir)

        for device in [d for d in a.rm_device.split(",") if d]:
            if device == "mps" and not torch.backends.mps.is_available():
                rec.check(f"{device}_available", False, "torch.backends.mps unavailable")
                continue
            calls.clear()
            mx.random.seed(a.seed)
            model, tokenizer = FastLanguageModel.from_pretrained(
                a.model, max_seq_length=256, load_in_4bit=True, text_only=True, random_state=a.seed)
            model = FastLanguageModel.get_peft_model(
                model, r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
                use_gradient_checkpointing=False, random_state=a.seed)
            judge = AutoModelForSequenceClassification.from_pretrained(rm_dir).eval().to(device)
            data = [{"prompt": [{"role": "user", "content": q}]} for q in PROMPTS] * 4
            config = MLXGRPOConfig(
                max_steps=a.max_steps, per_device_train_batch_size=4, gradient_accumulation_steps=1,
                num_generations=4, max_completion_length=16, learning_rate=a.lr, logging_steps=1,
                output_dir=str(out_dir / f"trainer_{device}"), save_steps=0, eval_steps=0, seed=a.seed,
                compile=False, gradient_checkpointing=False, max_seq_length=256)
            trainer = MLXGRPOTrainer(
                model=model, tokenizer=tokenizer, train_dataset=data, args=config,
                reward_funcs=[judge, str(path_dir), lambda completions, **_: [len(c[0]["content"]) / 100 for c in completions]],
                reward_processing_classes=[AutoTokenizer.from_pretrained(rm_dir), None, None])
            t0 = time.perf_counter()
            trainer.train()
            rec.summary(**{f"{device}_train_s": round(time.perf_counter() - t0, 2)})
            logs = [r for r in trainer.state.log_history if "loss" in r]
            for r in logs:
                rec.step(step=r.get("step"), loss=r.get("loss"), grad_norm=r.get("grad_norm"), time_ms=None,
                         tokens=None, tokens_per_s=None, peak_mem_gb=None, rm_device=device,
                         **{k: v for k, v in r.items() if k.startswith(("reward", "rewards/"))})
            rec.check(f"{device}_steps", len(logs) == a.max_steps, f"{len(logs)} logged steps")
            rec.check(f"{device}_loss_finite", logs and all(math.isfinite(r["loss"]) for r in logs),
                      [r.get("loss") for r in logs])
            names = sorted({k.split("/")[1] for r in logs for k in r if k.startswith("rewards/")})
            rec.check(f"{device}_metric_names", {"rm_obj", "rm_path"} <= set(names), names)
            where = sorted({(c[0], c[1]) for c in calls})
            rec.check(f"{device}_rm_ran_where_given", len(calls) >= 2 * a.max_steps
                      and all(dev.startswith(device if name == "rm_obj" else "cpu") for name, dev in where),
                      f"{len(calls)} calls, {where}")
            worst, n = 0.0, 0
            for _, dev, prompts, completions, ours in calls:
                ref = _trl_scores(oracle_model, oracle_tok, prompts, completions)
                n += len(ref)
                worst = max(worst, max(abs(x - y) for x, y in zip(ours, ref)))
            tol = 0.0 if device == "cpu" else 1e-3
            rec.check(f"{device}_matches_trl", n > 0 and worst <= tol, f"{n} rewards, max |ours - trl| = {worst:.3g} (tol {tol})")
            rec.summary(**{f"{device}_reward_samples": [c[4][:4] for c in calls[:2]]})
            del trainer, model, judge


if __name__ == "__main__":
    main()
