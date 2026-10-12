"""MLX speculative decoding A/B on Apple Silicon: merge-base zoo tree vs the checked-out (head) tree.

Each arm runs in a FRESH interpreter that imports unsloth_zoo from its own tree (editable finders
dropped, `unsloth_zoo.__file__` asserted), loads a Qwen3.5 target, builds the native MTP drafter from
the bf16 checkpoint's `mtp.*` tensors, and decodes greedy replies through `SpeculativeEngine` inside
`generation_mode` (the production scope stack). Arms run ABBA per target.

Checks: every arm emits the same tokens as base (greedy = exact), head's new paths ran (served
verify calls / shortlist picks counted), finite tok/s. Speed is reported, not gated (shared runner).

    python jobs/mlx_spec_ab.py --base-ref <merge-base sha> [--targets a,b] [--tokens 128] [--rounds 2]

Exit 3 off Apple Silicon.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROMPTS = [
    "Write a short Python function that returns the n-th Fibonacci number, then explain it.",
    "List five facts about the planet Mars, one per line.",
    "Translate to French: The quick brown fox jumps over the lazy dog. Then repeat it twice.",
    "Explain what a hash map is to a new programmer in a few sentences.",
]
MTP_SOURCE = "Qwen/Qwen3.5-0.8B"
DEFAULT_TARGETS = "mlx-community/Qwen3.5-0.8B-8bit,mlx-community/Qwen3.5-0.8B-4bit"


def child(args):
    tree = str(Path(args.tree).resolve())
    sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in type(f).__module__.lower()
                        and "editable" not in type(f).__name__.lower()]
    sys.path.insert(0, tree)
    os.environ["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    import mlx.core as mx
    import unsloth_zoo
    assert str(Path(unsloth_zoo.__file__).resolve()).startswith(tree), (unsloth_zoo.__file__, tree)
    from mlx_vlm import load
    from mlx_vlm.models.cache import make_prompt_cache
    from mlx_vlm.utils import get_model_path
    from unsloth_zoo.mlx import inference as I
    from unsloth_zoo.mlx import speculative as S
    from unsloth_zoo.mlx.generate import SamplingParams, generation_mode

    counts = {"verify_sliced_served": None, "verify_admitted": None, "shortlist_calls": None}
    if hasattr(I, "_verify_sliced"):
        counts.update(verify_sliced_served = 0, verify_admitted = 0)
        sliced, admitted = I._verify_sliced, I._verify_admitted

        def count_sliced(*a, **k):
            if getattr(I._VERIFY_STATE, "serving", False):
                counts["verify_sliced_served"] += 1
            return sliced(*a, **k)

        def count_admitted(*a, **k):
            counts["verify_admitted"] += 1
            return admitted(*a, **k)
        I._verify_sliced, I._verify_admitted = count_sliced, count_admitted
    if hasattr(S, "_ShortlistHead"):
        counts["shortlist_calls"] = 0
        call = S._ShortlistHead.__call__

        def count_call(self, hidden):
            counts["shortlist_calls"] += 1
            return call(self, hidden)
        S._ShortlistHead.__call__ = count_call

    model, processor = load(args.target)
    tok = processor.tokenizer
    ids = [tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt = True,
                                              tokenize = False), add_special_tokens = False) for p in PROMPTS]
    drafter = S.native_mtp_drafter(get_model_path(MTP_SOURCE), model)
    greedy = SamplingParams()
    lm = model.language_model

    def prefill(prompt):
        cache = make_prompt_cache(lm)
        out = lm(mx.array([prompt]), cache = cache, **S._capture(drafter))
        logits = out.logits[:, -1]
        first = int(S._RowSampler(greedy)(logits - mx.logsumexp(logits, -1, keepdims = True), 0).item())
        return cache, first, S._features(drafter, out)

    def run(prompt):
        engine = S.SpeculativeEngine(model, S.DraftController(max_depth = 4, max_copy = 16), drafter)
        cache, first, hidden = prefill(prompt)
        engine.add(S.EngineRow(cache = cache, pending = first, prompt = prompt, sampling = greedy,
                               max_tokens = args.tokens, hidden = hidden, uid = 0))
        out, drafted = [first], (0, 0)
        start = time.perf_counter()
        while engine.rows:
            for step in engine.step():
                out.extend(step.tokens)
                drafted = (step.draft_n, step.draft_n_accepted)
            if engine.draft_cache:
                engine.draft_cache.clear()
        mx.synchronize()
        return out, time.perf_counter() - start, drafted

    replies = []
    with generation_mode(model):
        run(ids[0][:])  # warm-up: kernels, verdicts, shortlist build
        for k in list(counts):
            if counts[k] is not None:
                counts[k] = 0
        for prompt in ids:
            out, seconds, drafted = run(prompt)
            replies.append({"tokens": out, "seconds": seconds, "tok_s": (len(out) - 1) / seconds,
                            "draft_n": drafted[0], "draft_accepted": drafted[1]})
    print("CHILD_RESULT " + json.dumps({"file": unsloth_zoo.__file__, "counts": counts, "replies": replies,
                                         "shortlist": getattr(drafter, "_shortlist", None) is not None}))


def run_arm(tree, target, tokens):
    cmd = [sys.executable, "-u", __file__, "--child", "--tree", str(tree), "--target", target, "--tokens", str(tokens)]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    proc = subprocess.run(cmd, capture_output = True, text = True, env = env, timeout = 3600)
    line = next((ln for ln in proc.stdout.splitlines()[::-1] if ln.startswith("CHILD_RESULT ")), None)
    if proc.returncode or line is None:
        return {"error": f"rc={proc.returncode}: {(proc.stderr or proc.stdout)[-3000:]}"}
    return json.loads(line[len("CHILD_RESULT "):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action = "store_true")
    ap.add_argument("--tree")
    ap.add_argument("--target")
    ap.add_argument("--base-ref")
    ap.add_argument("--base-repo", default = "https://github.com/unslothai/unsloth-zoo")
    ap.add_argument("--head-tree", default = ".")
    ap.add_argument("--targets", default = DEFAULT_TARGETS)
    ap.add_argument("--tokens", type = int, default = 128)
    ap.add_argument("--rounds", type = int, default = 2)
    ap.add_argument("--out", default = "outputs/jobs/mlx_spec_ab.json")
    args = ap.parse_args()
    if args.child:
        return child(args)
    result = {"job": "mlx_spec_ab", "config": vars(args), "checks": {}, "arms": {}, "summary": {},
              "passed": False, "skipped": False, "error": None}
    out = Path(args.out)
    out.parent.mkdir(parents = True, exist_ok = True)

    def finish(code):
        out.write_text(json.dumps(result, indent = 1))
        print("JOB_RESULT " + json.dumps({k: result[k] for k in ("passed", "skipped", "error", "summary")}))
        return code

    if sys.platform != "darwin" or platform.machine() != "arm64":
        result.update(skipped = True, error = "not Apple Silicon")
        return finish(3)
    import mlx.core as mx
    import mlx_vlm
    result["versions"] = {"mlx": mx.__version__, "mlx_vlm": mlx_vlm.__version__, "python": sys.version.split()[0]}
    head = Path(args.head_tree).resolve()
    base = Path("mlx_spec_ab_base").resolve()
    if not base.exists():
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", args.base_repo, str(base)], check = True)
    subprocess.run(["git", "-C", str(base), "checkout", "-q", args.base_ref], check = True)
    result["revisions"] = {
        "base": subprocess.run(["git", "-C", str(base), "rev-parse", "HEAD"], capture_output = True, text = True).stdout.strip(),
        "head": subprocess.run(["git", "-C", str(head), "rev-parse", "HEAD"], capture_output = True, text = True).stdout.strip()}
    from huggingface_hub import snapshot_download
    for repo in [MTP_SOURCE, *args.targets.split(",")]:
        snapshot_download(repo)
    ok = True
    for target in args.targets.split(","):
        arms = {"base": [], "head": []}
        order = []
        for _ in range(args.rounds):
            order += ["base", "head", "head", "base"]
        for arm in order:
            r = run_arm(base if arm == "base" else head, target, args.tokens)
            arms[arm].append(r)
            print(f"[{target}] {arm}: " + (r.get("error") or json.dumps(
                {"tok_s": [round(x["tok_s"], 1) for x in r["replies"]], "counts": r["counts"],
                 "accept": [f'{x["draft_accepted"]}/{x["draft_n"]}' for x in r["replies"]]})), flush = True)
        result["arms"][target] = arms
        errors = [r["error"] for rs in arms.values() for r in rs if "error" in r]
        if errors:
            result["checks"][f"{target}:ran"] = {"ok": False, "detail": errors[0][-1500:]}
            ok = False
            continue
        ref = [x["tokens"] for x in arms["base"][0]["replies"]]
        same = all([x["tokens"] for x in r["replies"]] == ref for rs in arms.values() for r in rs)
        files = {arm: {r["file"] for r in rs} for arm, rs in arms.items()}
        tree_ok = all(f.startswith(str(base)) for f in files["base"]) and all(f.startswith(str(head)) for f in files["head"])
        engaged = all((r["counts"]["shortlist_calls"] or 0) > 0 and r["shortlist"] for r in arms["head"])
        sliced = sum(r["counts"]["verify_sliced_served"] or 0 for r in arms["head"])
        result["checks"][f"{target}:tokens_identical"] = {"ok": same, "detail": f"{len(ref)} replies"}
        result["checks"][f"{target}:import_paths"] = {"ok": tree_ok, "detail": {k: sorted(v) for k, v in files.items()}}
        result["checks"][f"{target}:head_paths_engaged"] = {"ok": engaged, "detail": f"shortlist on every head arm; served sliced-verify calls {sliced}"}
        ok &= same and tree_ok and engaged
        per = {arm: [statistics.median(x["tok_s"] for x in r["replies"]) for r in rs] for arm, rs in arms.items()}
        result["summary"][target] = {
            "base_tok_s_median": statistics.median(per["base"]), "head_tok_s_median": statistics.median(per["head"]),
            "base_spread": [min(per["base"]), max(per["base"])], "head_spread": [min(per["head"]), max(per["head"])],
            "speedup": statistics.median(per["head"]) / statistics.median(per["base"])}
        print(f"[{target}] summary {json.dumps(result['summary'][target])}", flush = True)
    result["passed"] = ok
    return finish(0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
