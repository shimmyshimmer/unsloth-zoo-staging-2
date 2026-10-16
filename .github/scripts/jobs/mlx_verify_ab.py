"""Base vs head A/B of one exact speculative verify round on Apple Silicon (unsloth-zoo PR stacks).

Run from a checkout of the head tree with it installed. Builds a base copy by reverse-applying the
PR's diff for the package files, then runs fresh processes in ABBA order, each timing verify +
commit + evaluating the committed caches, and hashing every committed state so the arms can be
compared bit for bit.

    python mlx_verify_ab.py --pr 1813 --out job.json
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

MODEL = "mlx-community/Qwen3.5-0.8B-4bit"


def child(args):
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
    import copy
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map
    from mlx_vlm import load
    from mlx_vlm.models.cache import make_prompt_cache
    import unsloth_zoo
    from unsloth_zoo.mlx import inference

    model, processor = load(MODEL)
    lm = model.language_model
    tok = processor.tokenizer
    text = "Copy this code exactly, then add a docstring to it:\ndef merge_sort(values):\n    return values\n" * 6
    ids = tok.encode(text, add_special_tokens = False)
    rows = args.rows
    cache = make_prompt_cache(lm)
    mx.eval(lm(mx.array([ids[:48]] * rows), cache = cache).logits)

    def clone():
        copies = [copy.copy(entry) for entry in cache]
        for entry in copies:
            entry.__dict__.update(tree_map(lambda leaf: mx.array(leaf) if isinstance(leaf, mx.array) else leaf, vars(entry)))
        mx.eval([leaf for entry in copies for _, leaf in tree_flatten(entry.state) if isinstance(leaf, mx.array)])
        return copies

    result = {"zoo_file": unsloth_zoo.__file__, "cases": {}}
    with inference.exact_speculative_verify(model):
        for width in args.widths:
            block = mx.array([ids[48 + r : 48 + r + width] for r in range(rows)])
            for mode in ("full", "partial"):
                keep = [width] * rows if mode == "full" else [1 + (2 * r) % width for r in range(rows)]
                times, digest, taped = [], None, None
                for i in range(args.warmup + args.reps):
                    run = clone()
                    t0 = time.perf_counter()
                    out = lm(block, cache = run, speculative_verify = True,
                             capture_layer_ids = [0, len(lm.layers) // 2], return_hidden = True,
                             rope_deltas = mx.array([[3]]))
                    taped = len(getattr(out.gdn_states, "_unsloth_tapes", ()) or ())
                    out.gdn_states.commit(keep)
                    leaves = [out.logits] + [leaf for entry in run for _, leaf in tree_flatten(entry.state)
                                             if isinstance(leaf, mx.array)]
                    mx.eval(leaves)
                    dt = time.perf_counter() - t0
                    if i >= args.warmup:
                        times.append(dt * 1e3)
                    if digest is None:
                        h = hashlib.sha256()
                        for leaf in leaves:
                            h.update(str((leaf.dtype, leaf.shape)).encode())
                            h.update(bytes(memoryview(leaf.view(mx.uint8))))
                        digest = h.hexdigest()
                result["cases"][f"W{width}_{mode}"] = {"median_ms": statistics.median(times), "times_ms": times,
                                                     "digest": digest, "taped_layers": taped}
    print("CHILD_RESULT " + json.dumps(result), flush = True)


def build_base(pr, repo):
    head = os.getcwd()
    base = os.path.join(tempfile.mkdtemp(prefix = "zoo_base_"), "tree")
    shutil.copytree(head, base, ignore = shutil.ignore_patterns(".git", "__pycache__", "*.egg-info"))
    url = f"https://patch-diff.githubusercontent.com/raw/{repo}/pull/{pr}.diff"
    diff = urllib.request.urlopen(url, timeout = 60).read()
    path = os.path.join(base, "pr.diff")
    with open(path, "wb") as f:
        f.write(diff)
    subprocess.run(["git", "apply", "-R", "--include=unsloth_zoo/*", "pr.diff"], cwd = base, check = True)
    return base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pr", type = int, default = 1813)
    p.add_argument("--repo", default = "unslothai/unsloth-zoo")
    p.add_argument("--rows", type = int, default = 4)
    p.add_argument("--widths", type = int, nargs = "+", default = [4, 8])
    p.add_argument("--reps", type = int, default = 15)
    p.add_argument("--warmup", type = int, default = 3)
    p.add_argument("--rounds", type = int, default = 3)
    p.add_argument("--out", default = "mlx_verify_ab.json")
    p.add_argument("--child", action = "store_true")
    args = p.parse_args()
    if args.child:
        return child(args)

    report = {"job": "mlx_verify_ab", "model": MODEL, "device": platform.platform(), "config": vars(args),
              "arms": {"base": [], "head": []}, "checks": {}, "passed": False, "error": None}
    try:
        trees = {"head": os.getcwd(), "base": build_base(args.pr, args.repo)}
        child_args = [sys.executable, os.path.abspath(__file__), "--child", "--rows", str(args.rows),
                      "--reps", str(args.reps), "--warmup", str(args.warmup), "--widths", *map(str, args.widths)]
        order = []
        for r in range(args.rounds):
            order += ["base", "head", "head", "base"] if r % 2 == 0 else ["head", "base", "base", "head"]
        for arm in order:
            env = {**os.environ, "PYTHONPATH": trees[arm]}
            proc = subprocess.run(child_args, env = env, capture_output = True, text = True, cwd = tempfile.gettempdir())
            line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("CHILD_RESULT ")), None)
            if proc.returncode or line is None:
                raise RuntimeError(f"{arm} child failed rc={proc.returncode}: {proc.stderr[-3000:]}")
            res = json.loads(line[len("CHILD_RESULT "):])
            report["arms"][arm].append(res)
            print(arm, {k: round(v["median_ms"], 2) for k, v in res["cases"].items()}, res["zoo_file"], flush = True)
        base, head = report["arms"]["base"], report["arms"]["head"]
        report["checks"]["arms_import_own_tree"] = {
            "ok": all(r["zoo_file"].startswith(trees["base"]) for r in base)
                  and all(r["zoo_file"].startswith(trees["head"]) for r in head),
            "detail": [base[0]["zoo_file"], head[0]["zoo_file"]]}
        summary = {}
        for case in base[0]["cases"]:
            b = [r["cases"][case]["median_ms"] for r in base]
            h = [r["cases"][case]["median_ms"] for r in head]
            digests = {r["cases"][case]["digest"] for r in base + head}
            summary[case] = {"base_ms": statistics.median(b), "head_ms": statistics.median(h),
                             "base_spread_ms": [min(b), max(b)], "head_spread_ms": [min(h), max(h)],
                             "speedup": statistics.median(b) / statistics.median(h),
                             "bit_identical": len(digests) == 1,
                             "head_taped_layers": head[0]["cases"][case]["taped_layers"],
                             "base_taped_layers": base[0]["cases"][case]["taped_layers"]}
        report["summary"] = summary
        report["checks"]["bit_identical"] = {"ok": all(s["bit_identical"] for s in summary.values()), "detail": None}
        report["checks"]["head_taped"] = {"ok": all(s["head_taped_layers"] > 0 for s in summary.values()), "detail": None}
        report["checks"]["base_not_taped"] = {"ok": all(s["base_taped_layers"] == 0 for s in summary.values()), "detail": None}
        report["passed"] = all(c["ok"] for c in report["checks"].values())
        for case, s in summary.items():
            print(f"{case}: base {s['base_ms']:.2f} {s['base_spread_ms']} head {s['head_ms']:.2f} {s['head_spread_ms']} "
                  f"x{s['speedup']:.3f} identical={s['bit_identical']} taped={s['head_taped_layers']}", flush = True)
    except Exception as e:
        report["error"] = repr(e)
        print("ERROR", e, flush = True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent = 1)
    print("JOB_RESULT " + json.dumps({"passed": report["passed"], "error": report["error"],
                                      "summary": report.get("summary")}), flush = True)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
