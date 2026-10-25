"""CPU inference A/B through Unsloth: zoo --base-ref (private --target install, PYTHONPATH) vs the installed zoo,
alternating fresh processes. Per arm: prefill time (1 new token), decode tok/s, greedy output ids.

    python jobs/cpu_infer_ab.py --model unsloth/Qwen3.5-0.8B --base-ref main
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile

p = argparse.ArgumentParser()
p.add_argument("--model", default="unsloth/Qwen3.5-0.8B")
p.add_argument("--base-ref", default="main")
p.add_argument("--processes", type=int, default=2, help="fresh processes per arm, alternating")
p.add_argument("--repeats", type=int, default=3, help="timed generations per process")
p.add_argument("--new-tokens", type=int, default=48)
p.add_argument("--tiny", action="store_true", help="accepted for the staging harness; ignored")
p.add_argument("--out", default="outputs/jobs/cpu_infer_ab.json")
a = p.parse_args()

CHILD = r"""
import json, os, sys, time
os.environ.setdefault("UNSLOTH_FORCE_CPU", "1")
import unsloth
from unsloth import FastLanguageModel
import unsloth_zoo, torch
model, tok = FastLanguageModel.from_pretrained(sys.argv[1], max_seq_length=512, load_in_4bit=False)
FastLanguageModel.for_inference(model)
prompt = "Explain in two sentences why the sky looks blue during the day and red at sunset."
ids = tok(text=prompt, return_tensors="pt")
n = int(sys.argv[2]); reps = int(sys.argv[3])
def gen(k):
    t = time.perf_counter()
    out = model.generate(**ids, max_new_tokens=k, min_new_tokens=k, do_sample=False)
    return time.perf_counter() - t, out[0, ids["input_ids"].shape[1]:].tolist()
gen(4)
pre, full, out_ids = [], [], None
for _ in range(reps):
    pre.append(gen(1)[0]); t, out_ids = gen(n); full.append(t)
print("CHILD " + json.dumps({"zoo": unsloth_zoo.__file__, "prefill_s": pre, "gen_s": full, "ids": out_ids,
                             "dtype": str(next(model.parameters()).dtype), "threads": torch.get_num_threads()}))
"""

work = tempfile.mkdtemp(prefix="cpu_infer_ab_")
base = os.path.join(work, "zoo_base")
inst = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", base,
                       f"unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo@{a.base_ref}"], capture_output=True, text=True)
result = {"job": "cpu_infer_ab", "model": a.model, "base_ref": a.base_ref, "runs": [], "passed": False}
if inst.returncode != 0:
    result["error"] = inst.stderr[-2000:]
else:
    for i in range(a.processes):
        for arm in (("A", "B") if i % 2 == 0 else ("B", "A")):
            env = dict(os.environ)
            if arm == "A":
                env["PYTHONPATH"] = base + os.pathsep + env.get("PYTHONPATH", "")
            proc = subprocess.run([sys.executable, "-c", CHILD, a.model, str(a.new_tokens), str(a.repeats)],
                                  capture_output=True, text=True, env=env, cwd=work)
            line = next((x for x in proc.stdout.splitlines() if x.startswith("CHILD ")), None)
            run = json.loads(line[6:]) if line else {"error": proc.stdout[-1500:] + proc.stderr[-2500:]}
            run["arm"] = arm
            result["runs"].append(run)
            print(arm, json.dumps({k: v for k, v in run.items() if k != "ids"})[:600], flush=True)


def arm_stats(arm):
    rs = [r for r in result["runs"] if r["arm"] == arm and "gen_s" in r]
    if not rs:
        return None
    pre = [x for r in rs for x in r["prefill_s"]]
    gen = [x for r in rs for x in r["gen_s"]]
    span = statistics.median(gen) - statistics.median(pre)
    return {"prefill_ms_median": round(statistics.median(pre) * 1e3, 1), "prefill_ms_range": [round(min(pre) * 1e3, 1), round(max(pre) * 1e3, 1)],
            "gen_s_median": round(statistics.median(gen), 3), "gen_s_range": [round(min(gen), 3), round(max(gen), 3)],
            "decode_tok_s_median": round((a.new_tokens - 1) / span, 2) if span > 0 else None,
            "zoo": sorted({r["zoo"] for r in rs})}


result["A"], result["B"] = arm_stats("A"), arm_stats("B")
ids = {arm: [r["ids"] for r in result["runs"] if r["arm"] == arm and "ids" in r] for arm in ("A", "B")}
result["same_greedy_output"] = bool(ids["A"] and ids["B"]) and all(x == ids["A"][0] for x in ids["A"] + ids["B"])
if result["A"] and result["B"]:
    result["B_over_A_prefill"] = round(result["B"]["prefill_ms_median"] / result["A"]["prefill_ms_median"], 3)
    if result["A"]["decode_tok_s_median"] and result["B"]["decode_tok_s_median"]:
        result["B_over_A_decode_tok_s"] = round(result["B"]["decode_tok_s_median"] / result["A"]["decode_tok_s_median"], 3)
    result["passed"] = all(z.startswith(base) for z in result["A"]["zoo"]) and not any(z.startswith(base) for z in result["B"]["zoo"])
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(result, open(a.out, "w"), indent=2)
print("JOB_RESULT " + json.dumps({k: result.get(k) for k in ("passed", "A", "B", "B_over_A_prefill", "B_over_A_decode_tok_s", "same_greedy_output")}))
sys.exit(0 if result["passed"] else 1)
