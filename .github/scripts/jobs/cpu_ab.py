"""Interleaved A/B of jobs/cpu_sft.py on one machine (same VM, alternating arms, fresh process each).

    python jobs/cpu_ab.py --model unsloth/Qwen3-0.6B                      # zoo main (A) vs installed zoo (B)
    python jobs/cpu_ab.py --a-args "--grad-ckpt unsloth" --b-args "--grad-ckpt false" --same-zoo

A = unsloth_zoo from --base-ref installed into a private --target dir (PYTHONPATH shadows the installed
one; each arm records the unsloth_zoo file it imported and a mismatch voids the run), B = the installed
zoo (the staged PR head). Reports per-arm medians of steady step time, peak RSS and final loss.
"""

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--model", default="unsloth/Qwen3-0.6B")
p.add_argument("--repeats", type=int, default=2)
p.add_argument("--max-steps", type=int, default=6)
p.add_argument("--base-ref", default="main")
p.add_argument("--same-zoo", action="store_true", help="both arms on the installed zoo (compare args only)")
p.add_argument("--a-args", default="")
p.add_argument("--b-args", default="")
p.add_argument("--tiny", action="store_true", help="accepted for the staging harness; ignored")
p.add_argument("--out", default="outputs/jobs/cpu_ab.json")
a = p.parse_args()

work = tempfile.mkdtemp(prefix="cpu_ab_")
result = {"job": "cpu_ab", "model": a.model, "base_ref": a.base_ref, "same_zoo": a.same_zoo,
          "a_args": a.a_args, "b_args": a.b_args, "runs": [], "checks": {}, "passed": False}


def flush():
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2, default=str)


base_path = None
if not a.same_zoo:
    base_path = os.path.join(work, "zoo_base")
    spec = f"unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo@{a.base_ref}"
    proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", base_path, spec],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        result["error"] = proc.stderr[-3000:]
        flush()
        print(proc.stderr[-3000:])
        sys.exit(1)

arms = {"A": (base_path, a.a_args), "B": (None, a.b_args)}
for i in range(a.repeats):
    order = ("A", "B") if i % 2 == 0 else ("B", "A")
    for arm in order:
        path, extra = arms[arm]
        out = os.path.join(work, f"{arm}_{i}.json")
        env = dict(os.environ)
        if path:
            env["PYTHONPATH"] = path + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [sys.executable, os.path.join(HERE, "cpu_sft.py"), "--model", a.model, "--max-steps", str(a.max_steps),
               "--skip-reload", "--out", out, *shlex.split(extra)]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        try:
            d = json.load(open(out))
        except Exception:  # noqa: BLE001
            d = {"passed": False, "summary": {}, "error": (proc.stdout[-1500:] + proc.stderr[-2500:])}
        s = d.get("summary", {})
        zoo_file = s.get("zoo_file") or ""
        run = {"arm": arm, "repeat": i, "passed": d.get("passed"), "steady_step_ms": s.get("steady_step_ms"),
               "first_step_ms": s.get("first_step_ms"), "train_s": s.get("train_s"), "peak_rss_mb": s.get("peak_rss_mb"),
               "final_loss": s.get("final_loss"), "optim": s.get("optim"), "loss_route": s.get("loss_route"),
               "bf16": s.get("bf16"), "zoo_file": zoo_file, "error": (d.get("error") or "")[-1500:] or None}
        result["runs"].append(run)
        print(json.dumps(run))
        flush()


def med(arm, key):
    xs = [r[key] for r in result["runs"] if r["arm"] == arm and r.get(key) is not None]
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs)} if xs else None


result["summary"] = {arm: {k: med(arm, k) for k in ("steady_step_ms", "train_s", "peak_rss_mb", "final_loss")}
                     for arm in ("A", "B")}
sa, sb = result["summary"]["A"]["steady_step_ms"], result["summary"]["B"]["steady_step_ms"]
if sa and sb:
    result["summary"]["B_over_A_steady"] = round(sb["median"] / sa["median"], 3)
ra, rb = result["summary"]["A"]["peak_rss_mb"], result["summary"]["B"]["peak_rss_mb"]
if ra and rb:
    result["summary"]["B_over_A_rss"] = round(rb["median"] / ra["median"], 3)
ok_runs = all(r["passed"] for r in result["runs"])
result["checks"]["all_runs_passed"] = {"ok": ok_runs, "detail": [r["error"] for r in result["runs"] if not r["passed"]][:2]}
if base_path:
    a_ok = all(r["zoo_file"].startswith(base_path) for r in result["runs"] if r["arm"] == "A")
    b_ok = all(r["zoo_file"] and not r["zoo_file"].startswith(base_path) for r in result["runs"] if r["arm"] == "B")
    result["checks"]["arms_import_their_zoo"] = {"ok": a_ok and b_ok,
                                                 "detail": sorted({r["arm"] + ":" + r["zoo_file"] for r in result["runs"]})}
result["passed"] = all(c["ok"] for c in result["checks"].values())
flush()
import shutil  # noqa: E402
shutil.rmtree(work, ignore_errors=True)
print("JOB_RESULT " + json.dumps({"job": "cpu_ab", "passed": result["passed"], "summary": result["summary"]}))
sys.exit(0 if result["passed"] else 1)
