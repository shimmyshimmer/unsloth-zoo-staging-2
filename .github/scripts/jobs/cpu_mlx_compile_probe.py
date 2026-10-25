"""Does torch.compile work after `import unsloth_zoo` on this host? Base (zoo --base-ref, installed into a
private --target dir) vs head (the installed zoo), each in a fresh process.

    python jobs/cpu_mlx_compile_probe.py --base-ref main

On Apple Silicon with MLX installed, unsloth_zoo stubs out triton; this records whether the stub was
installed and whether CPU (and MPS, when available) Inductor compiles still build and run.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

p = argparse.ArgumentParser()
p.add_argument("--base-ref", default="main")
p.add_argument("--tiny", action="store_true", help="accepted for the staging harness; ignored")
p.add_argument("--out", default="outputs/jobs/cpu_mlx_compile_probe.json")
a = p.parse_args()

CHILD = r"""
import json, sys
import unsloth_zoo
import torch
triton = sys.modules.get("triton")
r = {"zoo_file": unsloth_zoo.__file__, "device_type": getattr(unsloth_zoo, "DEVICE_TYPE", None),
     "triton_stubbed": bool(triton is not None and "triton_stub" in repr(getattr(triton, "__spec__", None)) + str(getattr(triton, "__file__", ""))),
     "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())}
def attempt(name, fn):
    try:
        fn(); r[name] = "ok"
    except Exception as e:
        r[name] = (type(e).__name__ + ": " + str(e))[:300]
def cpu_pointwise():
    f = torch.compile(lambda x: torch.nn.functional.silu(x) * 2)
    x = torch.linspace(-4, 4, 64)
    assert torch.allclose(f(x), torch.nn.functional.silu(x) * 2)
def cpu_fwd_bwd():
    lin = torch.nn.Linear(16, 16); x = torch.randn(4, 16, requires_grad=True)
    torch.compile(lin)(x).square().mean().backward(); assert x.grad is not None
attempt("cpu_pointwise", cpu_pointwise)
attempt("cpu_fwd_bwd", cpu_fwd_bwd)
if r["mps_available"]:
    def mps_pointwise():
        f = torch.compile(lambda x: torch.nn.functional.silu(x) * 2)
        x = torch.linspace(-4, 4, 64, device="mps")
        assert torch.allclose(f(x).cpu(), torch.nn.functional.silu(x.cpu()) * 2, atol=1e-5)
    attempt("mps_pointwise", mps_pointwise)
print("CHILD " + json.dumps(r))
"""

work = tempfile.mkdtemp(prefix="mlx_probe_")
base = os.path.join(work, "zoo_base")
spec = f"unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo@{a.base_ref}"
inst = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", base, spec],
                      capture_output=True, text=True)
result = {"job": "cpu_mlx_compile_probe", "base_ref": a.base_ref, "arms": {}}
for arm, path in (("base", base if inst.returncode == 0 else None), ("head", None)):
    if arm == "base" and path is None:
        result["arms"][arm] = {"error": inst.stderr[-1500:]}
        continue
    env = dict(os.environ)
    if path:
        env["PYTHONPATH"] = path + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, env=env, cwd=work)
    line = next((x for x in proc.stdout.splitlines() if x.startswith("CHILD ")), None)
    result["arms"][arm] = json.loads(line[6:]) if line else {"error": (proc.stdout[-1000:] + proc.stderr[-2000:])}
    print(arm, json.dumps(result["arms"][arm]), flush=True)

b, h = result["arms"].get("base", {}), result["arms"].get("head", {})
result["checks"] = {
    "arms_import_their_zoo": bool(b.get("zoo_file", "").startswith(base)) and bool(h.get("zoo_file")) and not h["zoo_file"].startswith(base),
    "head_cpu_compiles": h.get("cpu_pointwise") == "ok" and h.get("cpu_fwd_bwd") == "ok",
}
result["passed"] = all(result["checks"].values())
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(result, open(a.out, "w"), indent=2)
print("JOB_RESULT " + json.dumps({"job": "cpu_mlx_compile_probe", "passed": result["passed"], "checks": result["checks"]}))
sys.exit(0 if result["passed"] else 1)
