"""Round 19: large CPU buffer churn per training step, per allocator setting, on every OS.

A child process runs "steps" that allocate, first-touch and free the large buffers of a Qwen3-0.6B CPU full
finetuning step (an fp32 [V, H] head gradient cast to bf16, two fp32 [80, V] logit chunks, 20 fp32 [3072, 1024]
gradients). Per step: wall time and page faults. Arms are environment settings applied at process start, so
Windows' torch allocator (mimalloc inside c10.dll when torch was built with it) reads them:

    d      defaults
    pd     MIMALLOC_PURGE_DELAY=-1 (freed memory is never decommitted)
    pd10   MIMALLOC_PURGE_DELAY=10000 (decommit after 10 s)
    nodc   MIMALLOC_PURGE_DECOMMITS=0 (reset instead of decommit)
    late   MIMALLOC_PURGE_DELAY=-1 set from Python after `import torch` (does a runtime set work?)
    api    mi_option_set(purge_delay, -1) through ctypes when a torch DLL exports it
    mo     Linux glibc mallopt(M_MMAP_MAX 0, M_TRIM_THRESHOLD -1), as zoo #2980
    reuse  the same buffers allocated once and rewritten in place each step (the zero-churn floor)

    python jobs/cpu_alloc_micro.py --out alloc.json
"""
import argparse, glob, json, os, platform, statistics, subprocess, sys, time

V, H, R = 151936, 1024, 80
ARMS = {
    "d": {}, "pd": {"MIMALLOC_PURGE_DELAY": "-1"}, "pd10": {"MIMALLOC_PURGE_DELAY": "10000"},
    "nodc": {"MIMALLOC_PURGE_DECOMMITS": "0"}, "late": {"CPU_ALLOC_LATE": "1"}, "api": {"CPU_ALLOC_API": "1"},
    "mo": {"CPU_ALLOC_MALLOPT": "1"}, "reuse": {"CPU_ALLOC_REUSE": "1"},
}
TAG = "CPU_ALLOC_RESULT "


def faults():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t), ("e", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PageFaultCount, pmc.PeakWorkingSetSize >> 20, pmc.PagefileUsage >> 20
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_minflt + r.ru_majflt, r.ru_maxrss >> (20 if sys.platform == "darwin" else 10), None


def mi_exports():
    """torch DLLs exporting mimalloc's API (Windows), with mi_version when callable."""
    out = {}
    if sys.platform != "win32":
        return out
    import ctypes
    import torch
    for dll in sorted(glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "*.dll"))):
        try:
            lib = ctypes.WinDLL(dll)
        except OSError:
            continue
        names = [n for n in ("mi_option_set", "mi_option_get", "mi_version", "mi_collect", "mi_malloc") if hasattr(lib, n)]
        if names:
            e = {"exports": names}
            if "mi_version" in names:
                e["mi_version"] = lib.mi_version()
            if "mi_option_get" in names:
                e["options_0_30"] = [lib.mi_option_get(i) for i in range(31)]
            out[os.path.basename(dll)] = e
    return out


def child(steps):
    late, api = os.environ.get("CPU_ALLOC_LATE"), os.environ.get("CPU_ALLOC_API")
    rec = {"env": {k: v for k, v in os.environ.items() if k.startswith(("MIMALLOC_", "CPU_ALLOC_"))}}
    if os.environ.get("CPU_ALLOC_MALLOPT") and sys.platform.startswith("linux"):
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        rec["mallopt"] = [libc.mallopt(-1, -1), libc.mallopt(-4, 0)]
    import torch
    if late:
        os.environ["MIMALLOC_PURGE_DELAY"] = "-1"
    if api:
        rec["api"] = "no export"
        for name, e in mi_exports().items():
            import ctypes
            lib = ctypes.WinDLL(os.path.join(os.path.dirname(torch.__file__), "lib", name))
            if "mi_option_set" in e["exports"]:
                before = lib.mi_option_get(15)
                lib.mi_option_set(15, -1)
                rec["api"] = {"dll": name, "opt15_before": before, "opt15_after": lib.mi_option_get(15)}
                break
    rec["threads"] = torch.get_num_threads()
    reuse = os.environ.get("CPU_ALLOC_REUSE")
    bufs = None
    if reuse:
        bufs = ([torch.empty(V, H), torch.empty(V, H, dtype=torch.bfloat16)] + [torch.empty(R, V) for _ in range(2)]
                + [torch.empty(3072, H) for _ in range(20)])
    src = torch.randn(R, V)
    walls, flts, lst = [], [], []
    for _ in range(steps):
        f0, t0 = faults()[0], time.perf_counter()
        if reuse:
            gw, gb, l1, l2, *gs = bufs
            gw.fill_(1.0)
            gb.copy_(gw)
            torch.mul(src, 1.0, out=l1)
            torch.mul(l1, 0.5, out=l2)
            for g in gs:
                g.fill_(0.5)
        else:
            gw = torch.empty(V, H).fill_(1.0)
            gb = gw.to(torch.bfloat16)
            l1 = src * 1.0
            l2 = l1 * 0.5
            gs = [torch.empty(3072, H).fill_(0.5) for _ in range(20)]
        lst.append(float(gb[0, 0]) + float(l2[0, 0]) + sum(float(g[0, 0]) for g in gs))
        del gw, gb, l1, l2, gs
        walls.append(round(time.perf_counter() - t0, 4))
        flts.append(faults()[0] - f0)
        time.sleep(0.05)  # the gap between steps (logging, the dataloader)
    rec.update(step_wall_s=walls, step_faults=flts, peak_ws_mib=faults()[1], commit_mib=faults()[2], ok=len(set(lst)) == 1)
    print(TAG + json.dumps(rec), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="cpu_alloc_micro.json")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="ignored (staging passes it to every job)")
    a = ap.parse_args()
    if a.child:
        return child(a.steps)
    import torch
    rep = {"torch": torch.__version__, "platform": platform.platform(), "machine": platform.machine(),
           "processor": platform.processor(), "threads": torch.get_num_threads(), "mi_exports": mi_exports(), "cells": []}
    try:
        rep["cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    except Exception:  # noqa: BLE001
        pass
    try:
        rep["amx_tile"] = bool(torch._C._cpu._is_amx_tile_supported())
        rep["avx512_bf16"] = bool(torch._C._cpu._is_avx512_bf16_supported())
    except Exception as e:  # noqa: BLE001
        rep["isa_error"] = repr(e)[:120]
    try:
        with open("/proc/cpuinfo") as f:
            info = f.read()
        rep["brand"] = next((ln.split(":", 1)[1].strip() for ln in info.splitlines() if ln.startswith("model name")), "")
        flags = next((ln for ln in info.splitlines() if ln.startswith("flags")), "").split()
        rep["flags"] = [x for x in ("avx2", "avx512f", "avx512_bf16", "amx_tile", "amx_bf16", "amx_fp16") if x in flags]
    except OSError:
        pass
    try:  # mimalloc announces itself and its options with MIMALLOC_VERBOSE
        p = subprocess.run([sys.executable, "-c", "import torch; t = torch.empty(64 << 20, dtype=torch.uint8); del t"],
                           env=dict(os.environ, MIMALLOC_VERBOSE="1", MIMALLOC_SHOW_STATS="1"),
                           capture_output=True, text=True, timeout=300)
        rep["mimalloc_verbose"] = [ln[:200] for ln in (p.stderr + p.stdout).splitlines() if "mimalloc" in ln.lower()][:60]
    except Exception as e:  # noqa: BLE001
        rep["mimalloc_verbose"] = repr(e)[:200]
    print(json.dumps({k: v for k, v in rep.items() if k != "cells"})[:3000], flush=True)
    names = [x for x in a.arms.split(",") if x] or (
        ["d", "pd", "pd10", "nodc", "late", "api", "reuse"] if sys.platform == "win32" else ["d", "mo", "reuse"])
    for r in range(a.repeats):
        for name in names:
            env = dict(os.environ, **ARMS[name])
            p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--child", "--steps", str(a.steps)],
                               env=env, capture_output=True, text=True, timeout=1200)
            res = [ln[len(TAG):] for ln in p.stdout.splitlines() if ln.startswith(TAG)]
            cell = {"arm": name, "rep": r, "rc": p.returncode}
            if res:
                cell.update(json.loads(res[-1]))
                w = cell["step_wall_s"][2:]
                cell["med_wall_s"] = round(statistics.median(w), 4)
                cell["med_faults"] = statistics.median(cell["step_faults"][2:])
            else:
                cell["tail"] = (p.stdout + p.stderr).splitlines()[-20:]
            rep["cells"].append(cell)
            print(name, r, cell.get("med_wall_s"), cell.get("med_faults"), cell.get("step_wall_s"), cell.get("api", ""),
                  cell.get("tail", ""), flush=True)
    rep["summary"] = {n: {"med_wall_s": statistics.median([c["med_wall_s"] for c in rep["cells"] if c["arm"] == n and "med_wall_s" in c] or [0]),
                          "med_faults": statistics.median([c["med_faults"] for c in rep["cells"] if c["arm"] == n and "med_faults" in c] or [0])}
                      for n in names}
    print(json.dumps(rep["summary"]), flush=True)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
