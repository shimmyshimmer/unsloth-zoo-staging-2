"""Run repo pytest files on Apple Silicon (Metal) as a strict staging job, e.g. the *_metal.py suites
mlx-ci.yml does not list.

    python jobs/mlx_pytest.py --tests tests/test_mlx_speculative_metal.py [--k EXPR] [--out r.json]

Checks: pytest exit 0, at least one test passed (an all-skipped run is vacuous, never a pass).
Exit 3 off Apple Silicon / without Metal.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", required=True, help="space-separated pytest paths")
    ap.add_argument("--k", default=None, help="pytest -k expression")
    ap.add_argument("--out", default="mlx_pytest.json")
    ap.add_argument("--tiny", action="store_true", help="ignored (staging_ci)")
    a = ap.parse_args()
    out = Path(a.out)
    data = {"job": "mlx_pytest", "model": None, "backend": "mlx", "device": C.detect_device(),
            "config": vars(a), "versions": C.versions(), "revisions": C.revisions(),
            "steps": [], "summary": {}, "checks": {}, "passed": False, "error": None, "finished": False}

    def finish(code):
        data["finished"] = True
        out.write_text(json.dumps(data, indent=2, default=str))
        print("JOB_RESULT " + json.dumps({k: data[k] for k in ("job", "device", "passed")}
                                         | {"skipped": data.get("skipped"), "out": str(out)}), flush=True)
        sys.exit(code)

    try:
        import mlx.core as mx
        metal = mx.metal.is_available()
    except Exception as e:  # noqa: BLE001
        metal, data["error"] = False, f"{type(e).__name__}: {e}"
    if not metal:
        data["skipped"] = "Metal unavailable"
        finish(C.SKIP_EXIT_CODE)
    xml = out.with_suffix(".junit.xml")
    cmd = [sys.executable, "-m", "pytest", *shlex.split(a.tests), "-rs", "-v", f"--junitxml={xml}"]
    if a.k:
        cmd += ["-k", a.k]
    t0 = time.perf_counter()
    rc = subprocess.run(cmd).returncode
    counts, failed, skipped = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}, [], []
    if xml.is_file():
        for case in ET.parse(xml).iter("testcase"):
            name = f"{case.get('classname')}::{case.get('name')}"
            kind = next((child.tag for child in case if child.tag in ("failure", "error", "skipped")), "passed")
            counts["failed" if kind == "failure" else kind] += 1
            if kind in ("failure", "error"):
                failed.append(name)
            elif kind == "skipped":
                skipped.append(name)
    data["summary"] = {"pytest_rc": rc, "wall_s": round(time.perf_counter() - t0, 1), **counts,
                       "failed_tests": failed, "skipped_tests": skipped}
    data["checks"]["pytest_exit_0"] = {"ok": rc == 0, "detail": f"rc={rc} failed={failed}"}
    data["checks"]["ran_tests"] = {"ok": counts["passed"] > 0, "detail": counts}
    data["passed"] = all(c["ok"] for c in data["checks"].values())
    finish(0 if data["passed"] else 1)


if __name__ == "__main__":
    main()
