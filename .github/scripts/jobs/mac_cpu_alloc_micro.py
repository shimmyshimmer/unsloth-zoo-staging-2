"""Apple Silicon leg of jobs/cpu_alloc_micro.py (staging CI runs only mac_* jobs on macOS arm64)."""

import os
import runpy
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.argv[0] = os.path.join(here, "cpu_alloc_micro.py")
runpy.run_path(sys.argv[0], run_name="__main__")
