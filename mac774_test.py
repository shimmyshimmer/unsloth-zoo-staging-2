"""macOS / Apple Silicon validation of PR 774 through unsloth_zoo's REAL
convert_to_gguf wrapper.

Mirrors the Linux integration proof: take a Qwen3.5 model in the merged-save
state (mtp.* tensors absent, config STILL declares mtp_num_hidden_layers=1) and
confirm 774's in-wrapper strip yields a consistent GGUF block_count == 24 == the
number of present tensor layers (not the inflated 25 that current main produces).

The earlier macOS attempt failed because convert_to_gguf runs an external
converter (~/.unsloth/llama.cpp/unsloth_convert_hf_to_gguf.py) that it does NOT
auto-install; the caller must bootstrap it. Here we clone llama.cpp pinned to the
exact commit validated on Linux, point UNSLOTH_LLAMA_CPP_PATH / _SCRIPTS_DIR at
it, then _download_convert_hf_to_gguf() writes the patched converter beside the
conversion/ package before convert_to_gguf() runs.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
import gguf

LLAMA_COMMIT = "08023072ef63e7da749870b771db156f36d5c935"  # ggml-org/llama.cpp, has Qwen3.5 + MTP block_count logic
WORK = Path(os.environ.get("MAC774_WORK", "mac774_work")).resolve()
WORK.mkdir(parents=True, exist_ok=True)
LLAMA = WORK / "llama.cpp"

# 1. Clone llama.cpp pinned to the validated commit (brings the conversion/ package)
if not (LLAMA / "convert_hf_to_gguf.py").is_file():
    print(f"== cloning ggml-org/llama.cpp @ {LLAMA_COMMIT[:7]} ==", flush=True)
    subprocess.run(["git", "init", "-q", str(LLAMA)], check=True)
    subprocess.run(["git", "-C", str(LLAMA), "remote", "add", "origin",
                    "https://github.com/ggml-org/llama.cpp"], check=True)
    subprocess.run(["git", "-C", str(LLAMA), "fetch", "-q", "--depth", "1", "origin", LLAMA_COMMIT], check=True)
    subprocess.run(["git", "-C", str(LLAMA), "checkout", "-q", LLAMA_COMMIT], check=True)
assert (LLAMA / "conversion" / "qwen.py").is_file(), "conversion/qwen.py missing from llama.cpp checkout"

# Point unsloth_zoo at this in-workspace checkout (no network converter, no ~/.unsloth clone)
os.environ["UNSLOTH_LLAMA_CPP_PATH"] = str(LLAMA)
os.environ["UNSLOTH_LLAMA_CPP_SCRIPTS_DIR"] = str(LLAMA)
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

# 2. Build the merged-save state: drop mtp.* tensors, KEEP config (still has mtp key)
print("== download Qwen3.5-0.8B ==", flush=True)
src = snapshot_download("Qwen/Qwen3.5-0.8B", local_dir=str(WORK / "src"),
                        ignore_patterns=["*.gguf", "original/*", "*.pth"])
model_dir = WORK / "merged"
if model_dir.exists():
    shutil.rmtree(model_dir)
model_dir.mkdir()
st = glob.glob(os.path.join(src, "*.safetensors"))[0]
state = load_file(st)
keep = {k: v for k, v in state.items() if not k.lower().startswith("mtp.")}
print(f"== dropped {len(state) - len(keep)} mtp.* tensors, kept {len(keep)} ==", flush=True)
save_file(keep, str(model_dir / "model.safetensors"), metadata={"format": "pt"})
for f in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json",
          "generation_config.json", "preprocessor_config.json",
          "video_preprocessor_config.json", "chat_template.jinja", "config.json"):
    p = Path(src) / f
    if p.exists():
        shutil.copy(p, model_dir / f)
tc = json.loads((model_dir / "config.json").read_text()).get("text_config", {})
print(f"== BEFORE convert: text_config.mtp_num_hidden_layers = {tc.get('mtp_num_hidden_layers')} "
      f"(num_hidden_layers={tc.get('num_hidden_layers')}) ==", flush=True)

# 3. Bootstrap the patched converter, then run 774's real convert_to_gguf
from unsloth_zoo.llama_cpp import convert_to_gguf, _download_convert_hf_to_gguf, LLAMA_CPP_DEFAULT_DIR
print(f"== LLAMA_CPP_DEFAULT_DIR = {LLAMA_CPP_DEFAULT_DIR} ==", flush=True)
print("== _download_convert_hf_to_gguf() ==", flush=True)
_download_convert_hf_to_gguf()

os.chdir(model_dir.parent)
print("== convert_to_gguf (PR 774 wrapper, strips mtp in place) ==", flush=True)
files, is_vlm = convert_to_gguf(
    model_name="qwen35-0.8b-merged",
    input_folder=str(model_dir),
    quantization_type="bf16",
    supported_text_archs=None,   # skip arch gate
    print_output=True,
)

tc2 = json.loads((model_dir / "config.json").read_text()).get("text_config", {})
print(f"== AFTER convert: text_config.mtp_num_hidden_layers = {tc2.get('mtp_num_hidden_layers')} (expect None) ==", flush=True)

gguf_path = files[0] if isinstance(files, (list, tuple)) else files
print("== produced GGUF:", gguf_path, flush=True)
r = gguf.GGUFReader(str(gguf_path))
bc = None
for fld in r.fields.values():
    if fld.name.endswith("block_count"):
        bc = int(fld.parts[fld.data[0]][0])
layers = {int(m.group(1)) for t in r.tensors for m in [re.search(r"blk\.(\d+)\.", t.name)] if m}
print(f"\n==== RESULT on Apple Silicon: block_count={bc}, distinct tensor layers={len(layers)} ====", flush=True)
assert bc == 24, f"expected block_count 24 after PR 774 strip, got {bc}"
assert bc == len(layers), f"block_count {bc} must equal present tensor layers {len(layers)}"
print("PR774 OK on Apple Silicon through the REAL unsloth convert_to_gguf wrapper: "
      "GGUF block_count is consistent (24), not the inflated 25.", flush=True)
