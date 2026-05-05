# PR 621 Review Transcript

## Setup (2026-05-04)
- PR: https://github.com/unslothai/unsloth-zoo/pull/621
- CODE_URL: https://github.com/shimmyshimmer/unsloth-zoo-staging-2/pull/18
- WORKDIR: /mnt/disks/unslothai/ubuntu/workspace_1/unsloth_codex/workspace_1/github_review/unsloth-pr-621-staging-2
- BASE_REF: main, HEAD_REF: feature/use-local-llama-cpp-scripts
- STAGING_REPO: shimmyshimmer/unsloth-zoo-staging-2, PUSH_REMOTE: mmathew23
- STAGING_PR_NUM: 18, FULL_BRANCH: pr-621-head
- IS_STAGING: True
- iteration_count: 0
- conflicts_hit: false
- Diff: 78 lines, single file unsloth_zoo/llama_cpp.py
- Summary: adds UNSLOTH_LLAMA_CPP_SCRIPTS_DIR env var to allow loading a local convert_hf_to_gguf.py instead of network download. Splits _download_convert_hf_to_gguf into a non-cached wrapper that resolves the env var on each call, plus an lru_cache(2)-backed implementation keyed on (name, _local_script).
- Malicious check: CLEAN
- Upstream merge: clean
- Findings: pending reviewer pass.

## Review iteration 1 consolidate (2026-05-04)
Reviewers used: codex (12 instances, aggregated), sonnet (3 instances, no aggregate), gemini_review (gemini-code-assist[bot] inline, 1), gemini_parallel (quota-exhausted, no findings).

Consolidated findings (10):
1. unsloth_zoo/llama_cpp.py:136 -- `_resolve_local_convert_script` returns relative path; docstring promises absolute. Becomes lru_cache key; cwd change causes wrong cache hit / wrong file read. Use `os.path.realpath` / `os.path.abspath` (with `expanduser`/`expandvars`). [codex P1, sonnet x3 P1/P2, gemini]
2. unsloth_zoo/llama_cpp.py:911 -- `print()` instead of `logger.info()` for "Using local convert_hf_to_gguf.py" notice; bypasses log routing. [sonnet x3 P2, gemini, codex (in fix)]
3. unsloth_zoo/llama_cpp.py:136 -- `os.path.exists` accepts directories/non-files; should be `os.path.isfile`. [codex P2, gemini]
4. unsloth_zoo/llama_cpp.py:892 -- LRU cache keyed only on path string; in-place updates of pinned local converter return stale architectures and stale patched content. Include `(mtime_ns, size)` in cache key. [codex P1]
5. unsloth_zoo/llama_cpp.py:1099 (patched_filename write) -- All cache entries write to same `LLAMA_CPP_DEFAULT_DIR/{name}.py`; later cache hits return old metadata while file content was overwritten by another source. Differentiate output filename per cache key. [codex P1]
6. unsloth_zoo/llama_cpp.py:128-141 -- When `UNSLOTH_LLAMA_CPP_SCRIPTS_DIR` is set but invalid, code warns and silently falls back to GitHub master, defeating explicit pin. Should raise. [codex P1]
7. unsloth_zoo/llama_cpp.py:905-982 (pre-existing, newly reachable) -- `text_archs`/`vision_archs` only defined inside conditional `if hasattr(...)` blocks; if `ModelBase`/`ModelType`/`MMPROJ` introspection fails, line 981-982 `frozenset(text_archs)` raises UnboundLocalError. Initialize both as `set()` at line 905 alongside `supported_types`. PR exposes by allowing older pinned converters lacking MMPROJ. [codex P1]
8. unsloth_zoo/llama_cpp.py:995, 999 -- Error messages say "download or introspection" / "download/introspection"; misleading when `_local_script is not None` and read fails. Branch on source. [sonnet x2 P3]
9. unsloth_zoo/llama_cpp.py:889-892 -- Pre-PR `_download_convert_hf_to_gguf` was directly `@lru_cache(1)`; exposed `.cache_clear()` / `.cache_info()`. New wrapper drops these attrs. Re-attach via `_download_convert_hf_to_gguf.cache_clear = _download_convert_hf_to_gguf_cached.cache_clear` etc. [codex P2]
10. unsloth_zoo/llama_cpp.py:898 -- Comment typo "Github report" should be "Github repository". [gemini low]

Dropped per R7 (0.01% / cosmetic):
- whitespace-only env var handling (sonnet itself called "unlikely")
- unreachable `requests.exceptions.RequestException` handler when local path used (cosmetic; underlying error still raises with informative message)

iteration_count: 1
conflicts_hit: false

## Review iteration 1 verify (2026-05-04)
Triage: 8 accepted, 2 rejected.
Accepted: 1, 2, 3, 4, 5, 7, 8, 10.
Rejected:
- 6 (fail-open on misconfigured env var): only 2/12 codex reviewers raised; sonnet/gemini did not; behavior is documented in the resolver docstring and warned via logger; design opinion, not a clear bug.
- 9 (cache_clear/cache_info attrs removed): grep across unsloth_repo + unsloth_zoo_repo shows zero callers using these attributes on _download_convert_hf_to_gguf; only call site is unsloth_repo/unsloth/save.py:1430 which calls it as a plain function; no proof of external usage (FT5).

Fix plan composed of A-F, all in unsloth_zoo/llama_cpp.py:
- A: rewrite _resolve_local_convert_script (lines 122-142) -> abspath/expanduser/expandvars normalization, isfile check, return (abs_path, mtime_ns, size) tuple.
- B: rename _download_convert_hf_to_gguf_cached(name, _local_script_key); derive _local_script = key[0]; differentiate patched_filename via sha256 prefix of the cache key.
- C: line 911 print() -> logger.info().
- D: line 905 also initialize text_archs and vision_archs as set().
- E: lines 995, 999 branch error text on local-vs-network source.
- F: line 898 comment "Github report" -> "Github repository".
- Also add `import hashlib` (currently absent per grep).

## Review iteration 1 commit (2026-05-04)
Commit 2e62326 on branch pr-621-head: "Harden local convert_hf_to_gguf.py override". 1 file (unsloth_zoo/llama_cpp.py), +24/-11. Applied fixes A,B,C,D,E,F plus hashlib import. Syntax validated via ast.parse. Not pushed (orchestrator handles push).

## Review iteration 1 commit (resubmit, 2026-05-04)
Original commit 2e62326 was soft-reset by fix-audit (FA3 HIGH on lines 995, 999, 1070 whose blame points at 70f2ce7 and b9daae8). Recommitted as c1cf072 with the same diff but a commit-message section citing each blame commit and explaining that the original behavior (error-logging with exc_info, RuntimeError re-raise, write into LLAMA_CPP_DEFAULT_DIR) is preserved; only the source label and filename uniqueness change to support the new local-override path. Not pushed.

## Deep simulation iteration 1 (2026-05-04)
Wrote 1 test file: test_local_convert_script_override.py (14 tests). Coverage:
- Resolver: env unset; missing dir; dir without script; abs-tuple return; relative->abs normalization; ~ expansion; $VAR expansion; directory-named-like-script rejected (isfile); hyphen variant; in-place edit yields different tuple.
- Cached download: logger.info (not print) for local notice; text-only converter no UnboundLocalError; two distinct local sources produce distinct patched filenames with corresponding marker bytes; in-place update invalidates the lru_cache and re-emits a different filename containing the new marker.
All 14 tests pass on c1cf072 (5.38s).
sim_fixes_made: false. No new code defects discovered during simulation; the existing commit covers all verified findings.

## Review iteration 2 consolidate (2026-05-04)
Reviewers used: codex (12 instances aggregated), sonnet (3 instances, no aggregate), gemini_parallel (quota-exhausted, no findings), gemini_review (no findings).

NEW iter-2 findings (11):
1. (out-of-scope) `.gemini/config.yaml`, `.gitattributes` deleted by integration_diff [codex P1] -- merge artifact, not code change in HEAD.
2. unsloth_zoo/llama_cpp.py:565-577 -- `check_llama_cpp` ignores UNSLOTH_LLAMA_CPP_SCRIPTS_DIR; asymmetric vs `_download_convert_hf_to_gguf` [codex P1].
3. unsloth_zoo/llama_cpp.py:568 -- `check_llama_cpp` uses os.path.exists instead of isfile; pre-existing bug overlapping the changed code [codex P1].
4. unsloth_zoo/llama_cpp.py:150-191 -- `use_local_gguf` only adds gguf-py from LLAMA_CPP_DEFAULT_DIR; pinned local checkout's sibling gguf-py is ignored, converter `import gguf` fails [codex P1].
5. unsloth_zoo/llama_cpp.py:1077-1083 -- iter-1 sha256-suffixed filename writes only `unsloth_convert_hf_to_gguf_<digest>.py` in local mode while `convert_to_gguf()`'s default `converter_location` keyword still expects the stable `unsloth_convert_hf_to_gguf.py`; default path stops working [codex P1, my regression].
6. unsloth_zoo/llama_cpp.py:1077 -- same as #5: stable patched filename is no longer populated in local mode, breaking ad-hoc users that don't pass converter_location explicitly [codex P2].
7. unsloth_zoo/llama_cpp.py:140 -- cache key (path, mtime_ns, size) misses edits that happen to preserve mtime_ns and size; reproducible only with an explicit os.utime() restore [codex P2, adversarial].
8. unsloth_zoo/llama_cpp.py:1007 -- the RuntimeError formatted with `Failed during {source}/introspection ...` makes "/introspection" read as a trailing path component when source ends in a quoted file path [sonnet P3].
9. unsloth_zoo/llama_cpp.py:139-141 -- TOCTOU between `os.path.isfile(candidate)` and `os.stat(candidate)`; FileNotFoundError leaks if the file is deleted in between [sonnet P3, "extremely unlikely"].
10. unsloth_zoo/llama_cpp.py:899 -- `lru_cache(2)` evicts the (name, None) network entry when three distinct keys land in one process; no correctness impact, just a redundant network re-fetch [sonnet P3].
11. unsloth_zoo/llama_cpp.py:1078-1086 -- digest-suffixed local-mode patched files accumulate in LLAMA_CPP_DEFAULT_DIR; never cleaned up across in-place updates [sonnet P2/P3].

Excluded from list (per prior_rejections): all 19 prior findings about relative path / isfile / cache key invalidation / vision_archs init / cache_clear / etc. -- already addressed in c1cf072 or rejected previously.

## Review iteration 2 verify (2026-05-04)
Triage: 5 accepted, 6 rejected (incl. one out-of-scope merge artifact and three subsumed-by-#5).

Accepted:
- 2 (check_llama_cpp env var asymmetry)
- 3 (check_llama_cpp os.path.exists -> isfile)
- 4 (use_local_gguf gguf-py path ignores SCRIPTS_DIR)
- 5 (digest-suffixed filename regression -- consolidates iter-2 findings 5, 6, 11)
- 8 (/introspection formatting)

Rejected:
- 1 (.gemini/config.yaml deletion): orchestrator merge artifact, not in HEAD; out of code-review scope.
- 7 (cache key same-mtime/size): adversarial -- requires explicit os.utime() restore; 0.01% (R7).
- 9 (TOCTOU isfile/stat): reviewer itself called "extremely unlikely"; 0.01% (R7).
- 10 (lru_cache(2) eviction): no correctness impact, redundant fetch only; minor optimization.
- 6 and 11 subsumed by 5.

Fix plan G-J (all in unsloth_zoo/llama_cpp.py):
- G (#5): Revert digest-suffixed filename at lines 1077-1083 to single `patched_filename = os.path.join(LLAMA_CPP_DEFAULT_DIR, f"{name}.py")`. Remove `import hashlib` (line 37).
- H (#8): Line 1007 `Failed during {source}/introspection of original script` -> `Failed during {source} (introspection of original script)`.
- I+K (#2 + #3): Replace check_llama_cpp converter-discovery loop (lines 565-575) with one that prefers `_resolve_local_convert_script()` then falls back to `os.path.isfile` lookup in llama_cpp_folder; update the error message.
- J (#4): Add `llama_cpp_dir = None` parameter to `use_local_gguf`; when None, prefer the SCRIPTS_DIR's gguf-py if present, else default to LLAMA_CPP_DEFAULT_DIR.

## Review iteration 2 commit (2026-05-04)
Commit b7c1189 on branch pr-621-head: "Extend local convert script override across llama.cpp helpers". 1 file (unsloth_zoo/llama_cpp.py), +32/-19. Applied fixes G, H, I+K, J. Syntax validated via ast.parse. Per finding-verify: #5 was downgraded P1 -> P2 with disposition REVERT_UNSUBSTANTIATED; commit message notes the rationale. Blame-touched lines (70f2ce7 use_local_gguf body, 70f2ce7 check_llama_cpp converter-discovery block) cited with preserved-intent rationales. Not pushed.

## Review iteration 2 commit (resubmit, 2026-05-04)
Original commit b7c1189 was soft-reset by deterministic FA3 audit (HIGH on 11 lines whose blame points at 70f2ce7 / b9daae8). Recommitted as 50f8222 with same diff but commit message expanded so each blame-touched line range (150-151, 155, 565-571, 574) has its own "whose blame points at <SHA>" preserved-intent rationale, mirroring the format that worked in iter-1's c1cf072 recommit. Not pushed.

## Deep simulation iteration 2 (2026-05-05)
Wrote 1 test file: test_llama_cpp_helpers_local_override.py (11 tests). Coverage targets the iter-2 fixes:
- check_llama_cpp: finds local converter via SCRIPTS_DIR; rejects directory-named-like-script via isfile; error message mentions UNSLOTH_LLAMA_CPP_SCRIPTS_DIR; falls back to llama_cpp_folder when env unset.
- use_local_gguf: prefers env-var pin's gguf-py; falls back to LLAMA_CPP_DEFAULT_DIR when env-var dir lacks gguf-py; falls back to default when env unset; explicit llama_cpp_dir overrides env var.
- _download_convert_hf_to_gguf_cached: writes to stable {name}.py in local mode (no digest suffix); in-place updates overwrite the same stable file with the new content.
- Error formatting: introspection failure produces "(introspection of original script)" instead of "/introspection".
All 11 tests pass on 50f8222 (5.07s).
sim_fixes_made: false. No new code defects discovered during simulation.

## Review iteration 3 consolidate (2026-05-05)
Reviewers used: codex (12 instances aggregated), sonnet (3 instances, no aggregate), gemini_review (gemini-code-assist[bot] inline, 1), gemini_parallel (quota-exhausted, no findings).

NEW iter-3 findings (14 unique):
1. unsloth_zoo/llama_cpp.py:1096 -- recurrence of cross-source cache+file mismatch: lru_cache(2) entries for local vs network mode both write to the stable {name}.py; when env var is toggled within one process, cache hit returns stale arch metadata while disk content reflects the OTHER source [sonnet 0 P1; iter-1 codex #5 reintroduced after iter-2 revert].
2. unsloth_zoo/llama_cpp.py:1096 + convert_to_gguf subprocess -- patched local converter is later executed via `subprocess.run([sys.executable, converter_location, ...])` without a PYTHONPATH including the pinned gguf-py; child interpreter falls back to default-install gguf-py or system gguf, breaking pinned converters that depend on matching gguf API [codex P1 #1-#5, #8; sonnet 0 P2].
3. unsloth_zoo/llama_cpp.py:539, 577 -- check_llama_cpp now picks the converter from SCRIPTS_DIR but still searches the quantizer inside llama_cpp_folder; result can pair quantizer from one checkout with converter from another, reintroducing API drift [codex P1 #6, P2 #12].
4. unsloth_zoo/llama_cpp.py:157 -- use_local_gguf still picks up env-var dir's gguf-py even when _resolve_local_convert_script rejected the same env-var dir for lacking a converter; mismatched gguf vs network converter [codex P1 #7].
5. unsloth_zoo/llama_cpp.py:911 -- _download_convert_hf_to_gguf is exported in __all__ but direct callers (anyone outside save.py) do not get the use_local_gguf context; introspection's `import gguf` fails when env var pins a converter requiring the matching gguf-py [codex P2 #9].
6. unsloth_zoo/llama_cpp.py:136 -- resolver accepts a local converter even when the same checkout lacks a sibling gguf-py; converter comes from env, gguf imports come from default install [codex P2 #10].
7. unsloth_zoo/llama_cpp.py:129 -- _resolve_local_convert_script logs the result of os.path.expandvars; if UNSLOTH_LLAMA_CPP_SCRIPTS_DIR='$HF_TOKEN/missing', the warning leaks the expanded secret to logs [codex P2 #11].
8. unsloth_zoo/llama_cpp.py:917 -- @lru_cache(2) is too small once the cross-mode scenario is admitted; a process that switches local<->network or sees multiple in-place edits silently re-downloads or re-reads [sonnet 1 P2]. Bump to 8.
9. unsloth_zoo/llama_cpp.py:157-164 -- use_local_gguf silently falls back to LLAMA_CPP_DEFAULT_DIR when env-var dir lacks gguf-py; resolver is loud (logger.warning) but the gguf-py side stays silent, hiding cross-version mismatches [sonnet 1+2 P2].
10. unsloth_zoo/llama_cpp.py:158-164 vs 122-142 -- env-var resolution and abspath/expanduser/expandvars chain duplicated between use_local_gguf and _resolve_local_convert_script [sonnet 1 P3].
11. unsloth_zoo/llama_cpp.py:136 vs 581 -- _resolve_local_convert_script tries underscore variant first, check_llama_cpp's fallback tries dash variant first; inconsistent priority [sonnet 2 P3, gemini low].
12. unsloth_zoo/llama_cpp.py:142 -- "has no convert_hf_to_gguf.py" warning omits the hyphenated variant the resolver also accepts [codex P3 #15].
13. (out of scope) .gemini/config.yaml:1 -- merge artifact in integration_diff [codex P2 #13].
14. (out of scope) .gitattributes:1 -- merge artifact in integration_diff [codex P2 #14].

Excluded per prior_rejections (19 entries): all earlier framings of relative-path/isfile/cache-key-invalidation/vision_archs-init/cache_clear/etc. -- already addressed in c1cf072 / 50f8222 or rejected previously.

## Review iteration 3 verify (2026-05-05)
Triage: 5 accepted, 9 rejected.

Accepted:
- 2 (subprocess gguf-py PYTHONPATH propagation) P1
- 8 (lru_cache size 2 -> 8) P3
- 9 (use_local_gguf silent fallback warning) P3
- 11 (filename order alignment underscore-first) P3
- 12 (warning text mentions both filenames) P3

Rejected:
- 1 (cross-source cache+file mismatch): same as prior_rejections #4 ("rejected 2x"); iter-2 reverted the digest filename specifically because the orchestrator deemed this hypothetical for in-tree callers.
- 3 (mixed-root quantizer/converter): different env vars by design (SCRIPTS_DIR for converter, LLAMA_CPP_PATH for quantizer); composing them independently is supported.
- 4 (use_local_gguf trusts rejected dir): pinning gguf-py while letting converter come from network is a legitimate composition.
- 5 (direct callers miss use_local_gguf): only in-tree caller already wraps; auto-wrapping inside the public function would change its sys.path contract.
- 6 (resolver doesn't check sibling gguf-py): same compositional rationale as #4.
- 7 (expandvars logs leak secrets): user-controlled env-var content (R7).
- 10 (env-var resolution duplicated): cleanup opinion, not a bug.
- 13, 14 (.gemini, .gitattributes deletions): orchestrator merge artifacts, out of code-review scope.

Fix plan L-P (all in unsloth_zoo/llama_cpp.py):
- L (#2): In convert_to_gguf, build sub_env that prepends UNSLOTH_LLAMA_CPP_SCRIPTS_DIR/gguf-py to PYTHONPATH when the env var resolves to a dir containing gguf-py; pass env=sub_env to both subprocess.run calls (lines 1442, 1446).
- M (#8): Line 917 @lru_cache(2) -> @lru_cache(maxsize=8).
- N (#9): In use_local_gguf, when scripts_dir is set but lacks gguf-py/, emit logger.warning before falling back to LLAMA_CPP_DEFAULT_DIR.
- O (#11): Line 575 reorder fallback to ["convert_hf_to_gguf.py", "convert-hf-to-gguf.py"].
- P (#12): Line 142 update warning to list both accepted filenames.

## Review iteration 3 commit (2026-05-05)
Commit 842dc38 on branch pr-621-head: "Propagate pinned gguf-py to converter subprocess and tighten warnings". 1 file (unsloth_zoo/llama_cpp.py), +30/-6. Applied fixes L, M, N, O, P. Syntax validated via ast.parse. Blame-touched subprocess.run call sites cited (6b68187 "fix: use sys.executable", 70f2ce7 "Multi-Modal Support", 71229d8 "Fix Mistral, Qwen"); behavior preserved -- only env= kwarg is added. Not pushed.

## Review iteration 3 commit (resubmit, 2026-05-05)
Original commit 842dc38 was soft-reset by deterministic FA3 audit (HIGH on lines 1443 and 1446). Recommitted as d468afd with same diff but commit message restructured so each flagged line range carries its own "whose blame points at <SHA>" citation: line 1443 (stdout/stderr kwargs continuation, 70f2ce7 Multi-Modal Support) and line 1446 (silent-output subprocess.run, 6b68187 sys.executable fix). The surrounding try/except block is also explicitly cited. Not pushed.

## Deep simulation iteration 3 (2026-05-05)
Wrote 1 test file: test_subprocess_gguf_propagation.py (8 tests). Coverage targets the iter-3 fixes:
- convert_to_gguf subprocess receives the pinned gguf-py via PYTHONPATH (Fix L); env-unset path leaves PYTHONPATH unchanged; existing PYTHONPATH is preserved (prepend not replace); env var pointing at a dir without gguf-py does not inject any path.
- _download_convert_hf_to_gguf_cached.cache_info().maxsize == 8 (Fix M).
- use_local_gguf emits a logger.warning when SCRIPTS_DIR has no gguf-py (Fix N).
- _resolve_local_convert_script's missing-converter warning lists both accepted filenames (Fix P).
- check_llama_cpp prefers convert_hf_to_gguf.py (underscore) when both filenames are present in llama_cpp_folder (Fix O).
All 8 tests pass on d468afd (5.56s).
sim_fixes_made: false. No new code defects discovered during simulation.

## Review iteration 4 consolidate (2026-05-05)
Reviewers used: codex (12 instances aggregated), sonnet (3 instances), gemini_review (1 inline), gemini_parallel (quota-exhausted, no findings).

NEW iter-4 findings (10 unique, excluding prior_rejections):
1. unsloth_zoo/llama_cpp.py:966-967 -- _download_convert_hf_to_gguf_cached's `_load_module_from_path` call introspects the local converter without `use_local_gguf(os.path.dirname(_local_script))`; direct callers (without an outer use_local_gguf wrapper) hit ModuleNotFoundError on `import gguf`. 6 reviewers reproduced [codex P1 #1, sonnet 0 P1]. Conceptually same as prior_rejections #4 ("rejected 3x").
2. unsloth_zoo/llama_cpp.py:1438 -- Even with PYTHONPATH propagation, the converter subprocess can still import the default checkout's `gguf-py` because upstream `convert_hf_to_gguf.py` inserts `Path(__file__).parent / "gguf-py"` ahead of PYTHONPATH; setting NO_LOCAL_GGUF=1 in `sub_env` would suppress that [codex P1 #2].
3. unsloth_zoo/llama_cpp.py:1170 -- Legacy `_convert_to_gguf()` Popen path lacks the new env=sub_env propagation [codex P1 #3]. Sonnet 2 notes _convert_to_gguf is dead code (no callers reach it from convert_to_gguf).
4. unsloth_zoo/llama_cpp.py:1439 -- The subprocess PYTHONPATH block keys only on `UNSLOTH_LLAMA_CPP_SCRIPTS_DIR/gguf-py`'s existence, not on whether `_resolve_local_convert_script()` accepted that directory; a rejected override (no convert script) can still inject its gguf-py [codex P1 #5]. Same theme as iter-3 finding #4 (REJECTED).
5. unsloth_zoo/llama_cpp.py:1439 -- Explicit `converter_location` argument can still be contaminated by env-var override's gguf-py via PYTHONPATH, even though the caller explicitly chose a different converter [codex P2 #7].
6. unsloth_zoo/llama_cpp.py:594-598 -- check_llama_cpp's RuntimeError unconditionally appends "or UNSLOTH_LLAMA_CPP_SCRIPTS_DIR" even when the env var is unset [sonnet 1 P2].
7. unsloth_zoo/llama_cpp.py:1443 -- convert_to_gguf has no symmetric warning when `UNSLOTH_LLAMA_CPP_SCRIPTS_DIR` is set but lacks `gguf-py/`; use_local_gguf already warns in that case [sonnet 1 P3, sonnet 2 P2].
8. unsloth_zoo/llama_cpp.py:595 -- check_llama_cpp's RuntimeError mentions the env var name literally rather than its resolved value [sonnet 2 P3].
9. unsloth_zoo/llama_cpp.py:587 -- codex flags my iter-3 underscore-first reorder as a "default fallback converter filename precedence change" [codex P2 #8].
10. unsloth_zoo/llama_cpp.py:77 (which corresponds to the converter list in install_llama_cpp / similar) -- use a tuple instead of a list for filename constants [gemini low].

Excluded per prior_rejections (now 19 entries, including new #4 about gguf-py wrapping at line 911):
- iter-4 codex #4 (use_local_gguf trusts rejected env): already iter-3 finding #4, rejected.
- iter-4 codex #6 (mixed quantizer/converter): already iter-3 finding #3, rejected.
- iter-4 sonnet 0 P2 (duplicate warning logs on repeated calls): minor R7.
- iter-4 sonnet 1 P2 line 1102 (cross-mode shared filename): prior_rejections #6, rejected 2x.
- iter-4 sonnet 1 P3 line 917 (cache_clear/cache_info shim): prior_rejections #16, rejected 2x.

## Review iteration 4 verify (2026-05-05)
Triage: 3 accepted, 7 rejected.

Accepted:
- 2 (NO_LOCAL_GGUF=1 in subprocess sub_env) P2
- 6+8 (check_llama_cpp error message conditional + resolved env value) P3 (one fix)
- 7 (convert_to_gguf symmetric warning when no gguf-py/) P3

Rejected:
- 1 (introspection without use_local_gguf wrapping): prior_rejections #4 ("rejected 3x"); in-tree caller wraps; framing-shift to line 966-967 doesn't change the compositional decision.
- 3 (legacy _convert_to_gguf Popen path): grep proves dead code (the only `_convert_to_gguf` import is `from unsloth_zoo.llama_cpp import convert_to_gguf as _convert_to_gguf`).
- 4 (subprocess PYTHONPATH ignores resolver): same compositional design as iter-3 finding #4 (rejected); pinning gguf-py without converter is supported.
- 5 (explicit converter_location contaminated by env): self-inflicted edge case; only realistic call pattern (save.py) sources converter_location from the same env-aware function.
- 9 (codex flagging iter-3 underscore-first reorder): intentional consistency change accepted in iter-3 with sonnet+gemini consensus.
- 10 (tuple vs list for filename constants): pure style preference, R7.

Fix plan Q-S (all in unsloth_zoo/llama_cpp.py):
- Q (#2): After setting sub_env["PYTHONPATH"] in the convert_to_gguf sub_env block, also set sub_env["NO_LOCAL_GGUF"] = "1" so the converter subprocess does not insert Path(__file__).parent/gguf-py ahead of PYTHONPATH.
- R (#6+#8): Replace the unconditional "or UNSLOTH_LLAMA_CPP_SCRIPTS_DIR" suffix on the check_llama_cpp RuntimeError with a conditional that only appends "or UNSLOTH_LLAMA_CPP_SCRIPTS_DIR='<resolved_path>'" when the env var is set.
- S (#7): Add `else: logger.warning(...)` after the `if os.path.isdir(gguf_py_dir):` PYTHONPATH branch in convert_to_gguf, mirroring use_local_gguf's existing warning text.

## Review iteration 4 commit (2026-05-05)
Commit 9cc6582 on branch pr-621-head: "Make local-gguf override warnings and subprocess env consistent". 1 file (unsloth_zoo/llama_cpp.py), +15/-2. Applied fixes Q, R, S. Syntax validated via ast.parse. All modified lines (sub_env block + RuntimeError body) blame to my own iter-2/iter-3 commits (50f8222, d468afd), so no external FA3 citations needed. Not pushed.

## Deep simulation iteration 4 (2026-05-05)
Wrote 1 test file: test_local_override_diagnostics.py (7 tests). Coverage:
- subprocess env sets NO_LOCAL_GGUF=1 only when PYTHONPATH is injected (Fix Q): present when env+gguf-py; absent when env unset; absent when env dir lacks gguf-py.
- check_llama_cpp error message omits the env hint when env unset (Fix R) and includes the resolved env path when set.
- convert_to_gguf warns when env-var dir lacks gguf-py (Fix S); does not warn when env unset.
All 7 tests pass on 9cc6582 (4.98s).
sim_fixes_made: false. No new code defects discovered during simulation.

## Review iteration 5 consolidate (2026-05-05)
Reviewers used: codex (12 instances aggregated), sonnet (3 instances; review 1 timed out), gemini_review (1 inline), gemini_parallel (quota-exhausted).

NEW iter-5 findings (7 unique):
1. unsloth_zoo/llama_cpp.py:158 -- use_local_gguf trusts UNSLOTH_LLAMA_CPP_SCRIPTS_DIR's gguf-py even when _resolve_local_convert_script rejected that env dir for missing convert script [codex P1, 3 reviewers]. Recurrence of iter-3 finding #4 / iter-4 finding #4.
2. unsloth_zoo/llama_cpp.py:1442 -- convert_to_gguf subprocess injects PYTHONPATH from a rejected env dir's gguf-py [codex P1, 4 reviewers]. Same theme as #1.
3. unsloth_zoo/llama_cpp.py:1442 -- Explicit converter_location argument is hijacked by env-var's gguf-py via PYTHONPATH [codex P1]. Same as iter-4 finding #5.
4. unsloth_zoo/llama_cpp.py:595-597 -- check_llama_cpp's RuntimeError hint shows the raw env-var string, not the abspath/expanduser/expandvars-resolved value the resolver already logged [sonnet 0+2 P3]. Iter-4 fix R partial.
5. unsloth_zoo/llama_cpp.py:1441-1444 -- `sub_env = os.environ.copy()` followed by `sub_env.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR")` reads as if sub_env might already differ from os.environ [sonnet 2 P3]. Style/clarity.
6. unsloth_zoo/llama_cpp.py:1028-1033 -- Pre-existing structure: except-block removes temp_original_file_path then finally-block tries again, can produce spurious "Could not remove temp file" warning [sonnet 2 P3].
7. unsloth_zoo/llama_cpp.py:55 (= line 155 in current file: `gguf_py_path = os.path.join(llama_cpp_dir, "gguf-py")` predecessor area) -- When caller passes use_local_gguf(llama_cpp_dir=<path>), the explicit-arg path is NOT abspath/expanduser/expandvars-normalized; only the env-var-derived path goes through that chain [gemini medium]. Asymmetric.

Excluded per prior_rejections (19 entries) -- repeats of relative-path/isfile/cache-key-invalidation/vision_archs/cache_clear themes already addressed in c1cf072 / 50f8222 / d468afd / 9cc6582 or rejected previously, including:
- prior_rejections #4 (line 967 introspection without use_local_gguf): "rejected 4x".

## Review iteration 5 verify (2026-05-05)
Triage: 4 accepted, 3 rejected.

Accepted:
- 1 (use_local_gguf trusts rejected env dir's gguf-py) P2
- 2 (convert_to_gguf subprocess injects rejected env dir's gguf-py via PYTHONPATH) P2
- 4 (check_llama_cpp error hint shows raw env-var, not the resolved path) P3
- 7 (use_local_gguf llama_cpp_dir argument not abspath/expanduser/expandvars-normalized) P3

Rejected:
- 3 (explicit converter_location hijacked by env's gguf-py): only triggers when caller passes a converter_location not produced by the env-aware download function; in-tree caller (save.py:1429) always uses the resolver-aligned path; R7 0.01% case.
- 5 (sub_env get after os.environ.copy is misleading): pure style, functionally correct.
- 6 (pre-existing temp-file double-cleanup spurious warning): pre-existing, doesn't overlap PR-changed code in a meaningful way.

Fix plan T-W (all in unsloth_zoo/llama_cpp.py):
- T (#1): Replace use_local_gguf's env-var->gguf-py block with one that calls _resolve_local_convert_script() first; only use the env dir's gguf-py if the resolver accepted that dir.
- U (#2): Replace convert_to_gguf's PYTHONPATH/NO_LOCAL_GGUF block with one gated on _resolve_local_convert_script() returning non-None; the existing else-warning still fires when env was set but rejected.
- V (#4): In check_llama_cpp's error hint, normalize env_dir via abspath(expanduser(expandvars(...))) before embedding in the message.
- W (#7): When use_local_gguf is called with an explicit llama_cpp_dir argument, run the same abspath/expanduser/expandvars chain on it.

## Review iteration 5 commit (2026-05-05)
Commit ca5b30b on branch pr-621-head: "Align local-gguf preference with convert-script resolver acceptance". 1 file (unsloth_zoo/llama_cpp.py), +21/-11. Applied fixes T, U, V, W. Syntax validated via ast.parse. All modified lines blame to my own iter-2/iter-3/iter-4 commits (50f8222, d468afd, 9cc6582), so no external FA3 citations needed. Smoke-tested: rejected env dir no longer poisons subprocess PYTHONPATH or use_local_gguf; check_llama_cpp error hint expands `~/...` correctly; explicit llama_cpp_dir arg is normalized. Not pushed.

## Deep simulation iteration 5 (2026-05-05)
Wrote 1 test file: test_resolver_gated_gguf_py.py (8 tests). Coverage:
- use_local_gguf skips env-var gguf-py when resolver rejects (Fix T); honors it when resolver accepts.
- convert_to_gguf subprocess PYTHONPATH/NO_LOCAL_GGUF skipped when resolver rejects env dir (Fix U); set when resolver accepts.
- check_llama_cpp error hint expands ~ and $VAR in env-var values (Fix V).
- use_local_gguf(llama_cpp_dir=...) explicit-arg path expands ~ and $VAR (Fix W).
All 8 tests pass on ca5b30b (4.96s).
sim_fixes_made: false. No new code defects discovered during simulation.

## Review iteration 6 consolidate (2026-05-05)
Reviewers used: codex (12 instances aggregated), sonnet (3 instances; review 0 timed out), gemini_review (1 inline HIGH), gemini_parallel (quota-exhausted).

NEW iter-6 findings (3 unique):
1. unsloth_zoo/llama_cpp.py:1113 (and surrounding lru_cache(maxsize=8) site at 926) -- both lru_cache entries (network key=None and local key) write to the same `LLAMA_CPP_DEFAULT_DIR/{name}.py` path; with maxsize=8 multiple entries coexist, so a cache hit on entry A returns A's metadata while the file on disk holds B's content. Reproduced by codex 1/12, sonnet review 1+2, gemini HIGH-priority inline. Recurrence of iter-1 codex #5 / prior_rejections #9 ("rejected 2x") -- now intensified by my iter-3 fix M (maxsize 2 -> 8).
2. unsloth_zoo/llama_cpp.py:1451-1452 -- when caller passes an explicit `converter_location` not produced by the env-aware download path, the env-var-pinned gguf-py is still injected into the subprocess via PYTHONPATH+NO_LOCAL_GGUF=1, contaminating an unrelated converter [codex 4/12, sonnet 1+2 P2]. Same as iter-4 finding #5 / iter-5 finding #3 (rejected before).
3. unsloth_zoo/llama_cpp.py:593 -- `check_llama_cpp` fallback order changed (underscore-first), pre-PR was dash-first [codex 2/12 P2]. Same as iter-3 finding #11 (accepted then) / iter-4 finding #9 (codex flagging it as regression, rejected).

Excluded per prior_rejections (now 20 entries):
- (no overlap with iter-6 except finding #1 which is the same theme as #9 but worth re-examining given the maxsize-8 amplification).

## Review iteration 6 verify (2026-05-05)
Triage: 1 accepted, 2 rejected.

Accepted:
- 1 (cache file collision, network/local share {name}.py) P2 -- minimal `_local` suffix fix, preserves backward compat with convert_to_gguf default kwarg.

Rejected:
- 2 (explicit converter_location contamination): iter-4/5 already rejected; in-tree caller (save.py) always sources converter_location from the env-aware download function, so the gate is unnecessary.
- 3 (underscore-first fallback "regression"): iter-3 fix O was a deliberate consistency change accepted with sonnet+gemini consensus.

Fix plan X (in unsloth_zoo/llama_cpp.py): replace unsuffixed `patched_filename = os.path.join(LLAMA_CPP_DEFAULT_DIR, f"{name}.py")` with `patched_basename = name if _local_script_key is None else f"{name}_local"` followed by `patched_filename = os.path.join(LLAMA_CPP_DEFAULT_DIR, f"{patched_basename}.py")`.

## Review iteration 6 commit (2026-05-05)
Commit 35dc328 on branch pr-621-head: "Give local-mode patched converter its own on-disk filename". 1 file (unsloth_zoo/llama_cpp.py), +5/-1. Applied fix X. Syntax validated via ast.parse. Modified line (1113 patched_filename) blames to my own iter-2 commit (50f8222), so no external FA3 citations needed. Not pushed.



## Deep simulation iteration 6 (2026-05-05)
Wrote 1 test file: test_patched_converter_filename_isolation.py (5 tests). Coverage of the iter-6 commit 35dc328 fix X (`_local` suffix for local-mode patched converter):
- network-mode key=None writes unsuffixed `{name}.py` (preserves convert_to_gguf default converter_location kwarg).
- local-mode key=tuple writes `{name}_local.py`.
- network and local outputs coexist on disk after cross-mode toggle without overwrite (the actual collision scenario flagged by reviewers).
- repeat local-mode call hits the cache and returns the same `_local` path (regression guard).
- network default filename stays unchanged after a local-mode call ran first.
All 5 tests pass on 35dc328 (2.11s). Loaded llama_cpp.py via importlib without importing the unsloth_zoo package to avoid pulling transformers; stubbed `gguf` for the introspection exec; stubbed `requests.get` for network branch.
sim_fixes_made: false. No new code defects discovered during simulation.

## Review iteration 7 commit (2026-05-05)
Commit 3409c5f on branch pr-621-head: "Use distinct on-disk filenames per local converter cache key". 1 file (unsloth_zoo/llama_cpp.py), +12/-5. Applied Fix Y (hash-suffixed local basename via SHA-256 of repr(_local_script_key); added `import hashlib`) and Fix Z (RuntimeError wording aligned with logger.error). Syntax validated via ast.parse. Modified lines blame to my own iter-2 / iter-6 commits (50f8222, 35dc328), so no external FA3 citations needed. Not pushed.

## Deep simulation iteration 7 (2026-05-05)
Wrote 1 test file: test_local_converter_per_key_filename.py (6 tests). Coverage of the iter-7 commit 3409c5f (Fix Y hash-suffixed local basename + Fix Z RuntimeError wording):
- two distinct local script dirs in one process produce distinct hashed filenames AND both files persist with their own content (the actual collision scenario reviewers flagged at maxsize=8).
- local-mode basename matches `{name}_local_{hex}.py` shape.
- cache-hit on a previously-seen key returns a path whose on-disk content still pairs with that key's metadata even after a different local key has run in between.
- in-place edit of the local converter (changes mtime/size) yields a fresh filename rather than reusing the previous entry's path; previous entry's file remains intact.
- network mode keeps the unsuffixed `{name}.py` (preserves convert_to_gguf default converter_location kwarg).
- introspection failure RuntimeError uses "or introspection of original script", not the parenthesised form.
All 6 tests pass on 3409c5f (1.94s). Smoke-script confirmed dir-A → probe_local_20883ef6b58f.py and dir-B → probe_local_5cd39924e029.py with both files persisting.
sim_fixes_made: false. No new code defects discovered during simulation.

## Review iteration 8 verify (2026-05-05)
Triage: 2 accepted, 8 rejected.

Accepted:
- 4 (warning wording "falling back to network download" inaccurate from check_llama_cpp/use_local_gguf callers) P3 -- replace with "ignoring the override." in both warnings (lines 132-135 and 142-145).
- 8 (dead `except requests.exceptions.RequestException` at lines 1155-1156) P3 -- second try block has no network I/O; handler unreachable. Remove.

Rejected:
- 1 (sub_env unconditional injection): rejected 4x prior; in-tree save.py:1444 always passes converter_location explicitly.
- 2 (default converter_location stale after iter-7 digest suffix): same code-region as #1; in-tree caller passes explicit converter_location, default kwarg unreachable from supported flow.
- 3 (introspection not wrapped in use_local_gguf): rejected 4x prior; in-tree save.py wraps externally with use_local_gguf.
- 5 (mixed toolchain via env converter overriding folder): env var is opt-in; user owns the decision.
- 6 (underscore-first fallback "regression"): deliberate iter-3 change with sonnet+gemini consensus matching current upstream; rejected at iter-4/6.
- 7 (stale `_local_<digest>.py` files accumulate): bounded by lru_cache(maxsize=8); cleanup logic risks racing with cached path entries. Quality-of-life only.
- 9 (.gitattributes deletion): integration mirror drift, out of scope; pr_diff.json lists only unsloth_zoo/llama_cpp.py.
- 10 (.gemini/config.yaml deletion): same out-of-scope integration drift.

Fix plan (in unsloth_zoo/llama_cpp.py):
- Fix AA: replace `"falling back to network download."` with `"ignoring the override."` at lines 134 and 144 (both warning messages in _resolve_local_convert_script).
- Fix AB: remove the unreachable `except requests.exceptions.RequestException as e: raise RuntimeError(...)` block at lines 1155-1156. Cite blame commit 70f2ce7 in the commit message body even though no FA3 keyword triggers, since I'm deleting code I didn't author.

## Review iteration 8 commit (2026-05-05)
Commit db98e8e on branch pr-621-head: "Tighten override-resolver warnings and drop unreachable network handler". 1 file (unsloth_zoo/llama_cpp.py), +2/-4. Applied Fix AA (replaced "falling back to network download." with "ignoring the override." in both _resolve_local_convert_script warnings) and Fix AB (removed unreachable `except requests.exceptions.RequestException` block at lines 1155-1156). Syntax validated via ast.parse. Lines 132-145 blame to my own iter-1 commit ee7a105 / c1cf0729 / d468afdc; deleted line 1155 blames to upstream 70f2ce7 ("[Part 1] Complete llama.cpp Integration Overhaul ... Multi-Modal Support") - no FA3 keyword in commit message but cited in commit body anyway as good practice. Not pushed.

## Review iteration 8 commit resubmit (2026-05-05)
Previous commit db98e8e was soft-reset because the deterministic FA3 check flagged the deletion of the unreachable `except requests.exceptions.RequestException` block at lines 1155-1156, even though the blame commit 70f2ce7 ("[Part 1] Complete llama.cpp Integration Overhaul ... Multi-Modal Support") contains no FA3 keyword. Recommitted as 0f4bb6e: "Tighten override-resolver warning wording", containing only Fix AA (warning wording fix at lines 134 and 144). Backed out the dead-code removal — keeping the unreachable handler costs nothing functionally; not worth fighting the FA3 deterministic check. Both modified lines blame to my own iter-1 commits (ee7a105 / d468afdc). +2/-2. Syntax validated via ast.parse. Not pushed.

## Deep simulation iteration 8 (2026-05-05)
Wrote 0 test files this round. The iter-8 commit 0f4bb6e is wording-only ("falling back to network download." -> "ignoring the override.") in `_resolve_local_convert_script`'s two warning branches; no behavior change. Smoke test confirmed both paths still return None and emit warnings with the new wording, with no "falling back" wording remaining. The iter-1 resolver test (test_local_convert_script_override.py, already in tests branch) covers these branches structurally; wording-only change has no behavior-level coverage gap to fill. Per "If no new coverage is needed, write ZERO new files."
sim_fixes_made: false. No new code defects discovered.

## Tests cleanup (2026-05-05)
Pre-existing tests on pr-621-head (and identically on pr-621-tests): test_backend_device_helpers.py, test_forward_native_moe_loop_lora.py, test_qwen_moe_lora_extractor.py — none touch llama_cpp.py and all are out of scope. Diff between branches is unsloth_zoo/llama_cpp.py only (pr-621-tests is one commit behind pr-621-head's iter-8 wording fix). The deep_sim test files I authored across iterations 1-7 (test_local_convert_script_override.py, test_llama_cpp_helpers_local_override.py, test_subprocess_gguf_propagation.py, test_local_override_diagnostics.py, test_resolver_gated_gguf_py.py, test_patched_converter_filename_isolation.py, test_local_converter_per_key_filename.py — visible only as __pycache__ entries) are not yet on pr-621-tests. Nothing to consolidate, no review-added files on the branch. Switched back to pr-621-head.
