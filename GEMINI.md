# Review-time standing instructions for gemini-3-pro

## Severity calibration -- worked examples

The reviewer pipeline rejects findings without specific triggers. Use these
examples to anchor your severity choices.

### [P0] -- Security or data-loss bug with a concrete trigger

Good: "rm -rf $UNSLOTH_HOME with no `set -u` runs as `rm -rf /` when
$UNSLOTH_HOME is unset under POSIX shell" -- cites the missing guard, the
specific shell behaviour, the destructive consequence.

Bad: "this script could potentially be exploited" -- no specific input, no
attack path, no verifiable trigger. Demote or remove.

### [P1] -- Bug with a triggering input or repro

Good: "uv pip install -r $NoTorchReq fails when $NoTorchReq path contains
spaces; uv 0.11.x truncates -r flag at first space; reproduces with
`uv pip install -r 'C:\Program Files\reqs.txt'`" -- specific tool, specific
version, specific input.

Bad: "the install path may have issues with spaces" -- no specific failure
mode, no trigger. Demote to P3 or remove.

### [P2] -- Behavioural difference or contract change with concrete impact

Good: "removing writer_batch_size=10 from dataset.map() changes memory
behaviour for vision DPO; user-visible OOM at batch sizes that worked
previously."

Bad: "this might affect performance" -- no measurement, no specific
operation, no trigger. Demote to P3 or remove.

### [P3] -- Style, comment, type-annotation, or minor consistency issue

Good: "_uv_safe_path(path: object) annotation is overly broad; all callers
pass Path or str."

Bad: "this code could be cleaner" -- subjective, no improvement specified.
Remove.

## Cross-block consistency check

For every guard, validation, error-path, ownership filter, or destructive
operation the diff adds or modifies, search the rest of the diff and the
surrounding source tree for analogous code paths that perform the same
operation WITHOUT the same protection. PRs that protect one code path
commonly miss a parallel path -- the asymmetric-fix pattern is a [P1]
finding worth surfacing.

Use grep_search heavily here. Pattern: when the diff introduces a check,
assertion, env-mode gate, ownership filter, or destructive operation
(`rm`, `Remove-Item`, file delete, process kill, table truncate, queue
purge, etc.), grep the workdir for the same operation in the absence of
the same check. Cite both file:line locations in the finding.

## Tool-use nudge

You are running in plan mode with these read-only tools available:
- grep_search (regex over the workdir)
- read_file (read any file by path)
- list_directory (enumerate directory contents)
- cli_help

Use them. The PR diff alone is insufficient context for cross-block
consistency checks, asymmetric-fix patterns, and surrounding-state
verification. For every finding you draft, grep at least once to verify
the surrounding code matches the diff's assumption. Wall-clocks under
300s typically indicate the tools were not used -- don't be that run.

grep_search uses regex: escape `( ) [ ] . * + ? | ^ $ \` when meaning them
literally (e.g. pattern `httpx\.Client\(` matches `httpx.Client(`).
