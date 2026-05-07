# Architecture Improvement Handoff

Date: 2026-05-07

This handoff captures the selected architecture candidates from the codebase
review so a future agent can start implementation without repeating discovery.

## Required Context

Read these before editing:

- `AGENTS.md` for repo-wide expectations.
- `docs/agents/domain.md` for domain-doc conventions.
- `docs/agents/modal.md` for Modal platform vocabulary.
- `docs/agents/workflow-development.md` before touching `src/biomodals/workflow/`.
- `docs/agents/app-development.md` and `.agents/skills/biomodals-app-development/SKILL.md` before touching `src/biomodals/app/**/*_app.py`.
- `docs/agents/ppiflow-workflow.md` before changing PPIFlow workflow behavior.

Important repo facts:

- The worktree was clean when this handoff was written.
- There is currently no committed `tests/` directory.
- Prefer `uv run ...` for project commands.
- After code edits, run `prek run --files <changed files>` when practical.
- For CLI or app-discovery changes, smoke test with:
  - `uv run biomodals list`
  - `uv run biomodals help alphafold3`
  - `uv run biomodals help rosetta`
  - `uv run biomodals help boltzgen`
- Do not claim Modal end-to-end behavior works unless it was actually run on
  Modal. Local tests should isolate command shaping, path resolution, and run
  state logic where possible.
- For the current PR, local tests are fine during development, but test files
  should not be included in the final PR.
- Use `polars` for tabular parsing/writing in Python code unless an upstream
  API specifically requires a different library.

Use this architecture vocabulary in plans and PR notes:

- Module, Interface, Implementation, Depth, Deep, Shallow, Seam, Adapter,
  Leverage, Locality, deletion test.

## Selected Candidates

The user selected these candidates for further exploration:

1. App discovery and help rendering.
2. Local result materialization.
3. Dual runner drift.
4. Volume run state.

Recommended implementation order:

1. Local result materialization, because it is low risk and creates a testing
   foothold.
2. App discovery and help rendering, because it is isolated from Modal runtime
   execution and unlocks better CLI tests.
3. Dual runner drift, starting with one tool only.
4. Volume run state, after tests and naming are established.

## Candidate 1: App Discovery And Help Rendering

Files:

- `src/biomodals/cli.py`
- Existing app files under `src/biomodals/app/**/*_app.py`

Observed friction:

- `cli.py` owns app discovery, path resolution, module path conversion, module
  import, Modal object inspection, local entrypoint docstring parsing, table
  rendering, and command execution.
- `help`, `run`, and `deploy` each perform app-name-or-path resolution.
- `_docstring_to_markdown_table()` is useful behavior, but it is hidden inside
  the Typer CLI module and is awkward to test through the current Interface.
- The current Module is Shallow: the CLI Interface is almost as complex as the
  Implementation because callers must know path formats, `::entrypoint`
  handling, import behavior, and Modal inspection details.

Likely target shape:

- Add a deeper app catalog Module, likely `src/biomodals/app/catalog.py` or
  `src/biomodals/cli_app_catalog.py`.
- Keep Typer commands in `cli.py` thin. They should call the catalog Module and
  render results.
- Candidate concepts for the Module:
  - app name/path resolution
  - app metadata discovery
  - entrypoint selection via `app-name::entrypoint`
  - local entrypoint flag docs from Google-style `Args:`
  - remote Modal function names for help output
- Do not over-abstract Modal itself. One Adapter for real imports is enough at
  first; test helper modules can provide fake modules/objects as internal seams.

First implementation slice:

1. Move pure path parsing/resolution and docstring flag extraction out of
   `cli.py`.
2. Add local tests for:
   - app-name lookup
   - path lookup
   - `::entrypoint` parsing
   - missing app error shape
   - Google-style docstring continuation lines
3. Update `cli.py` to call the new Module without changing command output.

Verification:

- `uv run biomodals list`
- `uv run biomodals help alphafold3`
- `uv run biomodals help rosetta`
- `uv run biomodals run alphafold3` should still print help when no flags are
  passed.
- `prek run --files src/biomodals/cli.py <new files> <new tests>`

Open questions:

- Should app discovery include workflow files later, or remain app-only?
- Should hidden/legacy apps be filterable, or should the catalog preserve the
  current `*/*_app.py` behavior exactly for now?

## Candidate 2: Local Result Materialization

Files with repeated behavior:

- `src/biomodals/app/fold/alphafold3_app.py`
- `src/biomodals/app/fold/protenix_app.py`
- `src/biomodals/app/fold/flowpacker_app.py`
- `src/biomodals/app/score/dockq_app.py`
- `src/biomodals/app/score/abnativ_app.py`
- `src/biomodals/app/bioinfo/paddleocr_app.py`
- `src/biomodals/app/design/ligandmpnn_app.py`
- `src/biomodals/workflow/ppiflow_workflow.py`

Observed friction:

- Many local entrypoints repeat:
  - resolve `out_dir` or default to `Path.cwd()`
  - choose a tarball/result filename
  - check overwrite policy
  - create the local directory
  - write returned bytes
  - print final location
- Behavior is inconsistent:
  - Some apps preflight existing tarballs.
  - Some apps write without an overwrite check.
  - Names vary between `out_dir`, `output_dir`, app suffixes, and directory
    outputs.
- The deletion test is strong: deleting a shared Module would re-spread this
  policy across many app wrappers.

Updated target shape from PR #23 review:

- Add a deeper Module in `src/biomodals/helper/io.py`; do not call it
  `output.py`, because it may grow beyond output filename construction.
- Keep the Interface small and boring:
  - resolve local output directory
  - build output path from prefix, run name, suffix, and extension
  - sanitize prefix, run name, and suffix so final filenames are clean
  - enforce overwrite policy inside `build_local_output_path(...)`
  - write bytes
  - return the final `Path`
- Do not keep a separate public `ensure_output_file_available(...)` helper for
  normal output construction. The existence check belongs on the normalized
  output path.
- Keep user-facing print text in app entrypoints initially, unless the helper
  can make that consistent without hiding app-specific wording.

First implementation slice:

1. Implement a helper for tarball writes and clean local path construction.
2. Migrate 2-3 reference apps first:
   - `alphafold3_app.py`
   - `flowpacker_app.py`
   - `protenix_app.py`
     Add `dockq_app.py` if it is already being touched for workflow review.
3. Add tests using temporary directories for:
   - default CWD behavior, if exposed through a pure function
   - explicit output directory
   - existing file raises by default
   - optional force/overwrite if introduced
   - safe filename behavior if the helper owns run-name sanitization

Do not migrate every app in one pass. Keep the first PR small enough to review.

Verification:

- `uv run biomodals help alphafold3`
- `uv run biomodals help flowpacker`
- `uv run biomodals help protenix`
- `prek run --files src/biomodals/helper/io.py <changed app files>`

Resolved expectations:

- The helper should sanitize filename components.
- Should overwrite support use a standard `force` flag everywhere, or only when
  an app already exposes one?
- Should directory outputs be included now, or should this Module start with
  `.tar.zst` bytes only?

## Candidate 3: Dual Runner Drift

Files:

- `src/biomodals/app/bioinfo/rosetta_app.py`
- `src/biomodals/app/score/af3score_app.py`
- `src/biomodals/app/design/ppiflow_app.py`
- `src/biomodals/workflow/ppiflow_workflow.py`

Observed friction:

- Some apps have both:
  - a volume-backed runner for direct local entrypoint usage
  - a bytes-backed runner for workflow usage
- The two Interfaces differ even when the Implementation repeats staging,
  validation, command construction, command execution, packaging, and result
  parsing.
- Workflow code should not add duplicate app runners only to return tarball
  bytes when the app already has a volume-backed runner.
- The app wrappers are public Interfaces for workflows, so changing them needs
  backwards compatibility.

Updated target shape from PR #23 review:

- Do not start with a generic cross-tool execution framework.
- Prefer reusing existing volume-backed app runners. Workflow code should stage
  inputs into the app's Modal output volume, call the existing deployed
  function, infer relevant output paths from that app volume, and copy those
  artifacts into the workflow volume.
- `PPIFlow.ppiflow_run(...)` should be reused rather than maintaining
  `ppiflow_run_bytes(...)`.
- `AF3Score.af3score_prepare(...)`, `af3score_run(...)`, and
  `af3score_postprocess(...)` should be reused rather than maintaining
  `af3score_run_bytes(...)`.
- `Rosetta.run_rosetta_batch_bytes(...)` can remain for now because the generic
  Rosetta volume scheduler has a queue/run-id interface that is not yet a clean
  workflow fit. Keep its command normalization shared and avoid broad app churn.
- `DockQ.run_dockq_batch(...)` can remain tarball-returning because DockQ is
  short-lived and does not currently have a persistent output volume runner.

First implementation slice:

1. Remove duplicated PPIFlow and AF3Score byte runners.
2. Teach the workflow scheduler to stage inputs into app output volumes and
   copy relevant output files back into the workflow output volume.
3. Keep public app entrypoint names stable.
4. Keep Rosetta and DockQ workflow interfaces unchanged until their existing
   app runners provide an equally reviewable volume-backed path.

Verification:

- `uv run biomodals help rosetta`
- `uv run biomodals help ppiflow`
- `uv run biomodals help af3score`
- `modal run src/biomodals/workflow/ppiflow_workflow.py --help`
- `prek run --files <changed files>`

Open questions:

- Should normalized job descriptions be tool-local dataclasses or shared helper
  dataclasses?
- Should app output volume names be explicit CLI flags, or are environment
  variables such as `PPIFLOW_OUTPUT_VOLUME_NAME` enough?
- Should workflow-only functions be discoverable in `biomodals help`, or just
  listed as Modal functions as they are today?

## Candidate 4: Volume Run State

Files:

- `src/biomodals/app/design/boltzgen_app.py`
- `src/biomodals/app/score/af3score_app.py`
- `src/biomodals/app/bioinfo/rosetta_app.py`
- `src/biomodals/app/bioinfo/gromacs_app.py`
- `src/biomodals/workflow/ppiflow_workflow.py`

Observed friction:

- Long-running or batch apps each handle Modal volume state differently:
  - run names
  - existence checks
  - locks
  - resume/salvage behavior
  - completion markers
  - upload layout
  - download layout
  - cleanup
- Some differences are tool-specific and should remain behind Adapters.
- Some behavior is shared Modal volume mechanics and has poor Locality today.
- AF3Score has a run lock implemented as a directory and notes this may not be
  atomic.
- BoltzGen has salvage/completion rules around run directories.
- Rosetta creates one queue and one run directory per `uuid4()` run.

Updated target shape from PR #23 review:

- Replace the narrow `volume_run.py` helper with
  `src/biomodals/helper/orchestrator.py`.
- Generic path helpers should only encode broadly useful names:
  - `mount_root`
  - `run_root`
  - `input_dir`
  - `output_dir`
- Tool-specific paths such as AF3Score `prepare`, `failed_records`, and metrics
  CSV paths should be passed as keyword path names or owned by the app Adapter.
  If a custom key collides with a default key, raise an error.
- Do not keep a low-value generic completion predicate unless there are several
  consumers. Prefer tool-local completion checks or `fd-find`/`find_with_fd`
  when locating expected output files across a tree.
- `orchestrator.py` is the right home for future helpers around queue names,
  lock lifecycle, queue population, and DAG/task graph construction.
- Tool-specific completion rules should be Adapters. One Adapter per tool is
  enough once there are two real consumers.
- Avoid replacing every app's volume behavior at once. This is higher risk than
  local output materialization.

First implementation slice:

1. Document the current run-state policies in tests or helper docstrings before
   changing behavior.
2. Extract only pure path/policy helpers first into
   `src/biomodals/helper/orchestrator.py`.
3. Apply to one target app, preferably AF3Score or BoltzGen:
   - AF3Score has clear lock and path functions.
   - BoltzGen has clear completion and salvage concepts.
4. Use fake filesystem tests for completion predicates and path policies.

Verification:

- App-specific `biomodals help` smoke test.
- `prek run --files <changed files>`
- Do not claim lock atomicity improved unless the Implementation actually uses a
  Modal-supported atomic primitive or the behavior was tested on Modal.

Open questions:

- Should active locks use Modal Dict/Queue instead of volume directories?
- Which apps should have resumability as a public Interface versus an internal
  Implementation detail?
- Should successful runs always write a standard marker file, or preserve each
  tool's natural completion artifact?

## Cross-Cutting Testing Plan

There is no committed test suite. Start with small local tests that do not
require Modal cloud execution or model artifacts.

Preferred local-only tests:

- App catalog parsing and app resolution.
- Google-style docstring flag table generation.
- Local output path and overwrite policy.
- Rosetta command/job normalization.
- AF3Score/BoltzGen run-state path helpers and completion predicates.

Do not include these tests in the final PR unless the user changes that
direction.

Test framework choice:

- If avoiding dependency churn, use `unittest` under `tests/`.
- If adding `pytest`, update project metadata intentionally and keep the first
  test set small.

## Non-Goals For The First Pass

- Do not run Modal jobs as part of the initial refactor unless explicitly asked.
- Do not migrate all apps to new helpers in one PR.
- Do not introduce a generic execution framework across all tools before one
  tool-local Module proves the shape.
- Do not change public app entrypoint names or deployed workflow function names
  without a compatibility plan.
- Do not add large example data, generated archives, or Modal result
  directories.

## Current PR Scope

Keep the final PR reviewable by limiting app changes to the currently touched
workflow-facing subset:

01. `src/biomodals/helper/io.py`
02. `src/biomodals/helper/orchestrator.py`
03. `src/biomodals/app/fold/alphafold3_app.py`
04. `src/biomodals/app/fold/flowpacker_app.py`
05. `src/biomodals/app/fold/protenix_app.py`
06. `src/biomodals/app/score/dockq_app.py`
07. `src/biomodals/app/design/ppiflow_app.py`
08. `src/biomodals/app/score/af3score_app.py`
09. `src/biomodals/app/bioinfo/rosetta_app.py`, only for the already-reviewed
    deterministic byte-job fallback naming fix.
10. `src/biomodals/workflow/ppiflow_workflow.py`

Avoid migrating every app to the new helpers in this PR.
