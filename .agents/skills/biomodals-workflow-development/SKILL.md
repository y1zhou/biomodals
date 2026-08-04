---
name: biomodals-workflow-development
description: Use when creating, editing, or reviewing Biomodals workflow code under src/biomodals/workflow/, shared workflow schemas under src/biomodals/schema/, workflow-compatible app functions, or workflow CLI/tests, including execution-kernel Task scheduling, ShortMD-style DAG construction, orchestrator composition, app dependency inclusion, workflow artifacts, and Modal volume handling.
---

# Biomodals Workflow Development

Use this skill for Biomodals workflow scripts, the reusable workflow runtime,
workflow schemas, and workflow-compatible app integration points.

## Core Workflow

Before making non-trivial workflow changes, read
`references/workflow-development.md` for the maintained standards and
`docs/agents/workflow-development.md` for repo-level coordination notes.

Use `src/biomodals/workflow/shortmd_workflow.py` as the primary end-to-end
example for app-composed workflows. Use
`src/biomodals/workflow/rfd_ligandmpnn_workflow.py` as the reference for
workflows that fan out one app's volume-backed outputs into another app's
workflow-compatible remote function. Use
`src/biomodals/workflow/ppiflow_workflow.py` as the reference for
candidate-manifest joins, retained-candidate filtering, candidate-wide remote
stage coordinators, and PPIFlow-specific stage wiring.

## Working Rules

- Keep `biomodals.schema` pure Pydantic and free of Modal imports.
- Compose workflow apps with the shared orchestrator and included dependency
  apps. Remote calls name functions exactly; the kernel resolves those names
  against the pinned containing deployment.
- Prefer `AppBackedNode` for nodes that primarily call app functions.
  Add `WorkflowNativeNode` only for adapters, summaries, selectors, and
  workflow-specific file-management glue.
- `REMOTE` nodes prepare `RemoteNodeCall` values through
  `prepare_remote(context)` and adapt raw results with
  `process_remote_result(...)` only when needed. They never submit directly.
- Keep hydrated Modal objects out of workflow Nodes. Explicit development runs
  may supply a function-name-to-handle map at the coordinator boundary.
- Keep generic execution state in `biomodals.execution` and scientific
  publications in workflow artifacts. Do not add workflow attempt tables,
  replacement-call loops, or a second task queue.
- Import app-owned volume handles, volume names, and mountpoints from source app
  modules, and reload relevant volumes before reading mutations committed by
  another container.
- When staging workflow-derived files for downstream apps, do not use full
  artifact/provenance strings as local filenames. Derive short deterministic
  names from candidate ids or content hashes because pipeline-derived names can
  exceed filesystem component limits.
- User-facing workflow local entrypoints should accept `dry_run: bool = False`.
  When set, build the workflow, call `print_workflow_dag(workflow.validate())`,
  and return before constructing or submitting the orchestrator. The workflow
  CLI forwards `biomodals workflow run --dry-run` to this entrypoint flag.
- When adding or changing workflow-compatible app functions, use RFdiffusion and
  LigandMPNN as the current app-side reference implementations and coordinate
  with the app-development skill.
- Keep the core runtime slim. Add public orchestrator/runtime API only for clear
  missing capabilities, not one-off workflow conveniences.

## Verification

For workflow changes, run focused pytest coverage first, then `prek run --files
<changed files>` when practical. For CLI or discovery changes, also smoke test
`uv run biomodals workflow list` and the affected `biomodals workflow help/run`
path.
