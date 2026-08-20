---
name: biomodals-workflow-development
description: Use when creating, editing, or reviewing Biomodals workflow code under src/biomodals/workflow/, shared workflow schemas under src/biomodals/schema/, workflow-compatible app functions, or workflow CLI/tests, including execution-kernel Task scheduling, ShortMD-style DAG construction, orchestrator composition, app dependency inclusion, execution artifacts, and Modal volume handling.
---

# Biomodals Workflow Development

Use this skill for Biomodals workflow definitions, deployment composition,
workflow schemas, and workflow-compatible app integration points. Apps and
workflows use the same `biomodals.execution` graph and runtime; workflow code
does not own a second orchestration core.

## Core Workflow

Before making non-trivial workflow changes, read the
[maintained standards](references/workflow-development.md) and
[repo coordination notes](../../../docs/agents/workflow-development.md).

When a workflow image, coordinator image, or included app image runs below the
repository's Python minimum, also read
[Python runtime compatibility](../../../docs/agents/python-runtime-compatibility.md).

Use [ShortMD](../../../src/biomodals/workflow/shortmd_workflow.py) as the
primary end-to-end example for app-composed workflows. Use
[RFD-LigandMPNN](../../../src/biomodals/workflow/rfd_ligandmpnn_workflow.py)
for workflows that fan out one app's volume-backed outputs into another app's
workflow-compatible remote function. Use
[PPIFlow](../../../src/biomodals/workflow/ppiflow_workflow.py) for
candidate-manifest joins, retained-candidate filtering, candidate-wide remote
Tasks, focused task-image runtimes, and PPIFlow-specific stage wiring.

## Working Rules

- Keep `biomodals.schema` pure Pydantic and free of Modal imports.
- Compose workflow deployments with the shared Modal execution host and
  included dependency apps. Provider calls name operations exactly; the Modal
  integration resolves them against the pinned containing deployment.
- Build a kernel-owned `ExecutionDefinition`. Prefer app-backed Execution Nodes
  for existing app operations; add workflow-owned Nodes only for adapters,
  summaries, selectors, and workflow-specific file-management glue.
- Execution Nodes prepare provider operations and adapt results, but never
  submit directly.
- Keep hydrated Modal objects out of workflow Nodes. Explicit development runs
  may supply a function-name-to-handle map at the coordinator boundary.
- Keep generic execution state and artifact records in `biomodals.execution`;
  keep publication meaning and validation workload-owned. Do not add workflow
  attempt tables, replacement-call loops, or a second task queue.
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
- Keep the kernel interface deep. Add shared behavior for repeated app and
  workflow needs, not one-off workflow conveniences.

## Verification

For workflow changes, run focused pytest coverage first, then `prek run --files
<changed files>` when practical. For CLI or discovery changes, also smoke test
`uv run biomodals workflow list` and the affected `biomodals workflow help/run`
path.
