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
[maintained standards](references/workflow-development.md) in full.
When changing app-owned functions, also use the
[app-development skill](../biomodals-app-development/SKILL.md).

When a workflow image, coordinator image, or included app image runs below the
repository's Python minimum, also read
[Python runtime compatibility](../../../docs/agents/python-runtime-compatibility.md).

Use [ShortMD](../../../src/biomodals/workflow/shortmd_workflow.py) as the
primary end-to-end example for app-composed workflows. Use
[RFD-LigandMPNN](../../../src/biomodals/workflow/rfd_ligandmpnn_workflow.py)
for workflows that fan out one app's volume-backed outputs into another app's
workflow-compatible remote function. Use
[PPIFlow](../../../src/biomodals/workflow/ppiflow/workflow.py) for
candidate-manifest joins, retained-candidate filtering, candidate-wide remote
Tasks, focused task-image runtimes, and PPIFlow-specific stage wiring.

## Verification

For workflow changes, run focused pytest coverage first, then `prek run --files
<changed files>` when practical. For CLI or discovery changes, also smoke test
`uv run biomodals workflow list` and the affected `biomodals workflow help/run`
path.
