---
name: biomodals-workflow-development
description: Use when creating, editing, or reviewing Biomodals workflow code under src/biomodals/workflow/, shared workflow schemas under src/biomodals/schema/, workflow-compatible app functions, or workflow CLI/tests, including ShortMD-style DAG construction, orchestrator composition, app dependency inclusion, workflow artifacts, and Modal volume handling.
---

# Biomodals Workflow Development

Use this skill for Biomodals workflow scripts, the reusable workflow runtime,
workflow schemas, and workflow-compatible app integration points.

## Core Workflow

Before making non-trivial workflow changes, read
`references/workflow-development.md` for the maintained standards.

Use `src/biomodals/workflow/shortmd_workflow.py` as the primary end-to-end
example for app-composed workflows. Ignore
`src/biomodals/workflow/ppiflow_workflow.py` as a reference pattern for now; it
is expected to be refactored.

## Working Rules

- Keep `biomodals.schema` pure Pydantic and free of Modal imports.
- Compose workflow apps with `from biomodals.workflow.core import orchestrator`
  and `modal.App(...).include(orchestrator.app)`.
- Declare app dependencies on `AppConfig.depends_on_apps`, mirror them into
  `CONF.tags["depends_on"]` for Modal UI metadata, and compose them with
  `include_dependency_apps(app, CONF.depends_on_apps)`.
- Prefer included-app Modal handles over deployed-app lookup strings. Do not add
  `modal.Function.from_name(...)` to new workflow code when the dependency app
  can be included.
- Prefer `AppBackedNode` for nodes that primarily call app functions.
  Add `WorkflowNativeNode` only for adapters, summaries, selectors, and
  workflow-specific file-management glue.
- Store hydrated Modal functions/classes in a small `*ModalNamespace` dataclass
  typed as `modal.Function` or `modal.Cls`, and exclude that namespace from DAG
  hashing with `repr=False`, `compare=False`, and `metadata={"dag_hash": False}`.
- Define workflow-specific remote file-management functions as top-level
  `@app.function`s in the workflow module and put their hydrated handles in the
  workflow's `*ModalNamespace`. Do not make ordinary node methods Modal
  functions.
- Import app-owned volume handles, volume names, and mountpoints from source app
  modules. Avoid duplicating volume strings in workflow scripts.
- Use `volume_path_from_mount_path(...)` to convert mounted app paths into
  `VolumePath` workflow storage references.
- Keep the core runtime slim. Add public orchestrator/runtime API only for clear
  missing capabilities, not one-off workflow conveniences.

## Verification

For workflow changes, run focused pytest coverage first, then `prek run --files
<changed files>` when practical. For CLI or discovery changes, also smoke test
`uv run biomodals workflow list` and the affected `biomodals workflow help/run`
path.
