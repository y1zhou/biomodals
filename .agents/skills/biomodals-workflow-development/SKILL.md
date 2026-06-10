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
example for app-composed workflows. Use
`src/biomodals/workflow/rfd_ligandmpnn_workflow.py` as the reference for
workflows that fan out one app's volume-backed outputs into another app's
workflow-compatible remote function. Ignore
`src/biomodals/workflow/ppiflow_workflow.py` as a reference pattern for now
because it is expected to be refactored.

## Working Rules

- Keep `biomodals.schema` pure Pydantic and free of Modal imports.
- Compose workflow apps with `from biomodals.workflow.core import orchestrator`
  and `modal.App(...).include(orchestrator.app)`.
- Declare app dependencies on `AppConfig.depends_on_apps`, mirror them into
  `CONF.tags["depends_on"]` for Modal UI metadata using a Modal-valid tag value
  such as `"-".join(DEPENDENCY_APPS)`, and compose them with
  `include_dependency_apps(app, CONF.depends_on_apps)`.
- Prefer included-app Modal handles over deployed-app lookup strings. Do not add
  `modal.Function.from_name(...)` to new workflow code when the dependency app
  can be included.
- Prefer `AppBackedNode` for nodes that primarily call app functions.
  Add `WorkflowNativeNode` only for adapters, summaries, selectors, and
  workflow-specific file-management glue.
- Every `REMOTE` node must define a node-level `submit_remote(context)` hook
  that returns `RemoteNodeSubmission` for the actual Modal `FunctionCall`.
  `process_remote_result(result, metadata)` is part of the node contract and
  defaults to `AppRunResult.model_validate(result)`. Override it when the raw
  remote result must be adapted before artifact materialization. `AppBackedNode`
  and `REMOTE` `WorkflowNativeNode` implementations inherit a default `run()`
  that submits the remote call, waits for `.get()`, and processes the result.
- Store hydrated Modal functions/classes in a small `*ModalNamespace` dataclass
  typed as `modal.Function` or `modal.Cls`, and exclude that namespace from DAG
  hashing with `repr=False`, `compare=False`, and `metadata={"dag_hash": False}`.
- Define workflow-specific remote file-management functions as top-level
  `@app.function`s in the workflow module and put their hydrated handles in the
  workflow's `*ModalNamespace`. Do not make ordinary node methods Modal
  functions.
- Import app-owned volume handles, volume names, and mountpoints from source app
  modules. Avoid duplicating volume strings in workflow scripts.
- Use `volume_path_from_mount_path(...)` from `biomodals.helper.app_run` to
  convert mounted app paths into `VolumePath` workflow storage references.
- Materialize inline workflow outputs once under
  `nodes/<node-id>/attempts/<attempt-id>/<artifact-id>/` and store materialized
  `VolumePath` app-result JSON in the ledger; do not persist base64
  `InlineBytes` payloads in SQLite.
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
