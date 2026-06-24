# PPIFlow Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `src/biomodals/workflow/ppiflow_workflow.py` as a ShortMD-style Biomodals workflow that follows upstream PPIFlow stage ordering, preserves full candidate sets, and uses included Biomodals app functions.

**Architecture:** Build a static stage-level DAG from upstream `task.yaml` and `steps.yaml`. `ppiflow_workflow.py` remains the public CLI/catalog workflow entrypoint containing `CONF`, namespace hydration, node classes, DAG construction, and local entrypoint wiring. PPIFlow-local manifest, table, staging, and coordinator helpers live under `src/biomodals/workflow/ppiflow/`. Node classes stay in `ppiflow_workflow.py`. Candidate manifests are Parquet `ArtifactKind.TABLE` workflow artifacts, one row per candidate with nested file records. Candidate-wide stages run through bounded remote coordinators and return `PARTIAL` for mixed candidate success.

Modal decorators, app registration, and app-bound remote helper functions stay in `ppiflow_workflow.py`. Helper modules under `workflow/ppiflow/` should avoid hidden Modal registration side effects and keep logic directly unit-testable.

**Tech Stack:** Python 3.13 workflow module, Modal, Biomodals workflow runtime, Pydantic app result schemas, `orjson`, `polars`, `pytest`, `prek`.

**Implementation Order:** Start with Phase 1 helper/test split, then build the
manifest foundation on stable file boundaries.

**Commit Strategy:** Commit by phase. Each phase should pass its focused tests
and `prek run --files <changed files>` before committing. Do not push without an
explicit user request.

---

## Phase 0: Preserve Current Contract

**Files:**

- Modify: `tests/workflow/test_ppiflow_workflow.py`

**Already Done:**

- [x] Assert `CONF.depends_on_apps` contains `ppiflow`, `rosetta`, `flowpacker`, `ligandmpnn`, `dockq`, `af3score`, and `alphafold3`, and that `CONF.tags["depends_on"]` mirrors tuple order.
- [x] Add a DAG-shape test for:
  `PPIFlowStep -> MPNNStep_stage1 -> FlowpackerStep_stage1 -> AF3scoreStep_stage1 -> FilterStep_stage1 -> RosettaFixStep -> FixedPositions -> PartialStep -> MPNNStep_stage2 -> FlowpackerStep_stage2 -> AF3scoreStep_stage2 -> FilterStep_stage2 -> ReFoldStep -> DockQStep -> RosettaRelaxStep -> RankStep -> ReportStep`.
- [x] Assert DAG vertices use specific node classes instead of the old generic `PPIFlowWorkflowNode`.
- [x] Add fake-Modal adapter tests for PPIFlow, LigandMPNN, FlowPacker, AF3Score, DockQ, Rosetta, and AlphaFold3.
- [x] Preserve stage-only execution behavior while requiring existing upstream artifacts for stage-2-only runs.
- [x] Update local input staging to cover initial PPIFlow inputs and preserve mounted paths.
- [x] Implement workflow-native filter parsing, Rosetta `residue_energy.csv` to `fixed_positions.csv`, score-aware ranking, and Markdown/HTML report generation.
- [x] Keep PPIFlow-specific Rosetta interface-energy analysis workflow-owned instead of adding it to the generic Rosetta app contract.
- [x] Keep PPIFlow Rosetta job manifests, script/flags selection, queue setup, queue cleanup, expected-output discovery, and per-candidate Rosetta status in the workflow while leaving the Rosetta app as a generic command worker.

## Phase 1: Split Helper Modules And Tests

**Scope:** Behavior-preserving extraction only. Do not introduce candidate
manifest semantics, coordinator behavior, status changes, or DAG changes in this
phase.

**Files:**

- Add: `src/biomodals/workflow/ppiflow/__init__.py`
- Add: `src/biomodals/workflow/ppiflow/manifests.py`
- Add: `src/biomodals/workflow/ppiflow/tables.py`
- Add: `src/biomodals/workflow/ppiflow/staging.py`
- Add: `src/biomodals/workflow/ppiflow/coordinators.py`
- Add: `tests/workflow/ppiflow/test_manifests.py`
- Add: `tests/workflow/ppiflow/test_tables.py`
- Add: `tests/workflow/ppiflow/test_staging.py`
- Add: `tests/workflow/ppiflow/test_coordinators.py`
- Modify: `src/biomodals/workflow/ppiflow_workflow.py`
- Modify: `tests/workflow/test_ppiflow_workflow.py`

**Tasks:**

- [x] Keep `ppiflow_workflow.py` as the public workflow entrypoint with `CONF`, namespace hydration, node classes, DAG construction, and local entrypoint wiring.
- [x] Do not move node classes into the helper submodule in this refactor.
- [x] Keep `src/biomodals/workflow/ppiflow/__init__.py` minimal or empty so the non-underscore package name does not imply a public API.
- [x] Move candidate manifest schema/constants/helpers into `ppiflow/manifests.py`.
- [x] Move sequence-table, score-table, filter-audit, attrition, ranking, and report table helpers into `ppiflow/tables.py`.
- [x] Move app-volume/archive/structure staging helpers into `ppiflow/staging.py`.
- [x] Move reusable candidate-wide remote coordinator logic into `ppiflow/coordinators.py`.
- [x] Keep Modal-decorated remote helper functions and app registration in
  `ppiflow_workflow.py`; call pure helper functions from those wrappers instead
  of decorating functions inside the helper submodule.
- [x] Import the `ppiflow` helper submodule only from `ppiflow_workflow.py` and tests.
- [x] Move pure manifest, table, staging, and coordinator helper tests out of `tests/workflow/test_ppiflow_workflow.py`.
- [x] Keep DAG, node contract, and hydrated namespace integration tests in `test_ppiflow_workflow.py`.
- [x] Verify no runtime behavior, DAG shape, app output contracts, or node status
  handling changes in this phase.

## Phase 2: Candidate Manifest Foundation

**Files:**

- Modify: `src/biomodals/workflow/ppiflow/manifests.py`
- Modify: `src/biomodals/workflow/ppiflow/staging.py`
- Modify: `src/biomodals/workflow/ppiflow_workflow.py`
- Modify: `tests/workflow/ppiflow/test_manifests.py`
- Modify: `tests/workflow/ppiflow/test_staging.py`
- Modify: `tests/workflow/test_ppiflow_workflow.py`

**Tasks:**

- [x] Keep candidate manifest schema/constants/helpers workflow-local to PPIFlow; do not add shared `biomodals.schema` models yet.
- [x] Store manifests as Parquet `ArtifactKind.TABLE` workflow artifacts in the workflow run volume.
- [x] Use Polars `read_parquet`/`write_parquet` for manifests.
- [x] Store one Parquet row per candidate with a nested `files` list of structs.
- [x] Include `candidate_id`, `parent_candidate_id`, `stage_name`, `stage_role`, `operation_mode`, `candidate_status`, `source_artifact_id`, `source_path`, `derived_path`, and summary/error fields.
- [x] Include workflow-relative artifact paths, app-volume names, app-volume paths, file roles, media types, size metadata when available, and expected-file flags in nested file records.
- [x] Add deterministic candidate-id helpers: initial candidates hash producing stage, source artifact id/path, and normalized basename.
- [x] Add derived candidate-id helpers: derived candidates hash parent candidate id, stage name, operation mode, and derived basename.
- [x] Keep synthetic stage-2 convenience candidate ids sequential (`stage2_input_000001`, etc.) while preserving source-path provenance.
- [x] Ensure generated candidate ids and manifests do not enter workflow DAG hash payloads unless user-facing node configuration changes.
- [x] Write node metadata only as summary counts and manifest artifact ids, not as the only manifest copy.
- [x] Add explicit `Stage2Input` manifest loading and validation.
- [x] Add `Stage2Input` path normalization that scans a user-provided structure location and emits a minimal candidate manifest with synthetic candidate ids.
- [x] Change `ExistingStructuresNode` so stage-2-only runs return both the configured structure artifact and the normalized candidate manifest artifact.
- [x] Add strict candidate-id join helpers that fail on missing required candidates unless explicit inspection/debug config enables missing-candidate tolerance.
- [x] Before skipping a completed candidate on retry, verify expected output files recorded in the manifest are still available.
- [x] Route expected-file checks through workflow-volume paths for materialized artifacts and app-volume paths plus `volume_name` for app-owned outputs.
- [x] Add tests for deterministic candidate ids across reruns.
- [x] Add tests that candidate ids change only when provenance-relevant inputs change.
- [x] Add tests that candidate-id helper internals and generated manifests do not affect DAG hash payloads.
- [x] Add tests that manifest outputs are materialized as Parquet `ArtifactKind.TABLE` artifacts.
- [x] Add tests that downstream nodes consume manifest artifacts rather than node metadata.
- [x] Add tests that stage-2-only runs support explicit manifest loading and path-to-manifest normalization.
- [x] Add tests that retry-skip verifies expected candidate output files and does not trust manifest completion rows alone.
- [x] Add tests that strict candidate joins fail on missing required candidates and only drop candidates when explicitly configured.

## Phase 3: Table And Report Helpers

**Files:**

- Modify: `src/biomodals/workflow/ppiflow/tables.py`
- Modify: `src/biomodals/workflow/ppiflow_workflow.py`
- Modify: `tests/workflow/ppiflow/test_tables.py`
- Modify: `tests/workflow/test_ppiflow_workflow.py`

**Tasks:**

- [x] Keep upstream-facing score, sequence, and report tables in their existing formats unless separately changed.
- [x] Add `mpnn_seqs.csv` extraction from LigandMPNN-designed outputs with `candidate_id` and parent provenance for stage 1 and stage 2.
- [x] Add upstream-equivalent FASTA collection into `mpnn_seqs.csv`.
- [x] Add ReFold metrics extraction from AlphaFold3 confidence/ranking JSONs into a candidate-keyed CSV.
- [x] Add score-table status helpers that classify requested candidate or pair counts as `SUCCEEDED`, `PARTIAL`, or `FAILED` based on usable score rows while retaining diagnostic rows and logs.
- [x] Change filter helpers to join structures, scores, and incoming candidate manifests by `candidate_id`.
- [x] Make filters emit a retained-candidate manifest for downstream nodes.
- [x] Add a filter audit table that records every input candidate, pass/fail status, evaluated filter metrics, and rejection reason.
- [x] Aggregate retained, rejected, failed, and skipped counts from manifests and audit tables into a candidate-attrition table.
- [x] Make ranking include only retained candidates with complete required structures and score rows.
- [x] Keep rejected, partial, failed, and skipped candidates out of `ranked_designs.csv` and visible through attrition/report tables.
- [x] Render Markdown and HTML reports from materialized tables/manifests without adding a report app.
- [x] Wire stage 1 and stage 2 `mpnn_seqs.csv` outputs into Rank/Report through candidate-id joins.
- [x] Wire filter audit tables and candidate manifests into Report for candidate attrition sections.
- [x] Add tests for sequence-table extraction.
- [x] Add tests for ReFold metrics extraction from AlphaFold3 archive JSONs.
- [x] Add tests for score-table `SUCCEEDED`/`PARTIAL`/`FAILED` classification.
- [x] Add tests for filter retained manifests and audit tables.
- [x] Add tests that `ranked_designs.csv` excludes rejected/partial/failed candidates while reports include attrition counts.
- [x] Add tests that `ReportNode` consumes manifests and audit tables without feeding rejected candidates back into downstream scientific nodes.

## Phase 4: Staging And App-Volume Helpers

**Files:**

- Modify: `src/biomodals/workflow/ppiflow/staging.py`
- Modify: `src/biomodals/workflow/ppiflow_workflow.py`
- Modify: `tests/workflow/ppiflow/test_staging.py`

**Tasks:**

- [x] Keep volume helpers that convert `WorkflowArtifact.storage` to mount paths for known app output volumes.
- [x] Keep archive extraction helpers for app functions that return `.tar.zst` bytes or archives in a volume path.
- [x] Update structure-selection helpers to return candidate-keyed selections instead of unkeyed `(name, bytes)` lists.
- [x] Keep candidate narrowing available only through explicit stage configuration such as `structure_index`, `max_structures`, or a future candidate selector.
- [x] Fold upstream before-partial structure selection into `PPIFlowPartialNode` staging.
- [x] Add partial sample directory discovery for PPIFlow stage 2.
- [x] Add candidate-wide DockQ model/reference pair preparation by candidate id.
- [x] Add Rosetta staging helpers that create PPIFlow-owned Rosetta job manifests with queue entries, command parameters, candidate ids, expected output paths, worker logs, and candidate outcomes.
- [x] Add tests for candidate-keyed archive extraction and app-volume staging.
- [x] Add tests for before-partial structure selection inside Partial staging.
- [x] Add tests for candidate-wide DockQ pair preparation.
- [x] Add tests for Rosetta job manifest staging.

## Phase 5: Candidate-Wide Remote Coordinators

**Files:**

- Modify: `src/biomodals/workflow/ppiflow/coordinators.py`
- Modify: `src/biomodals/workflow/ppiflow_workflow.py`
- Modify: `tests/workflow/ppiflow/test_coordinators.py`
- Modify: `tests/workflow/test_ppiflow_workflow.py`

**Tasks:**

- [x] Add reusable candidate-wide coordinator logic that loads the incoming manifest, verifies reusable completed candidates, submits missing candidates, and writes updated manifests.
- [x] Add bounded child-call concurrency with shared `candidate_concurrency` default of 4 and per-stage overrides.
- [x] Parse shared PPIFlow `candidate_concurrency` from task/steps YAML and copy the resolved value into candidate-wide node configs during DAG construction.
- [x] Do not add a workflow runtime or orchestrator flag for PPIFlow candidate concurrency.
- [x] Normalize candidate-wide node status: all requested candidates succeeded -> `SUCCEEDED`; some succeeded and some failed -> `PARTIAL`; none succeeded -> `FAILED`.
- [x] Preserve successful outputs, failed candidate records, diagnostic logs, and candidate manifests for `PARTIAL` and `FAILED` results.
- [x] Ensure `PARTIAL` does not unblock downstream nodes.
- [x] Make each remote stage coordinator consult its manifest before submitting child app calls.
- [x] Make completed manifest rows reusable only after expected-output availability checks pass.
- [x] Ensure every candidate-wide node returns the current candidate manifest as an output artifact alongside structures, scores, logs, or reports.
- [x] Move candidate-wide `AF3ScoreNode`, `RosettaFixNode`, `RosettaRelaxNode`, `ReFoldNode`, `LigandMPNNNode`, and `PPIFlowPartialNode` coordination into remote workflow-node execution.
- [x] Keep one recoverable Modal call id per candidate-wide stage.
- [x] Implement stage-specific thin Modal remote wrappers in
  `ppiflow_workflow.py` over shared candidate-wide coordinator helpers instead
  of one generic stage-parameterized Modal function.
- [x] Mount only the workflow/app volumes required by each stage-specific remote
  wrapper, and pass an explicit volume map into shared helper functions.
- [x] Give each stage-specific wrapper a clear function name for Modal logs and
  future stage-specific resource tuning.
- [x] Start stage-specific wrappers with the current workflow resource defaults.
- [x] Add `# TODO:` comments at likely wrapper tuning points for future CPU,
  memory, timeout, GPU, or mount-scope adjustments once real telemetry exists.
- [x] Add tests or static assertions that stage wrappers declare only the volume
  mounts their stage needs.
- [x] Add tests that candidate-wide stage coordinators respect configured child app-call concurrency limits.
- [x] Add tests that retries reuse manifests and skip only candidates with available expected outputs.
- [x] Add tests that mixed candidate success returns `PARTIAL` and blocks downstream execution.

## Phase 6: Stage-Specific Node Refactors

**Files:**

- Modify: `src/biomodals/workflow/ppiflow_workflow.py`
- Modify: `src/biomodals/app/score/dockq_app.py`
- Modify: `tests/workflow/test_ppiflow_workflow.py`
- Modify: `tests/app/test_dockq_workflow_contract.py`

**Tasks:**

- [x] Change `LigandMPNNNode` to process all selected structures by default instead of selecting one structure.
- [x] Represent upstream binder MPNN and AbMPNN as configuration modes on the same `LigandMPNNNode`.
- [x] Add a separate AbMPNN node only if AbMPNN later needs a distinct app function or output contract.
- [x] Change `LigandMPNNNode` to return designed structure artifacts and a PPIFlow-owned `mpnn_seqs.csv` table for every successful candidate.
- [x] Change `PPIFlowPartialNode` to process all selected structures by default.
- [x] Keep before-partial structure selection inside `PPIFlowPartialNode`.
- [x] Change `ReFoldNode` to process all selected structures by default.
- [x] Change `ReFoldNode` to return refolded structure artifacts and a `ReFold Quality Metrics` score artifact for every successful candidate.
- [x] Change `AF3ScoreNode` to compare requested input count against `metrics_rows`, `processed`, and `failed` from AF3Score postprocess.
- [x] Return `PARTIAL` from `AF3ScoreNode` for mixed success and `FAILED` when no usable metrics are produced.
- [x] Update `dockq_app.run_dockq_workflow` so DockQ archives with failed pairs return `PARTIAL` or `FAILED` instead of unconditional `SUCCEEDED`.
- [x] Update `DockQNode` to preserve DockQ diagnostic CSVs/logs for `PARTIAL` and `FAILED` results.
- [x] Make `RosettaFixNode` and `RosettaRelaxNode` remote coordinators own Modal queue creation, worker submission, queue cleanup, expected-output verification, and Rosetta job manifest updates through the hydrated namespace.
- [x] Ensure RosettaFix/RosettaRelax diagnostic logs and failed candidate records are materialized for `PARTIAL` and `FAILED` results.
- [x] Change `FilterStructuresNode` to expose retained structures, filtered scores, retained candidate manifest, and filter audit table.
- [x] Keep `ReportNode` workflow-native and consume candidate manifests/filter audit tables for attrition reporting.
- [x] Add node-level tests for each status and output contract above.

## Phase 7: DAG Wiring And Upstream Equivalence

**Files:**

- Modify: `src/biomodals/workflow/ppiflow_workflow.py`
- Modify: `tests/workflow/test_ppiflow_workflow.py`

**Tasks:**

- [x] Build stage 1 exactly as upstream: PPIFlow, binder MPNN or AbMPNN, collect `mpnn_pdbs/mpnn_seqs.csv`, FlowPacker, AF3Score, Filter.
- [x] Build stage 2 exactly as upstream: RosettaFix, fixed positions CSV, Partial with internal before-partial structure selection, binder MPNN or AbMPNN, FlowPacker, AF3Score, Filter, ReFold, DockQ, RosettaRelax, Rank, Report.
- [x] Preserve the full candidate set through stage 1 and stage 2 fan-out while keeping each upstream step represented as a static workflow DAG node.
- [x] Feed ReFold structure outputs into DockQ/RosettaRelax.
- [x] Feed ReFold quality metrics into Rank/Report through candidate-id joins.
- [x] Feed AF3Score, DockQ, RosettaRelax, ReFold quality metrics, sequence tables, and retained manifests into Rank/Report through strict candidate-id joins.
- [x] Ensure downstream scientific nodes consume only retained manifests.
- [x] Ensure rejected candidates remain available only to audit/report paths.
- [x] Keep stage-2-only runs compatible with explicit manifest input and convenience path-to-manifest normalization.
- [x] Keep old in-progress PPIFlow workflow ledger migration out of scope; document rerun with `force` or explicit `Stage2Input`.
- [x] Add DAG tests for candidate manifest edges.
- [x] Add tests that stage-2-only input manifests feed RosettaFix and downstream stage 2 nodes.
- [x] Add tests that rejected candidates do not flow into Partial, MPNN stage 2, ReFold, DockQ, RosettaRelax, or Rank.

## Phase 8: Verification And Documentation

**Files:**

- Modify: `docs/superpowers/plans/2026-06-01-ppiflow-workflow.md`
- Modify: `tests/workflow/test_ppiflow_workflow.py`
- Modify: `tests/workflow/ppiflow/test_manifests.py`
- Modify: `tests/workflow/ppiflow/test_tables.py`
- Modify: `tests/workflow/ppiflow/test_staging.py`
- Modify: `tests/workflow/ppiflow/test_coordinators.py`
- Modify: `src/biomodals/workflow/ppiflow_workflow.py`
- Modify: `src/biomodals/workflow/ppiflow/*.py`

**Migration note:**

Existing in-progress PPIFlow workflow ledgers and artifacts are intentionally
not migrated by this refactor. Users should rerun with `force=True` when they
want a fresh ledger, or provide completed app-owned outputs through explicit
`Stage2Input` plus an optional candidate manifest when resuming from previous
stage outputs.

**Tasks:**

- [x] Document that old in-progress PPIFlow workflow ledgers/artifacts are not migrated.
- [x] Document that users should rerun with `force` or re-enter completed app-owned outputs through explicit `Stage2Input`.
- [x] Update PPIFlow workflow help/docs if user-visible stage config changes.
- [x] Run `uv run pytest tests/workflow/ppiflow tests/workflow/test_ppiflow_workflow.py -q`.
- [x] Run `uv run pytest tests/workflow -q`.
- [x] Run `uv run pytest tests/app/test_dockq_workflow_contract.py tests/app/test_catalog_workflow_apps.py tests/app/test_cli_workflow_catalog.py -q`.
- [x] Run `uv run biomodals workflow list`.
- [x] Run `uv run biomodals workflow help ppiflow`.
- [x] Run `prek run --files <changed files>`.
- [x] Confirm the final branch consists of phase-sized commits rather than one
  large PPIFlow refactor commit.

### Current Verification Baseline

- [x] `uv run pytest tests/workflow/test_ppiflow_workflow.py -q`
- [x] `uv run pytest tests/workflow -q`
- [x] `uv run pytest -q`
- [x] `prek run --files src/biomodals/workflow/ppiflow_workflow.py tests/workflow/test_ppiflow_workflow.py docs/superpowers/plans/2026-06-01-ppiflow-workflow.md`
- [x] `uv run biomodals workflow list`
- [x] `uv run biomodals workflow help ppiflow`

### Self-Review

- Coverage: the DAG covers every named upstream PPIFlow step and every required app, including the approved `alphafold3` dependency for `ReFoldStep`.
- Remaining implementation work: candidate manifests, candidate-wide MPNN/Partial/ReFold/AF3Score/Rosetta coordination, DockQ preparation, strict candidate joins, ReFold quality metrics, attrition reporting, and helper submodule extraction.
- Type consistency: node names, app function handles, artifact kinds, candidate manifests, and app result statuses match Biomodals workflow vocabulary.
