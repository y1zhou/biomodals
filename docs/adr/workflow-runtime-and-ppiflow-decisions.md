# Workflow runtime and PPIFlow decisions

## Adopt the canonical App Run Layout without legacy path compatibility

Biomodals apps moving to `AppRunLayout` will read and write only the canonical `inputs/`, `outputs/`, `logs/`, `failures/`, `metrics/`, and `.markers/` locations. RFdiffusion, Rosetta, FlowPacker workflow outputs, PPIFlow logs, and IgGM logs may require one-time migration or recomputation; the branch will not retain legacy cache probes or dual write formats because it has not yet merged and maintaining two durable layouts would complicate artifact recovery.

## Recover interrupted work through durable ownership

The workflow runtime records Coordinator-Local Task ownership and Provider
Call identity instead of Node attempts. Interrupted local work first observes
its publication and may re-enter the same idempotent operation only when the
result is authoritatively missing. Remote work without a conclusively terminal
owner remains blocked because blindly replacing it could duplicate work that
is still writing deterministic outputs.

## Separate Provider Call completion from scientific publication

A successful remote call remains recoverable until its small Result Envelope
is committed. The Provider Call may then release its slot while its Tasks stay
running through result decoding, artifact materialization, publication, and
validation. Task, Node, and artifact transitions cross the workflow-volume
durability boundary together. This prevents preemption from losing a returned
result without conflating provider completion with scientific completion.

## Treat artifact availability as available, missing, or unknown

Artifact verification will treat an explicit missing result as authoritative, checker failures and unmounted volumes as unknown, and only missing artifacts as grounds to authorize work. Unknown availability suspends the Execution Run until explicit resume repeats validation. A producer must verify its new outputs before durable Task completion; outputs that remain missing fail the Task rather than entering an unbounded retry loop.

## Validate canonical run names at reusable app boundaries

Local entrypoints may normalize a human-provided run name once and report the canonical value, but remote and workflow-compatible app functions will reject non-canonical names. App Run Layout construction for mounted volumes will validate containment so traversal is impossible and distinct user inputs cannot silently collapse onto the same durable cache directory.

## Preserve partial batch results without treating them as success

Batch app adapters will return succeeded only when every requested candidate succeeds, partial when successful and failed candidates are mixed, and failed when none succeed. Partial results retain successful outputs, failure records, and logs, but remain terminal non-success so downstream scientific steps never consume an incomplete candidate or score set implicitly.

## Keep PPIFlow interface-energy analysis workflow-owned

PPIFlow-specific Rosetta interface-energy analysis stays in the workflow for now. The workflow owns Rosetta script generation, expected `residue_energy.csv` discovery, fixed-position derivation, and candidate identity preservation for this stage. The generic Rosetta app remains a command runner instead of growing a PPIFlow-specific workflow-compatible API until that contract proves reusable outside PPIFlow.

## Preserve PPIFlow candidate sets inside static stage Nodes

PPIFlow keeps a static semantic DAG while runtime-discovered candidate Tasks
fan out inside eligible stage Nodes. LigandMPNN, Partial, ReFold, DockQ
preparation, Rosetta analysis, ranking, and reporting preserve candidate
identity across all derived artifacts. Nodes may narrow candidates only
through explicit selector configuration, not by silently taking the first
structure or relying on sorted file order.

## Track expensive PPIFlow work as kernel Provider Calls

The workflow coordinator submits expensive PPIFlow work through the execution
kernel. Candidate-oriented stages use one Task per candidate, and each tracked
provider container invokes the established app function body directly rather
than submitting an untracked nested Modal call. Batch-oriented stages may
retain one Task when their scientific contract is genuinely batch-wide. The
workflow still owns candidate manifests and publication validation.

## Require ReFold quality metrics

PPIFlow ReFold outputs include candidate-keyed quality metrics in addition to refolded structures. AlphaFold3 inference may still return its native archive, but the workflow must derive or expose a metrics table from confidence and ranking outputs so DockQ, Rosetta relax, Rank, and Report do not rely on unkeyed JSON files or structure filename ordering.

## Keep PPIFlow Rosetta semantics in the workflow

PPIFlow owns Rosetta job manifests, script and flags selection, expected
outputs, and per-candidate result interpretation. Generic Task admission,
worker assignment, and Provider Call lifecycle belong to the execution
kernel. The generic Rosetta app remains a command worker so PPIFlow-specific
interface-energy and relax semantics do not leak into its scientific API.

## Derive PPIFlow sequence tables in the workflow

PPIFlow derives `mpnn_seqs.csv` from LigandMPNN artifacts inside the workflow instead of adding PPIFlow-specific sequence-table semantics to the LigandMPNN app. The table is candidate-keyed and records parent provenance so stage 1 and stage 2 sequence outputs can be joined with structures, scores, ranking, and reports.

## Normalize stage-2 PPIFlow inputs to candidate manifests

Stage-2-only PPIFlow runs start from a candidate manifest so downstream scoring, refolding, DockQ pairing, ranking, and reporting can use deterministic candidate identity. A plain user-provided structures path remains a convenience input, but the workflow normalizes it into a minimal manifest with synthetic candidate ids before stage-2 nodes run.

## Store PPIFlow candidate manifests as workflow artifacts

PPIFlow candidate manifests are first-class workflow artifacts stored in the workflow run volume, not only node metadata. Metadata can summarize manifest paths and counts, but downstream nodes and users need a durable table artifact for candidate joins, retry skipping, provenance inspection, and stage-2-only inputs.

## Use Parquet for PPIFlow candidate manifests

PPIFlow candidate manifests are stored as Parquet table artifacts. They are internal provenance and join artifacts, so compact storage, fast Polars reads, typed columns, and nested field support matter more than line-oriented text inspection; upstream-facing tables such as `mpnn_seqs.csv`, AF3Score metrics, DockQ scores, and reports keep their existing formats unless separately changed.

## Fail PPIFlow candidate joins by default

PPIFlow candidate joins fail by default when required candidate identities are missing from either side. Silent drops can hide lost structures or scores in scientific workflows, so missing-candidate tolerance must be an explicit inspection/debug setting rather than the normal execution path.

## Filter PPIFlow candidates through manifests

PPIFlow filter stages narrow the active candidate set by emitting a retained-candidate manifest, not just a filtered score CSV. Rejected candidates are preserved in an audit table with filter outcomes and reasons so downstream stages consume only retained candidates while users can still inspect what was removed.

## Report PPIFlow candidate attrition

PPIFlow keeps rejected, failed, and skipped candidates available for reporting even though downstream scientific stages consume only retained manifests. This separates execution semantics from audit/reporting needs: filters narrow the active candidate set, while the final report can explain where candidates were lost.

## Verify PPIFlow candidate outputs before reuse

PPIFlow validates a candidate's expected output files before completing a Task
from cache or copying a successful publication into a Successor Execution
Run. Candidate manifest rows are durable provenance, but they do not replace
artifact availability checks. Missing output authorizes new work only when no
active or unknown predecessor ownership remains.

## Record workflow and app-volume candidate file locations

PPIFlow candidate manifests record both workflow-relative artifact paths and app-volume paths when both are available. Workflow-relative paths support materialized downstream artifacts and user inspection, while app-volume paths plus volume identity support strict availability checks for app-owned durable outputs without guessing storage ownership.

## Generate PPIFlow candidate ids from provenance

PPIFlow candidate ids are deterministic and provenance-based. Initial candidates hash the producing stage, source artifact id or path, and normalized file basename; derived candidates hash the parent candidate id, stage name, operation mode, and derived output basename. Sequential ids are reserved for synthetic stage-2 convenience manifests and must keep source-path provenance so users can reconcile them later.

## Keep PPIFlow candidate ids out of DAG hashes

PPIFlow candidate ids are runtime provenance for produced artifacts, not semantic workflow DAG configuration. Changing candidate-id helper internals is a manifest migration concern unless user-facing workflow configuration changes; candidate ids and candidate manifests must not be added to node hash payloads.

## Use one-row-per-candidate PPIFlow manifests

PPIFlow candidate manifests store one Parquet row per candidate with a nested list of file records. Candidate-level joins, filtering, ranking, and reporting are the common operations, and file-level availability checks can expand the nested file list when needed.

## Keep the PPIFlow manifest schema workflow-local

The PPIFlow candidate manifest schema starts as workflow-local helpers rather than shared `biomodals.schema` models. It is currently a PPIFlow-specific provenance contract, and moving it into shared schemas should wait until another workflow needs the same candidate-manifest abstraction.

## Rank retained PPIFlow candidates with usable scores

PPIFlow `ranked_designs.csv` contains only retained candidates with at least one usable ranking signal from DockQ, AF3/ReFold, or Rosetta. Rejected, partial, failed, skipped, and unrankable candidates remain visible through attrition and report tables instead of null-ranked rows because that makes the ranking artifact ambiguous. If no retained candidates are rankable, the rank step writes empty rank artifacts with a warning so report generation can still complete.

## Keep PPIFlow report generation workflow-native

PPIFlow report generation stays a workflow-native transform that renders Markdown and HTML from materialized tables, manifests, and score artifacts. It has no expensive external runtime today, is easy to unit test, and should move to a separate app only if report rendering later needs heavyweight dependencies.

## Enforce Run-level Provider Call limits in SQLite

Workflow `max_parallel` sets the workflow adapter's
`max_parallel_nodes` limit, not a Modal container limit. A workflow caller may
also use that public value as the initial Provider Call ceiling, but Node
parallelism and call admission remain independent runtime controls. The
execution repository atomically enforces `max_active_provider_calls` and its
GPU subset by counting nonterminal Provider Calls in one Execution Run.
Candidate concurrency, AF3Score job count, Rosetta worker count, and BoltzGen
parallel runs are caller-side inputs to kernel dispatch; they do not form
separate durable schedulers, shared leases, or cross-run resource managers.

## Split PPIFlow workflow helpers into a submodule

PPIFlow-local manifest, table, staging, and coordinator helpers live under a `biomodals.workflow.ppiflow` submodule, while `ppiflow_workflow.py` remains the public workflow module discovered by the CLI and catalog. This keeps the top-level workflow module focused on DAG assembly and node contracts instead of absorbing all candidate-manifest and stage-coordinator mechanics.

## Keep PPIFlow node classes in the workflow module

PPIFlow node classes stay in `ppiflow_workflow.py` because they define the visible workflow DAG contract. Helper internals for manifests, tables, staging, and candidate-wide coordination move into `biomodals.workflow.ppiflow`, but the public workflow module remains the place to read node contracts and DAG assembly.

## Split PPIFlow helper tests from workflow integration tests

Pure PPIFlow helper tests live in focused test modules separate from `tests/workflow/test_ppiflow_workflow.py`. The workflow test file keeps DAG shape, node contract, and namespace integration coverage, while manifest, table, staging, and coordinator helper tests can fail independently with clearer scope.

## Use an internal non-underscore PPIFlow submodule

PPIFlow helpers live under `biomodals.workflow.ppiflow` rather than `_ppiflow`. The name is clearer and matches the workflow domain, but the submodule remains workflow-internal with a minimal or empty `__all__`; user-facing workflow access stays through `ppiflow_workflow.py` and the CLI/catalog.

## No migration for old PPIFlow workflow runs

The PPIFlow workflow refactor does not migrate old in-progress workflow ledgers
or artifact manifests. Old ledgers are rejected and a new launch receives a
fresh Execution Run ID. Useful completed app-owned outputs can be reintroduced
through explicit `Stage2Input`; `force` is a workload-output option, not an
execution-state migration.

## Keep PPIFlow Modal bindings in the workflow module

PPIFlow Modal decorators, app registration, and app-bound remote helper functions stay in `ppiflow_workflow.py`. The `biomodals.workflow.ppiflow` submodule provides pure or near-pure helper logic for manifests, tables, staging, and coordinator mechanics so importing helper modules does not create hidden Modal app registration side effects and helper tests can run without Modal bindings.

## Use stage-specific PPIFlow provider wrappers

PPIFlow uses stage-specific Modal wrappers whose granularity matches the
kernel Task or scientifically indivisible batch. Candidate wrappers call the
established app function body in the tracked provider container; they do not
coordinate nested Modal fan-out. Separate function names keep logs, failures,
mounts, and future resource settings understandable per stage.

## Mount only stage-required PPIFlow volumes

PPIFlow stage-specific remote wrappers mount only the workflow and app volumes needed by that stage. Shared helpers receive an explicit volume map from the wrapper, keeping mount scope smaller and making expected-file verification depend on declared stage inputs rather than every possible app volume.

## Defer PPIFlow stage resource tuning

PPIFlow stage-specific remote wrappers start with the current workflow resource defaults. Separate wrappers preserve a clean path for later CPU, memory, timeout, GPU, and mount tuning, but this refactor should not guess resource settings before real stage telemetry exists.

## Commit the PPIFlow refactor by phase

The PPIFlow refactor should be committed in phase-sized changes rather than one large commit. Phase commits make review, bisecting, and rollback easier; pushing remains a separate explicit action.

## Prune workflow runs from terminal nodes

The workflow runtime will decide run completion and resume scope from terminal workflow nodes. If every terminal node has durable completion and non-missing recorded outputs, the run succeeds without scheduling intermediate nodes, even when stale failed, running, or incomplete intermediate state remains. If some terminal nodes are incomplete, the scheduler only considers those terminals and their ancestor closure.

This keeps execution result-driven and avoids recomputing expensive
intermediate work when the externally relevant workflow outputs already exist.
Missing terminal artifacts invalidate completion; unknown external artifact
availability suspends the Run and admits no new work. `resume` continues a
suspended Run without retrying failed Tasks. An explicit Successor Execution
Run revalidates terminal publications and repairs only the missing backward
closure.
