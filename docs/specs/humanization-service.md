# Humanization website and API

Status: implemented and offline-verified. Deployment and scientific smoke
runs require separate approval.

## Website contract

- Provide ID, VH, and VL input boxes and an Add button that adds the complete
  pair to a batch. Suggest editable IDs such as `ab_001`; accept any valid,
  non-duplicate ID. CSV upload appends to the same editable batch. Users can
  edit or remove pairs before submission and must resolve duplicate IDs.
- Place Job name above the batch editor, outside Advanced. Accept CSV files
  up to and including 10 MiB; reject larger files before parsing without
  changing the batch. This browser import limit is separate from the normalized
  JSON API request limit and does not increase the allowed number of pairs.
- Display the workflow's `selection.csv` table on the webpage when the run
  finishes. Provide pagination, sorting, parent filtering, and all columns,
  including expandable/copyable sequences. Preserve the workflow's default
  ordering, scientific values, nulls, and unranked candidates. User sorting
  changes presentation only. Offer direct CSV download as well as the archive.
- Provide an Advanced section exposing the app-specific knobs available in
  `submit_humanization_workflow`, grouped into General, Sapiens, Humatch,
  p-AbNatiV2, and HuDiff subsections. One scientific configuration applies to
  the entire batch. General holds any shared scientific controls, not Job name;
  container ceilings remain admin-controlled. Do not expose CLI-only wait,
  deployment selection, or restart controls in the submission form.
- General Root seed defaults to 0 and explicitly sets both `pabnativ2_seed`
  and `hudiff_ab_seed`; the standalone HuDiff default remains 42. General
  Allow CDR mutations sets Sapiens, Humatch and p-AbNatiV2 controls together;
  HuDiff has no corresponding option. These are frontend mappings, not new
  generic API fields.
- Sapiens iterations retain every paired intermediate design for shared
  evaluation. p-AbNatiV2 Sampling attempts per parent maps to
  `pabnativ2_num_seeds` (integer 1–25, default 1), meaning independent optimizer
  runs, not guaranteed unique sequences. See the scientific
  [generation contract](humanization-workflow.md#verified-run-controls).
  The API and its pinned containing workflow must have matching scientific
  settings and identities. Older options without this control are unsupported
  by the current form; editing may continue, but submission is disabled.
- Admit at most 100 pairs per job by default. Control this limit with a
  backend environment variable, not separate frontend configuration.
- Keep invalid imported rows visible and highlighted so users can fix or
  remove them. Block submission until all rows are valid; never silently drop
  rows. Reject structurally malformed CSV without changing the existing batch.
- Keep unfinished batches in memory only. Preserve the mounted form during
  reauthentication, but do not persist antibody sequences in browser storage
  or restore drafts after a page reload. Warn before leaving an unsent batch.
- Provide explicit Check submission recovery after a lost submission response,
  reusing the original idempotency key and unchanged request. Editing the batch
  or settings creates a new submission intent, not a replay of the old one.
- Normalize sequences from manual entry and CSV by uppercasing and removing
  whitespace, showing the normalized sequences before submission. Reject other
  invalid characters rather than silently removing them.
- Both input experiences submit one typed JSON payload containing the job
  name, complete pairs, and batch-wide settings. Backend validation remains
  authoritative, including the environment-controlled pair limit.
- Perform result-table filtering, sorting, and pagination on the backend using
  Polars against the published `selection.csv`. Opening results fetches only
  the first page, not the complete table. Subsequent requests send query
  parameters and receive bounded pages; they do not upload table contents.
  Keep these reads separate from job-status polling and preserve the unchanged
  CSV download. Backend operations are preferred even if the current table
  would fit in browser memory, to avoid the initial full-table transfer.
- Default to 50 rows per page. Custom sorts place nulls last in either
  direction and break ties deterministically by parent/candidate ID. Without
  a custom sort, preserve the workflow's original row ordering.
- Viewing, filtering, sorting, and downloading are sufficient for this
  release. Do not add persistent experimental-panel selections.
- Complete VH-VL pairing, candidate generation/evaluation, per-parent ranking,
  and useful partial scientific results remain governed by the existing
  [workflow specification](humanization-workflow.md).

## HTTP contract

- `GET /api/v1/humanization/options`: authenticated limit, complete scientific
  defaults, and flat settings JSON schema. `BIOMODALS_HUMANIZATION_MAX_PAIRS`
  defaults to 100 and may be configured from 1 through 200. This website bound
  keeps supported result tables within the bounded reader; the CLI parser has
  a separate limit. There is no independent frontend limit configuration.
  Required `max_vh_length=142` and `max_vl_length=126` expose the Sapiens
  limits for website admission. Both the editor and server reject longer
  normalized sequences with row-addressable `sequence_too_long` errors before
  provider preflight. This website policy does not narrow the CLI/workflow
  parser or guarantee that every shorter input can be numbered by every model.
- `GET /api/v1/humanization/jobs/{job_id}/inputs`: owner-only editable
  `HumanizationSubmission` (`display_name`, `pairs`, `settings`), with
  `Cache-Control: private, no-store`. Read local pending input if available,
  otherwise the original job's immutable staged request in its Modal environment.
  Missing/foreign/wrong-tool jobs return 404; unavailable retained input returns
  404 `job_input_unavailable`. Retrieval is permitted for any job state and
  preserves historical oversized inputs for correction. Restored per-model
  settings remain unchanged until a shared control is edited. No models are
  prepared and no work is launched. The Job detail rerun button opens a memory-only
  editable draft; explicit submission creates a new Job and idempotency intent
  under the current deployment, not an execution-kernel Successor Run.
- `POST /api/v1/humanization/jobs`: typed `display_name`, `pairs`, and
  `settings`; existing session/Origin/CSRF and UUID Idempotency-Key contract.
  Returns the existing `JobView` with HTTP 202 after durable admission.
  Requests are capped at 4 MiB before JSON parsing. Semantic validation errors
  are HTTP 422 with code `humanization_input_invalid` and zero-based
  `errors[].row_index`, field, code, and message; batch/settings errors use a
  null row index. Structurally invalid requests use standard HTTP 422 errors.
- `GET /api/v1/humanization/jobs/{job_id}/selection`: `offset`, `limit`
  (default 50, maximum 200), optional exact `parent_id`, optional `sort_by`,
  and `descending`. Returns typed column metadata, only the requested rows,
  filtered total count, offset/limit, and available parent IDs. Cached ZIP
  reads and Polars operations run in the bounded I/O executor.
  Required `default_hidden_columns: string[]` reports data-dependent display
  defaults from the full authoritative result, before any filter/sort/page:
  each `*_error` column is hidden when all its values are null, and
  `evaluation_complete` is hidden unless at least one row is false. These
  defaults remain stable even for empty pages or parent filters. All columns
  and values remain available in the response and unchanged CSV; static
  defaults for `is_parent`, `cdr_preservation`, and Humatch family labels are
  frontend-owned. Computing the flags reuses the existing parsed table.
  Required `nativeness_ranges: Record<string, {min: number, max: number}>`
  contains finite minimum and maximum values over the full result for each
  of the three original `pabnativ2_*_nativeness` columns, not their deltas.
  Columns without finite values have both bounds zero. Ranges are independent
  of pages and filters, and reuse the parsed table without additional reads.
- `GET /api/v1/humanization/jobs/{job_id}/selection.csv`: original CSV as a
  native browser download. Both selection endpoints require ownership and a
  successful or partial result. An evicted local result returns coded 409
  `result_not_cached`; the frontend calls the existing authenticated
  prepare-download mutation then retries once. It does not rerun science.
- Full archives use the existing shared Job download endpoints. The adapter
  verifies workflow publication identity and file digests, then creates a
  deterministic ZIP suitable for exact cache restoration.

The workflow uses an immutable staged JSON request and the shared graph-host
lifecycle. Its coordinator run/resume return bounded execution overviews;
CLI result retrieval separately reads the actual terminal artifact locations.
Graph successors retain validated successful publications, including when the
terminal bundle belongs to a predecessor Run. The service does not own a
second scheduler or open remote execution ledgers.

The staged request retains scientific versions from the submitting API's
checkout; the coordinator requires them to match its own deployment. These
versions include HuDiff's resolved runtime environment digest, imported from
the app automatically, so a dependency update can change this identity without
a workflow-code change. Update the API checkout and containing workflow
together, pin the matching deployment, and restart the API. A mismatch error
names the differing components; submit a new Job after aligning versions.
Do not rewrite an older request's identities or bypass the check.

### Result preparation

Result membership and compatibility follow the
[workflow publication contract](humanization-workflow.md#result-publication).
Use the shared bounded manifest reader and bulk Modal downloader. Await bulk
transfers through `run_blocking_io`, which drains the worker before propagating
cancellation and releasing its temporary directory. Transfers do not occupy
the cache's single worker. Local archive construction, digest verification and
atomic publication remain on `cache.run_bounded`.

The 32 MiB selection-member guard bounds both table queries and direct CSV
reads. Retain this guard when changing admission; verify the largest advertised
pair/yield envelope. Result restoration never launches science.

### Concurrent generation stages

New workflow plans expose six Job stages: `generate_sapiens` (Sapiens),
`generate_humatch` (Humatch), `generate_pabnativ2` (p-AbNatiV2),
`generate_hudiff_ab` (HuDiff), `union`, and `evaluate`. Each generator owns
one independent kernel Node with per-pair Tasks, so status, timings, counts,
and stage-filtered log targets come from that method's actual records.
All four Nodes can run concurrently under the existing shared provider limits;
union collection waits for all four. No `JobView` schema change is needed.

Each method retains the existing local baseline publication and partial-result
policy, including when every provider Task for that method fails. Generation
errors are combined at the union boundary and still reach final publication.
Older pinned plans with a single `generate` Node retain their aggregate stage;
method timings must not be inferred from an aggregate record. The graph change
requires a new workflow deployment and corresponding service version pin before
new live Jobs can use these rows. Existing historical Runs are not rewritten.

### Runtime preparation

Services must establish runtime prerequisites before scientific execution.
Humanization invokes the pinned deployment's existing CPU-only HuDiff and
p-AbNatiV2 model stagers before launching its coordinator. Jobs remain queued
with `preparing_environment` while this runs and can be cancelled without
waiting for model downloads. Preparation is shared by jobs using the same
deployment within one service process; restart or deployment changes cause
the app-owned stagers to revalidate their publications. A conclusive failure
marks the unlaunched Job `failed` with `environment_preparation_failed`, freeing
its admission slot. The failed preparation is retained until operator
intervention and service restart rather than automatically repeating downloads.

Sapiens and Humatch bake their assets into images. GROMACS also installs its
runtime dependencies in its image; AlphaFold3 already owns automatic runtime
asset preparation. Service adapters reuse these app-owned contracts rather
than duplicating model/database setup logic in the API.

Cancellation records durable intent before returning to the browser. The
reconciler delivers that intent remotely; ordinary status reads return the
latest stored projection while a remote operation holds the per-job lock.
Reconciliation keeps at most four independent Jobs in flight across admission
wakeups, so one slow provider operation does not block the entire queue.
Completed snapshots are rescanned on wakeups or the interval, not continuously.

## Result display semantics

`quality_tier` and `panel_order` are per-parent, one-based ascending ranks.
Tier 1 is the first Pareto front; panel order is a diversity-aware suggested
selection sequence, not a composite fitness score. Null ranks identify the
parent or candidates outside ranking eligibility, not a worst numeric rank.

The service presents stored ranks without recalculation; sorting is only a
view operation. See the [workflow ranking policy](humanization-workflow.md#panel-ordering)
for objectives, eligibility, versioning and diversity selection.

Sapiens mean residue probabilities, Humatch classifier probabilities and
pairing score, and p-AbNatiV2 pairing score use fractions in `[0,1]`. The
p-AbNatiV2 upstream pairing column's percent sign is misleading: its value
is the raw sigmoid fraction, with no division by 100. Humatch germline
likeness is an average observed residue-frequency score in `[0,1]`, not
sequence identity or confidence. p-AbNatiV2 nativeness scores are affine
rescalings of reconstruction scores and can be negative; do not clamp or
present them as probabilities. See the [p-AbNatiV2 score research](../research/humanization/pabnativ2.md#score-semantics).

For visualization only, original nativeness bars use `(raw_value - min) /
(max - min)` to fit `[0,1]`. Their deltas use `raw_delta / (max - min)` with
the same original column's range, yielding `[-1,1]` because candidate and
parent scores both belong to the full result. Constant ranges produce a
neutral score extent of `0.5` and delta extent of `0`; null remains missing.
Always display the raw numeric value.
This is batch-relative visual scaling, not scientific standardization: bar
lengths are not directly comparable across columns or different Jobs.

Deltas are candidate minus parental score in the original score units, not
relative percentages. Positive means a higher score in the model's preferred
direction; it does not establish experimental improvement or reduced
immunogenicity. Best-family probability deltas are omitted because the best
family may change between sequences.

## Ownership and verification

Humanization reuses the registered Tool adapter/router, shared remote-authority
bridge, authentication, admission, idempotency, logs and Result lifecycle.
Its workflow-owned staged request adapts the generic graph coordinator to the
service's `run`/`resume` and bounded `ExecutionOverview` contract. It does not
introduce a service-local scheduler. Shared lifecycle behavior belongs to the
[API Tool service contract](api-tool-service.md) and
[remote-authority decision](../adr/0007-api-jobs-use-remote-coordinators.md).

Offline tests use fake providers and real HTTP/ZIP paths to cover admission,
ownership, replay/recovery, input restoration, six-stage mixed completion,
server paging/filter/sort, null handling, CSV/archive downloads and cancellation.
Frontend tests remain in the frontend repository; generated types use the exact
backend OpenAPI export. Local tests do not establish model equivalence.

An authorized one-pair service/UI smoke run on the older version 2 deployment
completed with three candidates and verified all five selection requests,
native CSV and ZIP downloads, and artifact hashes. Earlier missing-model and
slow-status failures motivated the preparation and lock behavior above.
That history is not evidence that the currently edited workflow is deployed.
For rollout, deploy/pin matching workflow and service versions, rerun offline
contract checks, then obtain approval for bounded live scientific verification.
