# Humanization website and API

Status: implemented and offline-verified. Deployment and scientific smoke
runs require separate approval.

## Agreed product requirements

Recorded 2026-09-07 during the coordinated backend/frontend interview.

- Provide ID, VH, and VL input boxes and an Add button that adds the complete
  pair to a batch. Suggest editable IDs such as `ab_001`; accept any valid,
  non-duplicate ID. CSV upload appends to the same editable batch. Users can
  edit or remove pairs before submission and must resolve duplicate IDs.
- Display the workflow's `selection.csv` table on the webpage when the run
  finishes. Provide pagination, sorting, parent filtering, and all columns,
  including expandable/copyable sequences. Preserve the workflow's default
  ordering, scientific values, nulls, and unranked candidates. User sorting
  changes presentation only. Offer direct CSV download as well as the archive.
- Provide an Advanced section exposing the app-specific knobs available in
  `submit_humanization_workflow`, grouped into General, Sapiens, Humatch,
  p-AbNatiV2, and HuDiff subsections. One scientific configuration applies to
  the entire batch. General includes the job name; container ceilings remain
  admin-controlled. Do not expose CLI-only wait, deployment selection, or
  restart controls in the submission form.
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

## Implementation plan

1. Integrate against current main's service/kernel contracts after resolving
   the humanization stack baseline, preserving the PRs under review. Finalize
   typed submission, validation-error, and paginated result contracts so both
   repositories share one API schema.
2. Add the narrow workflow-owned coordinator adaptation and humanization Tool
   registration/adapter/router. Reuse shared admission, authentication,
   idempotency, remote lifecycle, limits, progress, and result delivery.
3. Add owner-authorized, bounded Polars result queries with agreed sorting and
   paging, direct CSV download, and archive packaging/restoration. Reuse local
   result caching so page requests do not repeatedly fetch Modal artifacts.
4. Build the frontend batch editor, CSV import, grouped Advanced controls, and
   submission/recovery flow. Extend shared Tool routes and Job detail with a
   server-paginated selection table, not a separate job lifecycle.
5. Verify the shared OpenAPI contract, validation and ownership boundaries,
   idempotent recovery, partial results, numeric/null sorting, stable paging,
   and downloads. Run offline integration/browser tests and local performance
   checks before proposing a separately approved paid Modal smoke run.

The user approved implementation and merging the humanization stack first.
PRs #49–#54 were atomically merged into main at `e25d570` after conflict
resolution against the service baseline. Both histories are included.

## HTTP contract

- `GET /api/v1/humanization/options`: authenticated limit, complete scientific
  defaults, and flat settings JSON schema. `BIOMODALS_HUMANIZATION_MAX_PAIRS`
  defaults to 100 and may be configured from 1 through the parser limit of
  1000. There is no independent frontend limit configuration.
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

## Integration evidence

Use current main's service architecture, not the older service implementation
in the humanization PR stack. The shared service lifecycle is reusable, but
the generic workflow coordinator needs explicit protocol adaptation.
See [backend research](../research/humanization/service-integration.md) and
[frontend research](../../../biomodals-frontend/docs/research/2026-09-07-humanization-ui.md).

The Humanization Batch glossary distinguishes standalone-app batch failure
from the workflow's useful partial-results policy. The service preserves the
workflow policy rather than redefining its scientific outcomes.

## Local query measurement

A disposable synthetic check with 100 parents and 29 rows per parent (2,900
rows; 200 residues in each sequence) produced a 1,360,041-byte CSV and a
91,340-byte typed first-page response. Median local parse, descending numeric
sort, 50-row pagination, and JSON serialization was 5.39 ms across 20 measured
iterations after warm-up. This measures neither browser rendering nor network
or Modal download latency; no benchmark harness is added to production code.

## Offline verification

The backend suite passed 1,567 tests after integration, bounded ZIP-member
table parsing, runtime preparation, cancellation acknowledgements, and
independent reconciliation. Seven humanization browser
tests passed, including a real HTTP
fixture using the shared service lifecycle and a fake scientific adapter:
100 submitted pairs produced 300 rows, with only 50 rows (91,656 response
bytes) returned on opening results. Checks covered nulls-last sorting in both
directions, original-order reset, parent filtering, CSV/archive downloads,
anonymous denial, reauthentication, and submission recovery. The 100-pair
input render measured 859 ms in that local browser run.

The complete eight-test browser suite subsequently passed in 37.0 seconds,
including the existing MVP GROMACS workflow after the refresh correction.

The browser checks exposed an existing shared refresh bug: polling an active
root returned the old stage projection even on explicit Refresh. Explicit
Refresh now reads current stage details without resuming an already-active
coordinator; background polling retains its inexpensive root-only check.

These offline checks made no cloud submissions or deployments.

## Live verification (2026-09-07)

With user authorization and a total $10 development budget, the workflow was
deployed to `production/HumanizationWorkflow`. Two separate one-pair smoke
Jobs were submitted, each requesting one HuDiff candidate attempt. Neither
was a full 100-pair load test.

Version 1 exposed missing production model assets: Sapiens and Humatch
succeeded, HuDiff failed its manifest validation, and p-AbNatiV2 repeatedly
failed container startup because its read-only model mount did not exist.
The failing provider call and coordinator were explicitly cancelled. The
provider subsequently reported failure; all original execution calls are
terminal. This was not a successful end-to-end scientific run.

The follow-up changes added app-owned environment preparation before launch,
kept status reads responsive during cancellation, retained explicit provider
cancellation acknowledgements in the execution kernel, and isolated slow
reconciliation Jobs. Version 2 contains the preparation and execution-kernel
fixes; the queue-isolation change runs in the API process.

The live frontend verified authenticated options and status reads against
version 2, including the queued model-preparation explanation and responsive
cancellation presentation. Model preparation completed successfully and the
coordinator began execution just as the fixed 15-minute cutoff was reached.
The watchdog requested cancellation at 10:34:55 UTC; the API subsequently
confirmed terminal `cancelled` with cancelled workflow stages. No live result
table was produced, so successful end-to-end scientific execution remains
unverified by this service smoke test. The complete eight-test offline browser
suite passed again against the queue-isolation fix in 37.0 seconds.

After the API restarted with that fix, the frontend confirmed terminal
Cancelled and all three cancelled stages, with no result table or cancel
button, no browser errors, and a 463 ms authenticated Job read. Modal then
reported no active containers for this deployment. Both temporary smoke
accounts were disabled after browser verification, revoking their sessions.

No further scientific submissions were made. Exact billed spend is not
available from these checks and must not be inferred from elapsed time alone.

### Successful follow-up smoke test

The user subsequently authorized an additional $10 maximum testing budget.
Exactly one further one-pair Job was submitted to the unchanged version 2
deployment, again requesting one HuDiff candidate attempt. Job
`a0a6d7db-5335-4f0a-b96d-0aefb05965dd` was admitted at 12:08:05 UTC and
completed successfully at 12:18:13 UTC, before its 15-minute automatic cutoff.
Existing model assets were revalidated rather than downloaded again.

All four generation methods completed. Generation took 8 minutes 7 seconds;
union collection took 2 seconds; evaluation and ranking took 36 seconds.
The final table contains 45 columns and three unique paired sequences: the
parent plus two generated candidates, ranked first and second. Every row has
`evaluation_complete=true`, all annotation/evaluation errors are null, and
the manifest contains no generation or execution errors. Parental generation
provenance is null, and CDR preservation is reported for all three rows.

The live frontend verified Completed for the Job and all three stages, the
initial `offset=0&limit=50` table request, ascending and descending sorting
with nulls last, exact scientific-order restoration, and parent filtering.
All five selection requests returned HTTP 200, with no browser errors or
extra scientific submissions. Native CSV (3,391 bytes) and ZIP (1,501,826
bytes) downloads succeeded through the existing preparation/download flow.
The direct CSV exactly matches the archive member; all ten declared artifact
sizes and SHA-256 digests match, and ZIP integrity checks pass.

This closes the live end-to-end service/UI verification gap above. It does
not replace larger-batch scientific validation or experimental assessment.
No production code changes or additional deployments were needed. Modal
reported no active containers afterward, and the temporary account was
disabled with sessions revoked. Exact billed spend remains unavailable;
only one bounded run was used from the additional budget.
