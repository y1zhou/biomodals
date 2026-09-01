# API Tool service specification

Status: accepted and implemented

Last updated: 2026-08-27

This specification applies ADR 0007 to the implemented FastAPI service and its
frontend.

## Responsibilities

The service owns Tool discovery, typed submission endpoints, Service Jobs,
authentication, admission, cached projections, result delivery, and
Administrator configuration. A deployed Tool coordinator owns every kernel
execution transition.

The service exposes `tool` in HTTP and persistence contracts. `workload`
remains internal execution-kernel vocabulary.

## Service database

`service.sqlite3` contains only six service-owned tables: `users`,
`password_tokens`, `sessions`, `service_settings`, `tool_settings`, and
`jobs`. It contains no execution-kernel tables, input BLOB table, billing
cache, log records, or normalized local copy of Nodes, Tasks, Provider Calls,
or stage history.

The Service Job ID is also its Execution Run ID. One Job row retains ownership,
Tool and display identity, idempotency request digest, exact Deployment
Identity, root coordinator Function Call ID, current public state, durable
cancellation intent, refresh and finalization timing, and prepared-Result
metadata. Frequently filtered owner, Tool, state, and time values are ordinary
indexed columns.

Semantic stages, Task status counts, warnings, and safe errors form one bounded
`jobs.projection_json` document. A successful remote observation atomically
replaces that document together with `state` and `projection_observed_at`.
This document is a disposable website cache, not execution authority; it
contains no Task payloads, provider bindings, logs, submitted input, or DAG.
The exact Deployment Identity and Job ID can reconstruct it from the remote
coordinator.

Pending request bodies live under the state directory rather than in SQLite.
An AlphaFold3 Job temporarily references its validated resource through a
unique nullable `jobs.pending_validation_id`; the field is cleared after
verified remote staging. Unsubmitted validations expire after 24 hours.
Prepared Results retain filename, media type, size, SHA-256, archive schema,
and `cache_cleared_at`; no remote Volume path is duplicated locally. Clearing
and later reconstructing an archive preserves its Result metadata while
updating that cache timestamp.

## Registration

Each registered Tool contributes:

- a `ToolDefinition` containing its stable key, display metadata, ordered
  semantic stages, deployment defaults, and default log visibility;
- a `ToolAdapter` that validates and stages its request and retrieves its
  published Result; and
- a typed submission router.

The shared `RemoteExecutionClient` performs exact-version coordinator
preflight and resolution, launches Runs, polls root Function Calls, refreshes
bounded execution status, requests cancellation, pages Provider Calls, reads
one selected Provider Call directly by internal ID, and accesses its logs. It
rejects any returned overview whose Run ID or Deployment Identity differs from
the pinned Service Job locator. Tool registrations do not contribute lifecycle
callback bags or background reconcilers.

Tool discovery remains explicit at application assembly time. There is no
import-time plugin scan.

## Routes

Submission is Tool-specific:

```text
POST /api/v1/gromacs/jobs
POST /api/v1/alphafold3/validations
GET  /api/v1/alphafold3/validations/{validation_id}
GET  /api/v1/alphafold3/validations/{validation_id}/document
DELETE /api/v1/alphafold3/validations/{validation_id}
POST /api/v1/alphafold3/jobs
```

Lifecycle and delivery are shared:

```text
GET  /api/v1/jobs
GET  /api/v1/jobs/{job_id}
POST /api/v1/jobs/{job_id}/refresh
POST /api/v1/jobs/{job_id}/cancel
POST /api/v1/jobs/{job_id}/prepare-download
GET  /api/v1/jobs/{job_id}/download
```

Job list requests use only the local Job State Projection. A detail request
refreshes an active remote Run only when its projection is at least 60 seconds
old. The explicit refresh route bypasses that freshness check while preserving
per-Job serialization.

The background reconciler runs every 60 seconds by default. While a root
Function Call is active it uses the SDK's nonblocking root-call status only;
it reads the detailed coordinator ledger after terminal root completion or an
interactive detail/refresh request. Root-call failure is terminal rather than
indistinguishable from an active timeout.
An active timeout touches only `jobs.updated_at`, moving that Job behind older
reconciliation candidates without pretending that its detailed projection was
refreshed. This lets a bounded 100-Job pass rotate fairly without another
scheduler cursor.
Each pass advances at most four independent Jobs concurrently, regardless of
Tool. A terminal root observation enters `finalizing` and prepares the Result
in that same background pass rather than waiting for another interval.

The public Job states remain `queued`, `running`, `finalizing`,
`cancel_requested`, `state_unknown`, `blocked`, `succeeded`, `partial`,
`failed`, and `cancelled`. Before remote execution begins, an admitted Job is
`queued`. Active and terminal scientific states are projected from the remote
Run. A publishable remote `succeeded` or `partial` outcome temporarily becomes
`finalizing` while the service prepares or restores a verified download, then
returns to the corresponding terminal outcome. `blocked` represents either a
remote suspended Run requiring intervention or a recoverable service-owned
result-delivery failure. A Job is not `succeeded` until its Result is verified
as downloadable.

An AlphaFold3 Job with the same seed- and name-neutral publication scope can
remain `queued` before remote launch while an earlier matching Job may still
publish overlapping results. Its public
`state_reason` is `waiting_for_shared_publication`, and `state_message` provides
the owner-safe explanation. These fields describe service lifecycle state and
do not copy remote errors.

## Semantic stages

A Tool Definition owns an ordered list of semantic stages. Each stage maps one
or more kernel Node keys into one user-facing label. The service aggregates
Task status counts across mapped
Nodes. For an active stage, the disposable projection may expose
`provider_state` as `queued` while work is pending without an assigned Provider
Call, or while Modal has accepted a Function Call but has not assigned a
container. It becomes `running` after assignment. Failure to obtain the
presentation-only SDK hint for an existing call defaults to `running`.
Provider Call selection and raw Node keys are diagnostic details.
The regular Job-stage table therefore shows the semantic stage, status,
started time, and finished time without a provider-function column. Provider
function names remain available only as diagnostic log-target metadata.

## Provider Call limits

Each Tool's active Job limit also determines the per-Job Provider Call limits
snapshotted during admission. The total container ceiling is eight times the
active Job limit; the GPU subset ceiling equals the active Job limit. A disabled
Tool admits no Jobs. This keeps one capacity control in the Administrator UI
while preserving explicit immutable limits on every admitted Job.

The deployed Modal App name is startup configuration only. Changing it requires
updating the Tool's `.env` value and restarting the API service. Administrators
may still pin a deployment version and choose Job-log visibility per Tool.

## Result metadata

Prepared Results record a friendly filename, media type, byte size, SHA-256
digest, archive schema version, authoritative provider location, and local
cache state. Shared download handling uses this metadata rather than assuming
ZIP.

The initial formats are:

| Tool | Format | Media type |
| --- | --- | --- |
| GROMACS | `.zip` | `application/zip` |
| AlphaFold3 | `.tar.zst` | `application/zstd` |

AlphaFold3 reuses its existing content-addressed request manifest and verified
archive builder without a website-specific layout. The service stages the
resulting `{sanitized-job-name}_{view-id}_AlphaFold3.tar.zst` in its configured
cache and records its metadata. It does not reinterpret the manifest or select
scientific artifacts independently of the app.
Archive members are sorted and tar ownership, time, and format metadata are
normalized so rebuilding identical remote publications produces the recorded
SHA-256 byte for byte.

GROMACS intentionally retains its existing service-owned ZIP builder because
the app publishes a verified directory and exact file set rather than an
end-user archive. The GROMACS Tool Adapter packages that publication into the
existing `input.pdb`, `outputs/`, and `metadata/` schema. It does not infer
successful files independently of the coordinator publication, and this API
presentation behavior does not require changes to `gromacs_app.py`.

The service does not upload that ZIP back to Modal and does not delete the
app-owned GROMACS scientific publication. A cleared local archive is rebuilt
from the verified remote publication. Retention of scientific Modal outputs is
an app policy, not a service-cache cleanup operation.

## AlphaFold3 submission

AlphaFold3 first creates a validated submission resource:

```text
POST /api/v1/alphafold3/validations
```

Its JSON body is exactly one native AlphaFold3 Job document. `AF3Config` must
be able to read it and write the same schema; the service does not wrap it in
another request object. The document's required `name` is the form-owned Job
name. Operational search and inference options that are not part of the
upstream schema are query parameters: `search_msa`,
`search_protein_templates`, `recycle`, and `sample`. Both frontend modes
construct this same document and
the backend passes the retained validated `AF3Config` through the existing
`AlphaFold3ExecutionRequest` preparation and staging path.

Validation returns a `validation_id`, bounded preview, request digest, and
expiry. `POST /api/v1/alphafold3/jobs` accepts only that `validation_id`; it
does not upload the document again. Validation resources are private to their
creating User and may be read or deleted through the corresponding routes.
The document route exists only to download the retained native JSON without
embedding it in the bounded validation response.

The form uses one **Expert mode** toggle. Off selects the guided entity editor;
on selects native JSON upload. Both retain the form-owned Job name and converge
on the same server validation and confirmation flow.

Regular Mode constructs a native AlphaFold3 JSON configuration from one or
more entities. Each row selects Protein, DNA, RNA, or Ligand and contains an
identical-copy count and automatically assigned uppercase chain IDs. Polymer
rows accept sequences; Ligand rows accept either comma-separated CCD codes or
one SMILES string. The UI shows assigned IDs as read-only and defaults model
seeds to `1`. It does not expose modifications, covalent bonds, custom MSA,
custom templates, or user-provided CCD definitions.

The entity editor follows the supplied AlphaFold input interaction reference.
An empty entity is a compact horizontal row with a right chevron, an Entity
type dropdown, copy count, a large **Sequence or FASTA records** input, and row
controls. After parsing, a polymer row presents the normalized sequence in a
wrapped monospace view with residue numbers every ten positions. **Add entity**
creates another row. A raw sequence remains one entity. A multi-record FASTA is
expanded into one single-copy entity per record using the selected polymer
type; each FASTA header is written to that native entity's `description` field.
Clicking a formatted sequence returns that row to edit mode.

Chain IDs always follow the visible entity and copy order. Adding, removing,
resizing, or reordering entities recomputes one sequential Excel-style ID
series (`A` through `Z`, then `AA`, `AB`, and so on), closing any gaps. Changing
between polymer types preserves the entered sequence and revalidates it under
the selected type. Changing between a polymer and Ligand clears the
type-specific input. Rows use explicit arrow controls for reordering and do
not imply drag-and-drop support.

Expert Mode loads one self-contained native `alphafold3` JSON object, versions
1 through 4. Path-valued fields are rejected because browser-local paths have
no meaning to the API service. The frontend replaces the uploaded document's
`name` with the required form Job name. It parses the file in the browser and
submits ordinary JSON rather than multipart form data.

The AlphaFold3 validation route and standalone CLI accept a maximum 256 MiB
JSON document. Larger API bodies receive `413 payload_too_large` before JSON
parsing. The API does not add a streaming JSON parser; large Expert
requests therefore require API-process memory above their wire size.
The validation route admits at most two uploads at a time, streams request bytes
into temporary files, hashes them while writing, and runs JSON/Pydantic parsing
in one bounded background worker. Only successful validation atomically publishes the
24-hour resource. Retention is bounded to eight resources and 1 GiB per User,
64 resources and 8 GiB service-wide, while preserving at least 1 GiB of free
space on the state filesystem. Capacity exhaustion returns a typed `429` or
`507` before the memory-heavy parse.
Job names are limited to 120 characters. Pairformer recycles may be zero.
Cleanup, deletion, and the final validation claim share one short process-local
critical section so an admitted resource cannot be removed between reload and
claim. Reusing an idempotency key with a different validated request is a
conflict while that validation remains available. After successful staging
consumes the validation, the owner-scoped idempotency key is authoritative and
a lost-response retry returns the existing Job without validation lookup or
deployment preflight.

Both modes create the validation resource before proceeding to the same
confirmation page. This performs `AF3Config` parsing and the existing
Biomodals request validation needed to produce a bounded preview. It creates
no Job and calls no Modal API, but it atomically retains the validated document
and operational settings under the configured state directory for up to 24
hours. It does not introduce a second scientific validator.

The frontend also keeps the not-yet-validated form in browser-local IndexedDB.
After validation, the server resource restores the confirmation page after a
reload without transferring the document again. **Back to edit** and **Clear**
delete the validation resource and invalidate its idempotency key. Successful
Job submission consumes it.

The preview reports Job name; entity types, IDs, copies, and polymer lengths;
the exact seed list and count; samples and total prediction count; search and
recycle settings; separate counts of modifications, bonds, embedded custom
MSAs, custom templates, and custom CCD definitions; and validation warnings.
Ligands already appear in the entity list and are not counted again as an
advanced input. The page offers the exact parsed document as a JSON download
but does not render or copy the whole document,
because inline MSA and template content can be large.

If the request exceeds 5,000 seed/sample predictions, the required explicit
confirmation appears immediately above Submit. Entering confirmation creates
a pending idempotency UUID with `crypto.randomUUID()`. It survives reload and
duplicate clicks until admission succeeds. Returning to edit deletes the
validation and invalidates that key so the next reviewed payload receives a
new one.

Both modes expose the following settings in a collapsed Advanced section:

- search for missing MSAs for Protein and RNA entities, enabled by default;
- search for missing Protein templates, enabled by default;
- Pairformer recycles, default `10`;
- diffusion samples per seed, default `5`; and
- an explicit comma/range model-seed input, default `1`.

DNA entities require neither search. The confirmation page states which
entity types each enabled search setting actually affects.

Expert documents retain AlphaFold3's evidence-state semantics from UniAF3
0.2.1: null or omitted MSA/templates request the enabled searches, whereas
empty MSA strings and an empty template list explicitly disable those evidence
sources. Paired and unpaired Protein MSA states must be supplied together.

When normalized seed count multiplied by diffusion samples exceeds 5,000, the
frontend displays the prediction count and requires explicit confirmation.
The validation response marks that condition. Calling the Job-creation route
for such a validated request is the API-level acknowledgement, and the service
sets the app's internal `allow_large_inference` value while staging. The API
does not add a second confirmation field.

The initial Expert Mode UI is an upload surface rather than a second structured
editor for modifications, bonds, custom MSA, templates, or custom CCD content.
Those values are authored in the uploaded JSON. Regular Mode supports simple
CCD-code and SMILES Ligands; all entity types are also valid in uploaded Expert
documents. A future expert editor may offer an open CCD-code combobox with
suggestions rather than a closed built-in vocabulary.

AlphaFold3 exposes these ordered semantic stages:

1. Prepare input
2. Prepare environment
3. Search MSAs
4. Search templates
5. Predict structures
6. Prepare results

Their kernel Node mappings are:

| Semantic stage | Kernel Nodes |
| --- | --- |
| Prepare input | `stage-request-input` |
| Prepare environment | `prepare-environment` |
| Search MSAs | `raw-database-searches`, `combined-msa-publications` |
| Search templates | `protein-template-searches` |
| Predict structures | `stage-inference-input`, `seed-predictions` |
| Prepare results | `inference-summary`, `request-publication` |

Each stage may therefore aggregate several internal kernel Nodes. Provider Call
choices remain available through diagnostics and logs. Coordinator-local Nodes
have no log target.

## Job logs

The Remote Execution Client uses the Modal SDK's Function Call log APIs rather
than a subprocess or Modal CLI. Log responses are newline-delimited JSON. Each
record contains an optional provider timestamp, the unmodified message
including ANSI escapes, and its source. All sources are initially included in
provider order.

The frontend preserves ANSI styling, multiline messages, and compact
timestamps. Copy and download reconstruct readable text from the loaded
records. Historical windows remain cached for the lifetime of the page.

A semantic stage may have multiple log targets. The expanded stage shows a
compact selector labeled with function, status, and start time, preferring an
active call and then the most recently started call. Public responses use an
opaque log-target selector and never expose the Modal Function Call ID.
Target lookup is stage-filtered and bounded rather than paging every Provider
Call in the Run. Historical requests require timezone-aware start and end
times in ascending order and use windows of at most one hour. The backend opens
one SDK stream for each admitted live HTTP request. Modal's SDK preserves its
cursor while reconnecting its own RPCs; the frontend does not add a second
polling or reconnect loop. Historical output for any started, non-active stage
is fetched once and remains cached for the lifetime of the page.

An interactive terminal observation may persist `finalizing` and return
promptly. A background terminal observation continues directly into archive
preparation. Startup reconciles cache-presence markers with actual archives,
and a verified rebuild marks the Result cached. `completed_at` records when
service publication actually finishes; the stable `finalization_started_at`
remains the timestamp used for deterministic archive provenance.

Live reads use one Modal SDK stream each and are limited to 32 service-wide,
four per User, and four per Job. Excess requests receive a typed `429` with a
short `Retry-After`; historical window reads do not consume these permits.

Tool configuration controls whether logs are Administrator-only or also
available to the owning User. GROMACS initially allows owner access;
AlphaFold3 initially restricts logs to Administrators.

## Modal billing

Admin > Modal contains an Administrator-only Costs section backed by Modal's
billing report API. It shows total workspace cost, the cost attributed to the
effective Modal Environment selected under Admin > Modal > Environment, and
parallel Tool and Environment groupings. The current-environment summary uses
the selected report interval and displays zero when that Environment has no
reported usage.

Deployed Tool Apps carry a stable `biomodals_tool` tag (`gromacs` or
`alphafold3`). Billing reports request this tag and use it as the only reliable
Tool attribution key. Untagged or unknown values remain **Other / untagged**;
historical usage from before tagging is not guessed from object
descriptions. This fallback applies only to the Tool breakdown: every billing
row still contributes to its reported Environment total. Existing Apps must be
redeployed before future usage receives the tag.

The default interval is the current billing month. Presets include today, the
last seven days, last 30 days, and previous month, plus a custom date range. Normal
reports use daily resolution; intervals no longer than two days may use hourly
resolution.

Reports are cached for five minutes. A transient Modal failure returns the
failure explicitly with a Refresh action. Billing responses are not persisted
in `service.sqlite3`. Billing capability is optional and never affects
readiness, Tool validation, submission, or execution.

Modal exposes programmatic billing reports only to supported workspace plans.
An unsupported plan uses the same explicit unavailable state rather than a
synthetic zero-cost report.

## Frontend routing

GROMACS and AlphaFold3 retain contextual Job URLs while sharing one generic Job
detail component:

```text
/tools/gromacs/jobs/{job_id}
/tools/alphafold3/jobs/{job_id}
```

The component verifies that the Tool in the returned Job matches its route.
After implementation AlphaFold3 is an available catalog Tool rather than WIP.

## Deployment validation

Saving a Tool deployment version validates the exact deployed
`ExecutionCoordinator` configured by the startup App name. The coordinator
must expose launch, bounded status, cancellation, and cursor-paginated Provider
Call diagnostics. Failure names the missing or incompatible capability in the
existing validation popup and does not update the setting.

## Pre-release database cutover

The new service schema has no compatibility reader or shipped migration from
the service-local execution schema. Pre-release deployments initialize fresh
state. If development users and settings need to survive the local cutover, an
operator may copy only those rows once into a fresh database; this is not
product migration code. Sessions and old Jobs are not retained. Those Jobs
cannot be assigned truthful remote Execution Locators.

## Submission and recovery

The Service Job ID is also the kernel Execution Run ID. `service.sqlite3`
does not persist a second execution identifier, the submitted document,
operational settings, or a confirmation summary. The exact remote Deployment
Identity remains required because a Job ID alone cannot identify the Modal
App after Administrator configuration changes. Additional scientific request
details are recovered from the exact remote Run or its final archive.

GROMACS submission validates its direct request. AlphaFold3 submission verifies
that the named validation belongs to the User, remains unexpired, and has not
already been consumed. Submission then preflights the exact Tool deployment.
One transaction admits a queued Service Job, uses its Job ID as the Execution
Run ID, stores the Deployment Identity, full request digest, Tool-specific
publication-scope digest, and any pending validation reference, and commits
before any provider side effect.

After admission, the service returns the queued Job with `202` and wakes the
bounded background reconciler. The Tool Adapter then stages the immutable
request and launch identity, the Remote Execution Client spawns the deployed
coordinator, and the service records its root Function Call ID. The ordinary
60-second reconciliation interval remains a retry fallback; a newly admitted
Job does not wait for it. A staging failure before any spawn attempt leaves the
Job queued, and background processing retries the idempotent staging and launch
sequence.

Before staging AlphaFold3, the adapter lists only earlier Jobs with the same
Tool and publication-scope digest. AlphaFold3's scope excludes the Job name and
selected seeds but retains the biological input, search choices, recycles, and
samples per seed. If any matching Job is `queued`, `running`,
`cancel_requested`, `state_unknown`, or remotely blocked without a terminal
result, the new Job waits locally and retries after 60 seconds. A recorded root
Function Call is polled without waking coordinator code. Once that root
conclusively completes, a still-non-terminal remote Run status is recorded as
a terminal service failure rather than active work. Once no predecessor may
still be active, failed, partial, and cancelled predecessor Job IDs are staged
with the new request. They authorize AlphaFold3 to abandon only deterministic
Task generations derived from those Execution Run IDs plus stable Node and
Task keys. The new Job's
snapshotted Provider Call limits and other current arguments take precedence.
Successful publications are reused through their normal markers and receipts.

This policy does not add Task records to `service.sqlite3`, inspect remote
ledgers, or coordinate independently submitted CLI Runs. It handles compatible
API publication scopes only; broader overlap between different inputs remains
under the app's ordinary marker and conservative claim-expiry rules.

If coordinator spawn may have occurred but no Function Call ID was returned,
the Job becomes `state_unknown` with
`submission_outcome_unknown`. It is not automatically resubmitted. Exact
deployment preflight failure before admission rejects the request without
creating a Job. A queued Job cancelled before spawn becomes locally cancelled
and never creates a remote Execution Run. Cancellation racing an ambiguous
spawn retains cancellation intent and requires remote-state resolution.

Owner-visible projections never copy remote status messages, Node errors,
Modal exceptions, or storage paths. Failed Jobs use fixed service-defined
error copy; partial Runs use one generic incomplete-results warning. Raw remote
diagnostics remain available only through Administrator-authorized logs and
Provider Call inspection.

Each admitted live log read uses one Modal SDK stream. The SDK preserves its
log cursor while reconnecting internal RPCs and ends the stream when the
Function Call completes. Selecting a log target performs one direct bounded
coordinator read rather than scanning Provider Call pages.

A result-preparation-blocked Job is served from its local projection. Owner
detail and manual refresh do not wake the remote coordinator or toggle it back
to `finalizing`; the background reconciler alone retries transient
finalization when its recorded retry time is due. `result_integrity` has no
automatic retry time: an explicit download preparation may try again after an
operator restores the authoritative remote publication.

Large pending inputs are staged atomically outside SQLite. GROMACS uses
`<state-dir>/pending-inputs/<job-id>/`; AlphaFold3 validations use
`<state-dir>/validated-inputs/<validation-id>/`. A submitted validation is
claimed by at most one Job. Its 24-hour expiry no longer applies while that Job
still references it. After the service verifies immutable remote staging, it
removes the local pending copy and clears the reference. A pre-staging failure
retains the copy for retry or Administrator resolution; cancelling such a Job
removes it. Expired unclaimed validations and orphan directories are removed
automatically and remain visible in Administrator staging-usage totals until
cleanup.
