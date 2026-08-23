# API Tool service specification

Status: in design

Last updated: 2026-08-24

This specification applies ADR 0007 to the FastAPI service and its frontend.
It records accepted implementation behavior while the remaining AlphaFold3,
logging, and billing details are being designed.

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
  semantic stages, deployment defaults, Provider Call limits, and default log
  visibility;
- a `ToolAdapter` that validates and stages its request and retrieves its
  published Result; and
- a typed submission router.

The shared `RemoteExecutionClient` performs exact-version coordinator
preflight and resolution, launches Runs, polls root Function Calls, refreshes
bounded execution status, requests cancellation, pages Provider Calls, and
accesses their logs. Tool registrations do not contribute lifecycle callback
bags or background reconcilers.

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

## Semantic stages

A Tool Definition owns an ordered list of semantic stages. Each stage maps one
or more kernel Node keys into one user-facing label and optional provider
function description. The service aggregates Task status counts across mapped
Nodes. Provider Call selection and raw Node keys are diagnostic details.

## Provider Call limits

Each Tool has Administrator-configurable maximum active Provider Calls and
maximum active GPU Provider Calls. Both are positive integers and the GPU limit
cannot exceed the total. The effective values are snapshotted during Job
admission and supplied to the deployed coordinator. Later changes affect only
new Jobs.

Initial defaults preserve current behavior:

| Tool | Total | GPU |
| --- | ---: | ---: |
| GROMACS | 3 | 1 |
| AlphaFold3 | 4 | 1 |

The Administrator Tool table keeps active Jobs and the per-User active Job
limit separate from execution capacity. Total and GPU Provider Call limits
share one compact **Maximum containers** column with **Total** and **GPU**
fields. This layout refinement is lower priority than the execution behavior.

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

Regular Mode constructs a native AlphaFold3 JSON configuration from one or
more polymer inputs. Each entity row selects Protein, DNA, or RNA and contains
only its sequence, identical-copy count, and automatically assigned stable
uppercase chain IDs. It shows those IDs as read-only and defaults model seeds
to `1`. It does not expose ligands, modifications, covalent bonds, custom MSA,
custom templates, or user-provided CCD fields.

The polymer editor follows the supplied AlphaFold input interaction reference.
An empty entity is a compact horizontal row with a drag handle, an Entity type
dropdown containing Protein, DNA, and RNA, copy count, a large **Paste sequence
or FASTA** input, and row controls. After parsing, the same row presents the
normalized sequence in a wrapped monospace view with residue numbers every ten
positions. **Add entity** creates another row. Each row accepts exactly one raw
sequence or one FASTA record; multi-record FASTA is rejected with guidance to
add separate entities. Clicking the formatted sequence returns that row to
edit mode. Reordering rows does not change chain IDs already assigned to an
entity. Copies receive monotonically allocated Excel-style uppercase IDs (`A`
through `Z`, then `AA`, `AB`, and so on). IDs are not reused within an editing
session, so removing a copy or entity never renames another entity.

Each row exposes only the drag handle, Entity type, Copies, sequence, assigned
chain IDs, and a remove action. It has no overflow menu or collapse controls in
the MVP. Changing Entity type preserves the entered sequence and revalidates
it under the selected Protein, DNA, or RNA type.

Expert Mode loads one self-contained native `alphafold3` JSON object, versions
1 through 4. Path-valued fields are rejected because browser-local paths have
no meaning to the API service. The frontend replaces the uploaded document's
`name` with the required form Job name. It parses the file in the browser and
submits ordinary JSON rather than multipart form data.

The AlphaFold3 validation route accepts a maximum 256 MiB JSON document.
Larger bodies receive `413 payload_too_large` before JSON parsing. This is an
API-service upload limit rather than a change to the AlphaFold3 app's existing
1 GiB CLI ceiling. The API does not add a streaming JSON parser; large Expert
requests therefore require API-process memory above their wire size.
The validation route streams request bytes into a temporary file, hashes them
while writing, and runs JSON/Pydantic parsing in one bounded background worker.
Only successful validation atomically publishes the 24-hour resource. This
serializes the memory-heavy parse without blocking unrelated event-loop work;
concurrent uploads may still progress.

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
recycle settings; counts of modifications, bonds, ligands, custom templates,
and custom inputs; and validation warnings. The page offers the exact parsed
document as a JSON download but does not render or copy the whole document,
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

When normalized seed count multiplied by diffusion samples exceeds 5,000, the
frontend displays the prediction count and requires explicit confirmation.
The validation response marks that condition. Calling the Job-creation route
for such a validated request is the API-level acknowledgement, and the service
sets the app's internal `allow_large_inference` value while staging. The API
does not add a second confirmation field.

The initial Expert Mode UI is an upload surface rather than a second structured
editor for ligands, modifications, bonds, custom MSA, templates, or CCD
content. Those values are authored in the uploaded JSON. Unmodified Protein,
DNA, and RNA entities are also valid in uploaded Expert documents. A future
expert editor may offer an open CCD-code combobox with suggestions rather than
a closed built-in vocabulary.

AlphaFold3 exposes these ordered semantic stages:

1. Prepare input
2. Search sequence databases
3. Search templates
4. Predict structures
5. Prepare results

Their kernel Node mappings are:

| Semantic stage | Kernel Nodes |
| --- | --- |
| Prepare input | `stage-request-input` |
| Search sequence databases | `raw-database-searches`, `combined-msa-publications` |
| Search templates | `protein-template-searches` |
| Predict structures | `stage-inference-input`, `seed-predictions` |
| Prepare results | `inference-summary`, `request-publication` |

Each stage may therefore aggregate several internal kernel Nodes. Provider Call
choices remain available through diagnostics and logs. Coordinator-local Nodes
show `N/A` as their running function and have no log target.

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

Tool configuration controls whether logs are Administrator-only or also
available to the owning User. GROMACS initially allows owner access;
AlphaFold3 initially restricts logs to Administrators.

## Modal billing

Admin > Modal contains an Administrator-only Costs section backed by Modal's
billing report API. It shows total workspace cost, Environment grouping,
configured Tool App costs, and unmatched usage as Other workspace usage.

Deployed Tool Apps carry a stable `biomodals_tool` tag (`gromacs` or
`alphafold3`). Billing reports request this tag and use it as the only reliable
Tool attribution key. Untagged or unknown values remain **Other workspace
usage**; historical usage from before tagging is not guessed from object
descriptions. Existing Apps must be redeployed before future usage receives
the tag.

The default interval is the current billing month. Presets include the last
seven days, last 30 days, and previous month, plus a custom date range. Normal
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

Saving a Tool App name or deployment version validates the exact deployed
`ExecutionCoordinator`. The coordinator must expose launch, bounded status,
cancellation, and cursor-paginated Provider Call diagnostics. Failure names
the missing or incompatible capability in the existing validation popup and
does not update the setting.

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
Run ID, stores the Deployment Identity, request digest, and any pending
validation reference, and commits before any provider side effect.

After admission, the Tool Adapter stages the immutable request and launch
identity, the Remote Execution Client spawns the deployed coordinator, and the
service records its root Function Call ID before returning `202`. A staging
failure before any spawn attempt leaves the Job queued; background processing
retries the idempotent staging and launch sequence.

If coordinator spawn may have occurred but no Function Call ID was returned,
the Job becomes `state_unknown` with
`submission_outcome_unknown`. It is not automatically resubmitted. Exact
deployment preflight failure before admission rejects the request without
creating a Job. A queued Job cancelled before spawn becomes locally cancelled
and never creates a remote Execution Run. Cancellation racing an ambiguous
spawn retains cancellation intent and requires remote-state resolution.

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
