# MVP release readiness

Status: accepted

This specification records the work that must be resolved before BioModals is
presented as a production MVP. It covers the FastAPI service, the static
frontend, their shared contracts, and the single-host deployment boundary.
The review questions are resolved, and this document is the implementation
contract for the pre-release MVP work.

## Settled constraints

### Deployment boundary

The MVP is an internal department service restricted to the trusted private
network. It runs on one Linux host with one FastAPI worker, local SQLite state,
and separately deployed Modal compute Apps. A publicly resolvable hostname does
not authorize public network access.

Internet-facing deployment is outside this release. It requires a new review
of identity, login throttling, abuse controls, monitoring, and scaling rather
than silently inheriting the private-network risk acceptances.

### Backend configuration-file security

When `BIOMODALS_API_CONF_ENV` names a configuration file, the backend must
accept only a regular file owned by the service user with no group or world
permissions. Mode `0600` is the normal deployment setting. Startup fails before
loading service credentials when the ownership or permissions are unsafe.

Explicit process environment variables remain supported and override file
values. Setup documentation must create a private file directly, for example
with `install -m 600`, rather than copying the public example with default
permissions.

Full service startup through `biomodals api serve` requires both Modal token
variables and performs the configured deployed-resource preflight before
accepting traffic. Offline `biomodals api admin` account commands instead load
only the shared configuration needed by that command. They use the same state
path and transactional account invariants but do not require Modal credentials,
construct a Modal client, start reconciliation, or run deployed-resource
preflight. Password Link creation still requires the configured public URL.

The service reads the selected configuration file and process environment once
at startup; it does not watch or hot-reload them. Editing a file-controlled
value or credential requires an API restart. The restarted process repeats all
startup validation and deployed-resource preflight before readiness returns.
Database-backed Administrator Runtime Settings remain live. Administrator
responses report values loaded by the running process, never newer contents
that merely exist on disk.

### State and cache path resolution

When `BIOMODALS_API_CONF_ENV` is configured, relative values of
`BIOMODALS_STATE_DIR` and `BIOMODALS_CACHE_DIR` resolve against the directory
containing that configuration file. The same base applies when a process
environment variable overrides one of those file values, so the server and
Admin CLI cannot select different databases merely because they start in
different working directories.

Without a configuration file, relative paths retain current-working-directory
semantics for local development. Production deployment must configure absolute
state and cache paths.

### Frontend deployment

`beta.biomodals.example.com` is a development-only Vite site and is not a production
release target. The production frontend is built into a static directory at
`/srv/biomodals.example.com` and served at `https://biomodals.example.com`.

The production server must provide SPA fallback for browser routes, proxy
same-origin `/api/*` requests to FastAPI, cache hashed assets immutably, and
serve `index.html` without persistent caching. Development-only Vite modules,
React Refresh, and HMR endpoints must not be present on the production site.

Production and non-local development use HTTPS and current evergreen browsers.
Submission idempotency keys always come directly from
`crypto.randomUUID()`. The frontend does not add a `getRandomValues` helper,
`Math.random()` fallback, UUID dependency, or non-secure-origin compatibility
path. Local development uses `localhost` or the configured HTTPS beta origin;
an arbitrary HTTP hostname is outside the supported browser boundary.

### Public URL and cookie mode

Backend startup must require the configured public URL scheme and cookie mode
to agree. An HTTPS public URL requires secure `__Host-` session cookies; a local
HTTP development URL requires non-secure development cookies. Either mismatch
is a configuration error and stops startup.

Logout must expire the session and CSRF cookies with attributes matching their
creation attributes, including `Secure` for the production session cookie.

### Production browser-security headers

Production Caddy configuration must provide a same-origin Content Security
Policy with framing and plugins disabled, `X-Content-Type-Options: nosniff`,
`Referrer-Policy: strict-origin-when-cross-origin`, and a restrictive
Permissions Policy.

`biomodals.example.com` must send HSTS, but the MVP policy must not use
`includeSubDomains` or preload until every affected subdomain has been audited.
These are deployment requirements; the application repositories do not own the
live Caddyfile.

### Disposable pre-release state

SQLite state remains disposable during the current active-development phase,
and destructive schema resets are allowed. Backup implementation and restore
drills are therefore outside the current implementation cycle.

The backend never deletes, migrates, or destructively rewrites an incompatible
database automatically. It reports the version and configured database
location, then exits. During active development an Administrator must stop the
service and explicitly migrate selected records or remove that exact database
before restart initializes the current schema.

Pre-release and production must not share service configuration or host-local
mutable state. Every pre-release service definition explicitly selects
`BIOMODALS_STATE_DIR` and `BIOMODALS_CACHE_DIR` locations distinct from
production. Its public URL is `https://beta.biomodals.example.com`, while production
uses `https://biomodals.example.com`. This also prevents a pre-release Clear Result Cache
action from touching production files. Reset instructions name only the
pre-release target; they must not operate on production configuration, state,
or cache.

The Modal Environment remains an independent explicit value in each
configuration and may be `production` for pre-release when intentionally
authorized. Local-state isolation does not silently select or create a Modal
Environment.

This exception ends before real production Users are onboarded. At that point,
a SQLite-aware backup and verified restore policy becomes a production launch
gate; the rebuildable Result cache remains excluded from backup.

### Deployment documentation only

The current implementation cycle must add a production runbook and example
native and container service definitions. They must document a dedicated
service user, private service configuration, absolute state and cache paths,
one API worker, an explicit working directory where applicable, restart policy,
logging, and safe update and rollback checks. The native systemd example also
uses a restrictive umask and journald.

These artifacts are examples only. This cycle must not install a unit, modify
the live production host, start a production service, or deploy the production
frontend.

### Liveness and readiness

`GET /api/v1/health` is a minimal liveness probe: it returns `200` when the API
event loop can serve a request and performs no database, filesystem, or Modal
work. `GET /api/v1/ready` returns `200` only after startup has completed and a
local SQLite check, configured cache-storage check, bounded artifact worker,
and Job reconciler are usable; otherwise it returns `503`.

Readiness never contacts Modal. Startup and Administrator configuration
preflight own that validation, avoiding an external-provider flap on every
probe. Both endpoints are unauthenticated, expose only a simple status rather
than paths or configuration, and are part of OpenAPI. The deployment runbook
waits for readiness before directing Users to the candidate service.

### Operational correlation and logging

The API generates a request identifier for every HTTP request, returns it in
`X-Request-ID`, and includes it in related journald records. OpenAPI documents
the response header. Unexpected frontend API failures show that identifier so
an Administrator can correlate the report; ordinary validation and expected
recovery messages do not add support noise.

Service records use consistent searchable fields such as `event`, `job_id`,
`workload`, public stage code, and safe Blocking Category. They cover Job
admission and idempotent replay, stage transitions, Cancellation, finalization
retry/block/recovery, Runtime Setting changes, Result Cache cleanup, and
readiness transitions. They never contain Password Links, session or CSRF
values, request bodies, PDB content, Modal credentials, or secrets.

The MVP continues using local journald and adds no external metrics, tracing,
error-reporting, or audit-log service.

### Change-aware deployment guidance and real Modal smoke

Before an Administrator performs a deployment, the assisting agent asks the
User for exactly two last-deployed Git commit hashes: one for the backend
repository and one for the frontend repository. The backend hash is also the
source baseline for the GROMACS App in that repository. The agent verifies the
hashes locally and compares each candidate with its baseline, then provides an
explicit, ordered checklist tailored to the changed surfaces. The checklist
names prerequisites, configuration and preflight checks, deployment order,
expected observations, failure stop conditions, targeted smoke checks, and
rollback steps. It must distinguish mandatory steps from checks that are
irrelevant to that Change Set.

The MVP does not add a deployment manifest, deployment-history service, or
database table for this purpose. If there has been no prior deployment, the
User says so and the agent treats the whole candidate as the Change Set.

When the GROMACS App, the backend Modal adapter, or effective Modal
configuration changes, the checklist includes one real-Modal smoke test. The
Administrator—not the agent—manually confirms the cost and submits the smallest
valid Job through the normal deployed Function sequence. The Administrator
verifies one invocation of each expected Function, accurate Job stage
reporting, successful finalization, and the Result archive allowlist. The run
uses an unmistakable smoke-test display name. This paid check is never part of
routine CI and is not performed automatically during implementation or
deployment.

An unexpected invocation, stage, archive member, or permission/configuration
error stops the procedure. The checklist must direct the Administrator to
capture the safe identifiers and observations needed for diagnosis before
retrying; it must not suggest blind resubmission of paid work.

### Recoverable finalization failures

The public Job Status vocabulary adds `blocked`. A Job becomes blocked when
scientific compute is preserved but finalization cannot continue until an
Administrator repairs a permanent service configuration, credential, or
permission problem.

Transient publication failures remain `finalizing` and retry with bounded
backoff. A blocked Job does not consume User, Tool, or Global Active Job Limits.
After repair, resuming it returns the same Job to `finalizing` and reuses its
existing outputs; scientific compute must never be submitted again as part of
that recovery.

Recovery is automatic and uses a persisted exponential retry schedule capped at
approximately 15 minutes. It retains the Job's original Modal Configuration
Snapshot and retries only finalization. Restarting the API resumes
reconciliation. Administrators may see blocked counts and safe blocking
categories, but do not gain access to another User's Input, Result, or private
Job detail. The MVP has no per-Job rerun or unblock action.

Authentication, permission, invalid service configuration, and missing
configured Modal resources block immediately. Connection, internal-service,
resource-exhaustion, and upload-timeout failures remain `finalizing` for a
persisted 30-minute retry window before becoming blocked. Local cache or
staging filesystem errors follow the same window and then use safe category
`local_storage`; they never redefine valid remote scientific output as an
invalid Result. Missing, corrupt, or
scientifically invalid required output instead produces terminal
`failed/result_invalid`. A blocked Job may remain blocked indefinitely while
low-frequency automatic finalization retries continue.

Owner-visible Job detail supplies generic blocked copy, `blocked_at`, and
`next_retry_at`. Existing terminal `error_code` and `error_message` fields
remain exclusive to failed Jobs. The Admin Modal page exposes only aggregate
blocked counts grouped by safe Blocking Category and the oldest blocked age;
it exposes no owner identity, Job identifier, Input, Result, raw provider
detail, or storage path.

### Unknown remote execution state

The public Job Status vocabulary adds `state_unknown` for cases where remote
work may exist but BioModals no longer has enough durable provider state to
track it safely. This is distinct from `blocked`: a blocked Job has known
scientific output and retries only recoverable finalization, while a
state-unknown Job may still be consuming paid remote compute.

A Job enters `state_unknown` when either:

- a direct Modal `.spawn()` may have been accepted but its Function Call ID
  could not be durably recorded; or
- Cancellation cannot be confirmed because the known call status expired, and
  a verified final Result cannot be recovered.

An explicit ambiguous submission outcome enters the state immediately. If the
API process stops after acquiring a submission lease, the restarted reconciler
waits for that short lease to expire before entering the state. Neither path
automatically submits another Function. The Job is excluded from automatic
reconciliation but continues consuming User, Tool, and Global Active Job Limits
until it is resolved.

Owner-visible Job detail labels the state `Status unknown`, explains that an
Administrator must review Modal, exposes `state_unknown_at`, and provides no
Cancel, Download, or Start Again action. It does not automatically poll because
only an Administrator mutation can resolve the state; focus, page reload, and
manual Refresh still load the current record. The latest recorded Stage remains
visible without a spinner or invented outcome.

The Admin Modal page exposes a dedicated list containing only Job ID, workload,
display name, safe run name, `state_unknown_at`, and one of the fixed reasons
`submission_outcome_unknown` or `cancellation_outcome_unknown`. It does not
expose owner identity, Input, Result, Function Call ID, raw provider exception,
or storage path. The Administrator must inspect Modal and stop remote work there
first when necessary. The only MVP resolution is a confirmed destructive
`Mark failed` action. It records terminal `failed/compute_failed`, closes any
still-open Stage as failed, preserves the unknown-state timestamp and reason for
audit, and releases admission capacity. The action does not contact Modal and
cannot be undone in the Admin panel.

### Modal configuration preflight

Changing the effective Modal Environment must preflight the output Volume and
every required GROMACS Function at the configured deployment version in that
Environment before committing. Each Tool has a positive integer Modal App
version alongside its App name. Changing either field preflights the candidate
name/version pair in the current effective Environment. Limit-only changes do
not validate unrelated fields. Startup performs the same read-only preflight
for process/file-controlled settings. Validation never invokes a paid
Function, and failure leaves the previous effective setting intact with a
stable configuration error.

Job admission snapshots the Modal Environment, App name, and exact deployment
version in one transaction. Every direct deployed-Function lookup for that Job
passes the snapshot as Modal's `version` argument, so an App redeployment cannot
mix Function implementations within an in-flight workflow. Administrators use
`modal app history <app> --env <environment> --json` to discover deployment
version integers; the Admin Tool form configures and restores the value with
the same field-specific source behavior as App name and limits. Result
provenance records both the snapshotted deployment version and the GROMACS
software version read from the production TPR header.

The Admin interface displays a spinner and disables save/restore controls only
for fields participating in a pending preflight. Unrelated fields retain their
own pending state and provenance.

### Restart-safe GROMACS analysis outputs

The established GROMACS App must treat each required RMSD,
radius-of-gyration, and RMSF CSV/PNG pair as independently recoverable output.
After an interruption, it regenerates only missing or stale members and commits
the complete pair coherently.

This is a narrow App reliability correction that benefits both the Local
Entrypoint and API callers. The API service must not generate scientific plots,
change deployed Function implementations or scientific arguments, or introduce
an API-only coordinator.

### GROMACS Job timeline

The public timeline represents only work directly orchestrated by the API. It
uses these stable stage codes, labels, and Running Function values in order:

| Stage code | User-facing label | Running Function |
| --- | --- | --- |
| `prepare_simulation` | Prepare simulation | `prepare_tpr_cpu` or `prepare_tpr_gpu` |
| `analyze_nvt` | Analyze NVT | `collect_traj_stats` |
| `analyze_npt` | Analyze NPT | `collect_traj_stats` |
| `run_production` | Run production | `production_run_cpu` or `production_run_gpu` |
| `analyze_production` | Analyze production | `collect_traj_stats` |
| `prepare_result` | Prepare result | none; local service work |

These values are fixed by the GROMACS workload definition, not by the shared
Job schema. OpenAPI exposes stage codes and Running Function names as strings
so another workload can define its own timeline without changing the common
Job contract.

Preparation, minimization, NVT, and NPT execution occur inside the selected
`prepare_tpr_*` Function and are not presented as separately observable
stages. Likewise, nested implementation calls such as `postprocess_traj` are
not public timeline rows. Each direct Function stage starts when its call is
durably attached and ends when the backend records its observed terminal
outcome. `prepare_result` spans local finalization through Result publication.
Every started stage exposes `started_at`, nullable `ended_at`, and a nullable
outcome of `completed`, `failed`, or `cancelled`; an active, state-unknown, or
blocked stage has no end or outcome. OpenAPI exposes all current rows through
`active_stages`; the singular `stage` remains a compatibility summary. The Job
table may therefore show several Running rows with overlapping timestamps. It
exposes no invented substage, timestamp, or outcome.

The durable dependency graph is:

```text
prepare_simulation
  |-> analyze_nvt --------------------------------|
  |-> analyze_npt --------------------------------|
  `-> run_production -> analyze_production -------|
                                                   `-> prepare_result
```

The three branches after preparation run concurrently. Production analysis may
start while NVT or NPT analysis is still running. Result preparation starts only
after all three analysis stages complete. A definite branch failure stops new
dependencies, requests cancellation of active siblings, and reaches `failed`
only after every known remote call is inactive. If the service cannot confirm a
sibling's state while stopping it, the Job becomes `state_unknown` and keeps its
admission capacity until an Administrator resolves it.

### Administrator Job logs

An enabled Administrator can expand any started remote Stage directly inside
the Execution stages table. Every row is collapsed by default and only one row
is expanded at a time. Ordinary Users see the same stage table without log-row
interactivity and cannot use the diagnostic endpoints.

The target response contains Job ID, Stage code, Running Function, operation
state, live-or-historical mode, start time, and nullable end time. It never
contains a Modal Function Call ID. The selected Stage code is resolved again
against current durable state before opening logs. Running and `state_unknown`
Modal operations use live mode; completed, failed, and cancelled Modal
operations with retained call IDs and completion times use historical mode.
Local Result preparation and unsubmitted operations are excluded. An ineligible
selection returns typed `409 job_log_target_unavailable`.

The GROMACS registration opens live logs with Modal's supported
`modal app logs --follow --function-call --timestamps` CLI command. Historical
logs use the same call filter and timestamps without follow mode, bounded by the
operation's recorded start and end times. Both use the App and Environment
snapshot stored on that Job. The backend terminates a live CLI process when the
browser collapses the row, opens another row, navigates away, or otherwise
closes the HTTP response. A terminal Stage's first successful fetch is cached
for the lifetime of that Job-detail page, so collapsing and reopening it does
not contact Modal again; refreshing the page permits a fresh fetch. Target
metadata refreshes every ten seconds only while the selected target is live.

The log viewport has a fixed maximum height. Provider timestamps, when present,
are separated from monospace messages. Copy and Download controls operate on
the retained text; downloads use
`<current-timestamp>_<tool>_<stage>.log`. The frontend keeps at most the latest
500,000 characters and marks when earlier output was omitted. Empty and failed
fetches display diagnostic guidance instead of silently presenting a blank row.

The selected Function Call ID is an internal filter, not browser data. The
backend redacts that exact identifier from provider output even when it is split
across stream chunks. Modal deployment versions select Function versions within
the named deployed App; filtering on the durable call ID selects the originating
invocation without a separate logs-version argument. Deleting and recreating an
App under the same name can make an in-flight Job's diagnostic logs unavailable,
but cannot redefine the Job lifecycle or justify resubmission.

Provider logs are fallible Administrator diagnostics. They do not determine or
advance Job Status, Stage History, Progress, Cancellation, or Result validity,
and an empty or interrupted stream does not imply that remote work stopped.

The Job page displays the last recorded update but never derives a stale,
stalled, or failed state from elapsed time: a deployed Function can
legitimately run for hours without changing `updated_at`, and the API has no
heartbeat contract. Only an actual refresh failure displays "Unable to refresh
job status." The frontend then retains the last known state, leaves manual
Refresh available, and continues its normal polling cadence. Only the backend
changes Job Status.

### Workload and Catalog registration

Each executable API workload has one fixed descriptor owning its stable key,
User-facing Tool name, Runtime Setting environment-variable names, and mapping
from durable operations to public timeline stages. Runtime configuration,
Admin Tool rows, routing registration, and Job views consume that descriptor
instead of carrying separate GROMACS name and stage tables. The descriptor does
not make scientific orchestration generic: GROMACS keeps its own adapter,
request schema, sequencing, archive builder, and tests. All executable workload
routes do share the operation-scoped Modal submission state machine so
idempotency and ambiguous paid-call outcomes cannot drift between Tools.

The frontend Catalog separately includes an AlphaFold3 placeholder marked
`WIP`. Its card is visibly muted, is not an interactive navigation target, and
cannot submit a Job. The backend does not register an AlphaFold3 workload,
route, Modal App, configuration row, or speculative scientific contract until
that workflow is designed and deployed.

### Durable Cancellation

Accepting Cancellation persists `cancel_requested_at` and moves the Job to
`cancel_requested` before contacting Modal. The reconciler must not start
another stage after that transition. For every known active direct call it
requests termination with `terminate_containers=False`, retries transient
provider failures, and resumes the same work after an API restart. This mode
cancels inputs without forcibly terminating workers that may contain unrelated
inputs.

The Job becomes `cancelled` only after Modal confirms every active call is
inactive. Calls that finish first retain their completed outcome, but the
backend starts no successors and continues cancelling their siblings. If a
complete Result was already published, its prior `succeeded` or `partial` state
wins the race. The backend never reports `cancelled` merely because a timeout
elapsed.

`cancel_requested` continues consuming User, Tool, and Global Active Job Limits
until its remote outcome is known. After 15 minutes, Job detail shows
"Cancellation is taking longer than expected" using the persisted request
timestamp; the warning does not change status or stop reconciliation. The
Cancellation timestamp and coded `409 job_not_cancellable` response are part
of OpenAPI. If Modal's call status expires before cancellation can be confirmed,
the backend recovers a verified final Result when possible; otherwise it moves
the Job to `state_unknown` for manual Administrator review.

### Immutable Result archive identity and layout

When the final deployed Function completes, the backend persists
`finalization_started_at` exactly once as part of the transition into
`finalizing`. Archive provenance uses that value as its completion timestamp;
publication retries and later reconstruction never substitute their current
time. Together with fixed ZIP metadata and ordered members, unchanged remote
outputs must therefore reproduce the same bytes, size, and SHA-256 digest.
Every archive member uses ZIP's stored method so byte identity does not depend
on the host zlib implementation.

Each completed Job also persists the positive Result archive schema version
used for its publication. Cache reconstruction dispatches only to a retained
builder for that exact Tool/schema pair; it never rebuilds an older Result with
the current writer by assumption. If that builder is unavailable, the Result
moves to `blocked/result_integrity` and remains recoverable from its published
Modal ZIP.

The GROMACS Result ZIP has exactly three top-level entries or namespaces:
`input.pdb`, `outputs/`, and `metadata/`. `outputs/` contains the files useful
to an end User: the no-PBC trajectory, centered structure, production topology
and parameters, and each CSV/PNG analysis pair. Debugging and verification
documents live under `metadata/`, including normalized parameters, safe
provenance, the service run log, manifest, and checksums. No such document is
left loose at the archive root.

The fixed metadata allowlist is:

- `metadata/parameters.json` for normalized Submission parameters;
- `metadata/provenance.json` for safe Job, Tool, deployed App, software-version,
  and persisted timestamp facts;
- `metadata/stages.json` for public stage codes, deployed Function names,
  started/ended timestamps, and terminal stage outcomes;
- `metadata/run.log` for the service-generated lifecycle summary;
- `metadata/manifest.json` and `metadata/checksums.sha256`; and
- `metadata/gromacs/` for the existing text logs from minimization, NVT, NPT,
  and production plus the small minimization/equilibration `.mdp` files.

The submitted Input, required end-User outputs, every required CSV/PNG analysis
pair, and all service-generated metadata documents are mandatory at initial
publication; absence or an empty mandatory file produces terminal
`failed/result_invalid`. Publication also performs inexpensive structural
checks for PDB, MDP, XTC, TPR, CSV, PNG, and JSON members so a checksummed but
obviously truncated archive cannot become a successful Result.
Allowlisted GROMACS logs and `.mdp` files are diagnostic and included only when
present. Their absence does not invalidate an otherwise complete Result. The
manifest enumerates the exact optional members included.

This remains an explicit allowlist. The metadata directory must not become a
recursive dump of provider state, working files, credentials, internal paths,
database records, or raw exceptions. It excludes equilibration trajectories,
`.trr`, `.edr`, `.cpt`, and intermediate `.tpr` and structure files. Deeper
diagnosis uses the authoritative Modal Volume rather than expanding every User
download.

On a local cache miss, the backend first restores the published Modal ZIP and
completion marker. If either is missing or corrupt, it deterministically
reconstructs from the allowlisted raw Volume outputs and persisted provenance
timestamp. It may republish only when the rebuilt size and SHA-256 equal the
Job's recorded Result. It never substitutes different bytes or reruns
scientific compute.

If exact reconstruction is impossible or produces another digest, a previously
`succeeded` or `partial` Job moves to `blocked` with safe Blocking Category
`result_integrity`. Restoring the exact archive returns it to its prior
completed state. This is the sole recovery transition out of a normally
terminal completed state, and it does not consume Active Job Limits.

After first publication, both mandatory and optional membership are immutable.
A later reconstruction cannot silently add a newly available diagnostic file
or omit one that was originally present, because either action changes the
recorded Result digest.

### Unified Result staging and cache storage

Result staging lives under the configured cache directory on the same
filesystem as completed local Result archives. A successfully published staged
archive becomes the local cache entry instead of being deleted and downloaded
back from Modal.

There is no per-Result ceiling and no automatic size-driven cache eviction.
Every finalized Job persists its Result byte size and whether a local cached
copy is currently present. The Admin Storage page separately shows total
published Result bytes, completed local cache bytes, active staging bytes, and
filesystem free space.

Storage metrics load on page entry, refetch on focus, and provide manual
Refresh plus a last-updated time. They do not poll periodically because totals
may require filesystem work and are not time-critical. Cache cleanup and a
successful Result cache fill in the same frontend session invalidate the
snapshot. Changes made by another browser appear on focus or manual Refresh;
the MVP adds no push channel.

A process/file-controlled soft-warning threshold defaults to 1 TiB. The warning
uses completed local cache plus active staging bytes and does not reject or
delete data. `Clear cache` removes every unleased completed archive, marks its
Job as not locally cached, and reports the entries and bytes reclaimed. Later
reconstruction marks that Result cached and includes it in totals again. Active
staging and leased downloads are protected. Abandoned staging files are cleaned
automatically and included in disk accounting. Startup reconciles cache-presence
markers against actual files.

Local cache cleanup never deletes a Job or authoritative Modal Volume data. A
later download restores or reconstructs the Result from Modal without replaying
scientific compute.

### Result-work responsiveness

Unbounded Result size must not allow whole-file work to monopolize the single
FastAPI event loop. ZIP validation, whole-file SHA-256 verification, cache
reconciliation, and large directory scans run through one bounded artifact
worker thread. Modal byte streaming remains asynchronous and yields between
chunks.

Cache fills coordinate with per-Job locks so one miss does not block unrelated
downloads. This does not introduce a separate worker service or task queue. A
large synthetic archive test must demonstrate that health, login, Job polling,
and Cancellation remain responsive during Result validation.

### Prepared Result downloads

The Job detail page does not navigate directly into a potentially slow cache
restore. Its Download action first makes an idempotent, CSRF-protected
`POST /api/v1/jobs/{job_id}/prepare-download`. The page displays an
indeterminate `Preparing download...` spinner and disables duplicate actions
while the backend restores or reconstructs and verifies the immutable archive
through the per-Job cache-fill lock.

A successful preparation returns `204`, after which the frontend immediately
starts the ordinary authenticated GET download so the browser streams the ZIP
without a JavaScript Blob. A coded failure remains on the Job page, which
refetches state when integrity recovery may have moved the Job to `blocked`.
Aborting one browser request does not cancel or corrupt a cache fill shared with
another waiter; a later preparation joins or safely retries that work. Both
operations and their errors are part of OpenAPI.

After preparation, the cache holds a short-lived one-download reservation.
`Clear cache` treats that reservation like an active lease, so cleanup cannot
remove the archive between the `204` response and the browser's immediate GET.
The GET consumes the reservation when it acquires its streaming descriptor;
an abandoned reservation expires automatically.

The GET response uses a safe `Content-Disposition` filename derived only from
the sanitized display name: `<display-name>-results.zip`, with
`gromacs-results.zip` when no usable name remains. It exposes no Job UUID,
Modal run name, or storage path. Repeated-download suffixing is left to the
browser. Result identity continues to come from the manifest and recorded
size/SHA-256 rather than the friendly filename.

### User Status and admission authorization

The account contract replaces the ambiguous active flag with User Status:
`pending_setup`, `enabled`, or `disabled`. Provisioning starts in pending
setup; successful Password Setup enables the User. A disabled User cannot
authenticate, consume Password Links, or create Submissions. Re-enabling
returns a User with a password to enabled and one without a password to pending
setup.

The Admin Users table separates the editable display name from the immutable,
normalized email used for login. `PATCH /api/v1/admin/users/{user_id}` accepts
the optional trimmed display name with the same 120-character maximum as User
creation; it never changes email or login identity. The OpenAPI contract and
generated frontend client carry that field.

Disabling a User does not cancel already admitted Jobs. They retain their owner
and continue according to their existing lifecycle. They consume applicable
Active Job Limits, including while `state_unknown`, until terminal or blocked.
Their Results are inaccessible while the User is disabled and become available
again after re-enabling.
Administrator access does not grant access to those Jobs. The Disable
confirmation states these consequences before committing the status change.

Administrator role remains independent, but only an enabled Administrator
satisfies the last-administrator safeguard. Job admission rechecks enabled
status inside the same SQLite transaction that applies idempotency and Active
Job Limits, including before returning a same-payload idempotent replay. A
concurrent disable therefore prevents that request from claiming an
unsubmitted Job's provider lease and initiating paid work.

### Admin Active Job capacity display

The Admin Modal Tool contract exposes `active_jobs`, not `running_jobs`. It
counts the same `queued`, `running`, `finalizing`, `cancel_requested`, and
`state_unknown` Job states consumed by Active Job Limits; `blocked` and terminal
Jobs are excluded. The table heading is `Active jobs / active job limit`, so its
numerator and denominator always describe the same admission-capacity rule.

Every User, Tool, and Global Active Job Limit accepts a non-negative integer,
including zero. An Administrator may lower a limit below its current count;
existing Jobs continue unchanged, and no Job is cancelled or paused. The
resulting over-limit display, such as `5 / 2`, is valid and carries the warning
"Over limit; new jobs are blocked" rather than a validation error. New
Submissions remain blocked until the applicable count falls below its limit.
Setting a limit to zero therefore pauses new Submissions within that scope
without disabling a User or affecting admitted Jobs.

All Active Job counts and limits are local to one BioModals backend and its
SQLite database. In particular, Global means all Users and Tools admitted by
that deployment; it is not a combined limit across beta, production, or the
Modal account. When pre-release intentionally shares production's deployed
Modal App and Environment, the two services do not coordinate counters. The
pre-release configuration example sets its User, Tool, and Global defaults to
`1` to bound testing cost. Provider-account capacity enforcement or a shared
cross-deployment coordinator is outside the MVP.

Concurrent same-field editing by multiple Administrators is outside the MVP.
Field-specific PATCH requests continue to prevent one stale form from
overwriting unrelated settings, but the contract adds no ETag, revision, or
conflict dialog. Ordinary last-commit-wins behavior is acceptable for the
private deployment's small Administrator population.

The Admin Modal page refreshes this operational snapshot every 60 seconds while
visible, backs off to every 5 minutes while the document is hidden, and
refetches on focus and after a successful setting mutation. It also provides
manual Refresh and a small last-updated indicator. A count refresh must not
overwrite unsaved Environment or Tool form values. The existing combined Modal
Admin endpoint remains sufficient; the MVP adds no push or streaming channel.

### Administrator Password Link handoff

Creating a User or issuing a replacement Password Link opens one focused
Administrator dialog tied to that User. The dialog identifies the display name
and email, shows the read-only one-time link, provides a Copy button with
visible copied feedback, and retains the plaintext only in page memory until
the dialog closes. Closing it clears the plaintext; the frontend never stores,
refetches, or reconstructs the link.

The create-user and replacement-link responses include an absolute
`expires_at` timestamp in OpenAPI alongside the one-time URL. The dialog shows
that timestamp in the Administrator's locale plus "Valid for approximately one
hour." It does not run a countdown. The Admin CLI likewise prints the
expiration beside the URL while emitting the URL itself only once. Expiration
metadata remains non-secret and does not permit the link to be refetched.

Before a replacement is issued, the interface warns that every earlier
Password Link for that User will become invalid. Pending and error state remain
attached to the initiating Create User form or User row. The previous shared
link card below the Create User form is removed because it does not identify
which User a reset link belongs to.

### Administrator action confirmations

Five disruptive actions require focused confirmation: Disable User, Remove
Administrator role, issue a replacement Password Link, Clear Result Cache, and
mark a state-unknown Job failed.
Each dialog identifies its exact User or cache scope, explains the consequence,
and uses the destructive red confirmation style. A typed confirmation phrase is
not required.

The Admin Storage contract supplies the latest reclaimable cache entry count and
byte estimate for the Clear Result Cache dialog. Because a download lease may
begin while the dialog is open, the clear response reports the entries and
bytes actually removed and may be lower than the estimate. Saving or restoring
one setting, enabling a User, granting Administrator role, and copying a value
do not require confirmation.

Selecting Clear Result Cache first refreshes the reclaimable estimate and shows
a pending indicator before opening its confirmation. A successful cleanup then
refreshes the displayed metrics and reports the actual result.

### Bounded password work

Argon2 work uses one process-local bounded executor with two active operations
and at most eight queued operations. Login and Password Setup requests that
arrive after that bound return `503 authentication_busy` with `Retry-After`;
the frontend shows a generic temporary-busy message and permits a retry.

Password Setup checks the high-entropy Password Link digest before doing the
Argon2 work, then atomically rechecks and consumes the link when it commits the
new password and Session. Login retains a dummy Argon2 verification for unknown
Users. This small availability bound does not add account lockout, per-User
throttling, or a wider authentication-hardening project to the functional MVP.

### Bounded control-plane work

Successful authentication updates a Session's persisted idle-activity time at
most once every five minutes rather than writing SQLite on every authenticated
request. Expiry continues using the last persisted activity; the coalescing
window is intentionally negligible relative to the 30-day idle lifetime.

One GROMACS reconciliation pass runs at most four independent Jobs concurrently.
Within one Job it polls every active direct call and may submit the fixed
parallel branches described above. It creates only the fixed Job worker count,
isolates an unexpected per-Job error so other Jobs still progress, and performs
intermediate cleanup after the workers finish. Per-Job lifecycle locks remain
shared across HTTP cancellation and reconciliation while in use, then leave the
process registry automatically when no task retains them. Durable Job and
per-stage call state remain the restart-safe source of truth.

### OpenAPI contract discipline

The live FastAPI OpenAPI document is the executable backend-frontend API
contract. The glossary, ADRs, and specifications record the intended product
and architecture behavior; neither may silently disagree with OpenAPI. A
contract-affecting change is incomplete until its owning ADR or specification,
backend OpenAPI output and contract tests, and generated frontend TypeScript
agree.

OpenAPI must describe request and response bodies, Job states and conditional
fields, per-operation frontend-handled error codes, authentication requirements,
CSRF headers, relevant response headers, and binary and byte-range Result
downloads. This includes the `blocked` Job contract and the Admin Modal
preflight and Storage contracts, plus `state_unknown`, its safe timestamp and
reason fields, the Admin resolution operation, and the admin-only log-target and
plain-text stream operations. Protected operations declare the runtime
session-cookie security scheme. The Password Link's
`/set-password#token=...` SPA URL remains a tested cross-repository navigation
contract rather than an OpenAPI operation.

Any backend change that alters a public operation must check and, when needed,
update OpenAPI and its tests. Any frontend change that relies on API behavior
must check the live schema rather than hand-writing a wider local type. The
frontend's generated file is never edited by hand. A cross-repository change
cannot merge or release until `api:check` passes against the intended backend
revision.

The frontend continues to generate compile-time TypeScript schema and operation
types but keeps one small handwritten runtime client. Its centralized
`fetch`/`XMLHttpRequest` boundary owns same-origin cookies, CSRF, request-ID
extraction, coded errors, and upload progress. Functions use generated
operation types where practical; no second runtime-client generator is added
for the single-Tool MVP. This choice is reconsidered when additional Tools make
the wrapper materially larger.

### Browser integration release gate

The MVP requires a small Playwright suite that exercises the built frontend
against a real local FastAPI HTTP server and temporary SQLite database. The
server uses a deterministic fake at the existing GROMACS Modal-adapter seam;
the suite must never resolve deployed Functions, contact Modal, or incur
compute cost. It must not add a test-only route to the production API.

The browser suite covers Password Setup and login, a GROMACS Submission, Job
stage display and advancement, successful Result download, Cancellation to a
terminal state, and sign-out. It includes a regression assertion that one User
submission action creates exactly one Job and exactly one initial adapter call,
including the double-click/duplicate-event case. Separate Jobs may be used for
the success and Cancellation paths.

This focused suite is a required cross-repository CI and release check, not the
start of exhaustive visual or browser automation. Unit and API contract tests
continue to own permutations and edge cases.

### Automated merge gates

The backend merge gate installs the committed lockfile, runs the existing
`prek` checks, and runs the full pytest suite. The frontend merge gate installs
the committed lockfile, runs lint and unit tests, and produces the production
build. The manually dispatched cross-repository gate requires full
40-character candidate frontend and backend commit hashes, rejects mutable
branches, tags, and abbreviated hashes, and records the checked-out pair in its
workflow summary. It runs the live OpenAPI `api:check` and the focused
Playwright suite against that immutable pair.

The MVP does not add a version matrix, coverage threshold, or new mandatory
backend type checker. CI never contacts Modal. The Administrator-run real-Modal
smoke test remains a deployment check rather than a merge check.

### Frontend crash recovery

The router root owns one error boundary for unexpected render, lazy-route, and
chunk-loading failures. It replaces a blank application with a plain
`BioModals could not load this page` recovery screen offering Reload and Return
home actions. It never displays a stack trace, raw API response, provider
detail, or secret.

Expected authentication, validation, API, Job, and form failures remain with
their specific page-level handling and must not be collapsed into the root
screen. The boundary reports technical detail to the development console only;
the MVP adds neither an external frontend telemetry service nor an automatic
reload loop.

### My Jobs view state

My Jobs stores non-default column filters and sort order in `/jobs` search
parameters. Opening a Job and returning with browser Back, refreshing, or
bookmarking therefore preserves the same table view. Default values are
omitted, and unknown columns, choices, directions, or malformed dates are
ignored and normalized without failing the route.

The MVP does not duplicate this state in local or session storage. Search
parameters control the existing client-side filtering and sorting only; they do
not silently imply server-side query behavior.

The My Jobs and Admin Users endpoints use stable owner-scoped cursor pages with
a default of 50 and maximum of 100 records per response. Each response returns
its collection plus a nullable continuation cursor, and a cursor cannot cross
the authenticated owner boundary. The frontend follows those cursors to
assemble the complete MVP collection, preserving existing client-side sorting
and filtering without a hidden result cap while bounding each SQLite query and
HTTP response. Server-side filter/sort protocols and total counts remain
outside the MVP.

### Responsive layout boundary

Every core User and Administrator workflow remains functional at a viewport
width of 360 CSS pixels. Navigation and actions do not clip; authentication,
Password Setup, GROMACS Submission, Job detail, Cancellation, and Result
download remain usable; and Admin forms stack vertically.

Wide semantic tables retain every column and scroll horizontally rather than
hiding data or duplicating a card representation. File upload retains a
touch-friendly native Choose file path in addition to drag-and-drop. The MVP
does not create a separate mobile design or device-specific feature set.

### Approved frontend interaction corrections

The User avatar menu closes when focus or pointer interaction moves outside it,
when Escape is pressed, or when its trigger is used again. Focus returns to the
trigger after keyboard dismissal. Administrator Users see an Admin link in this
menu. The persistent Tools navigation link gains an icon consistent with My
Jobs.

The Catalog removes the redundant Tools heading: Tool count and cards follow
the search control directly. My Jobs uses generic cross-Tool copy: `New job`
links to the Tool Catalog and empty-state copy says `Start a new job`, never a
GROMACS-only simulation action.

The GROMACS overview says `Configure the simulation` and `Run in the
background`; the latter description contains a real My Jobs link. Job detail
labels the deployed call column `Running Function`. User-facing prose follows
sentence case; BioModals, GROMACS, Modal, PDB, PDBFixer, API, acronyms, and
explicit page names retain meaningful capitalization.

The PDB file selector gives its native Choose file control visible separation
through a light neutral rounded treatment. Drag entry changes the complete drop
zone to an explicit ready-to-drop state, drag leave restores it, and dropping a
valid file follows the same validation and selection path as native file
choice. The native path remains available for touch and keyboard Users.

The display-only Modal service-user token ID uses a visually muted read-only
field with an in-field Copy control and visible copied feedback. The Environment
form's one Save action sits on its own bottom row aligned to the right and saves
only changed fields. Every editable setting keeps its field-specific in-box
Restore control. Admin User-row actions share control height, spacing, and
typography; Disable and Remove admin retain destructive red styling, while
non-destructive actions use one consistent secondary treatment.

### Navigation scroll restoration

Ordinary link navigation starts at the top of the destination page. Browser
Back and Forward restore the prior vertical scroll position, so returning from
Job detail returns to the approximate My Jobs position while URL parameters
restore its exact filters and sort. The MVP does not separately persist a
table's horizontal scroll offset. Focused dialogs continue returning keyboard
focus to their initiating controls.

## Review outcome

The product owner accepted the constraints above and authorized implementation
without deployment. Concurrent Administrator conflict detection,
cross-deployment capacity coordination, and broader security work remain
deliberately deferred under their owning sections. Completion still requires
all automated gates, independent backend and frontend reviews, and an
Administrator-run human smoke test before any deployment.
