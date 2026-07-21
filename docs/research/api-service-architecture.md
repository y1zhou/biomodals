# Biomodals API service architecture

Status: accepted for the department development service

Decision date: 2026-07-16

Scope: HTTP ingress and job orchestration for GROMACS, AlphaFold3, and future
Biomodals apps and workflows

## Decision

Run one unified FastAPI control plane on the department's internal Linux host.
Run it as one Uvicorn worker supervised by systemd, with a local SQLite database
in WAL mode. The frontend and API share one browser origin: the production
reverse proxy serves the frontend at `https://biomodals.internal/` and proxies
`/api/*` to FastAPI. A frontend development server uses the same `/api` proxy
pattern.

Scientific compute remains in separately deployed Modal Apps. Each workload
module contributes an `APIRouter` and a narrow Modal adapter to the control
plane. The adapter resolves each deployed stage by its Function name with
`modal.Function.from_name()`, submits it with `.spawn()`, and reconciles the
detached `FunctionCall`. GROMACS has no API-only coordinator or packaging
Function. The control plane calls the existing preparation,
trajectory-analysis, and production Functions through a fixed durable
dependency graph. After preparation, NVT analysis, NPT analysis, and production
run concurrently; production analysis starts when production finishes; Result
preparation waits for all three analyses. The control plane then packages the
established Volume outputs itself. The GROMACS App and its command-line behavior
remain unchanged. This keeps the HTTP contract and account data local while
preserving independent compute images, scaling, and deployment lifecycles.
[Modal: invoking deployed Functions](https://modal.com/docs/guide/trigger-deployed-functions)

No Modal Web Function, webhook, `@modal.asgi_app`, or
`@modal.fastapi_endpoint` is needed for this deployment. Those components are
HTTP ingress options, not prerequisites for calling Modal Functions through the
SDK. They remain valid future hosting choices if the control plane itself ever
moves to Modal. [Modal: Web Functions](https://modal.com/docs/guide/webhooks)

```text
employees' browsers
        |
        v
frontend + /api reverse proxy (one internal origin)
        |
        v
one FastAPI process
  |-- shared auth, jobs, downloads, SQLite, reconciler
  |-- /api/v1/gromacs/*   -> GROMACS adapter
  |                            |-> prepare_tpr_{cpu,gpu}
  |                            |     |-> collect_traj_stats(nvt_)
  |                            |     |-> collect_traj_stats(npt_)
  |                            |     `-> production_run_{cpu,gpu}
  |                            |             `-> collect_traj_stats(production_)
  |                            `-> service-built result.zip
  `-- /api/v1/alphafold3/* -> future AF3 adapter -> deployed AF3 Modal App
                                      |
                                      `-> Modal Volumes (authoritative data)
```

This is one API surface, not one giant Modal App. A compute App can be deployed
without rebuilding the control plane, and adding an API router does not merge
scientific container images.

## Why a local FastAPI server is sufficient

The users are a few dozen employees on the company network. The department
already has a continuously available Linux machine and does not need public
ingress, scale-to-zero, or Internet-facing autoscaling for the control plane.
A normal ASGI server therefore supplies everything required for ingress. Modal
continues to autoscale the expensive CPU and GPU Functions independently.

Compared with a Modal-hosted ASGI App, this choice provides:

- a stable internal origin with ordinary browser cookies;
- local control of identity and private job history;
- no Modal-specific HTTP headers or URLs in the frontend contract; and
- less ingress vendor lock-in.

The tradeoff is operating one small service: TLS/reverse proxying, systemd,
backups, disk monitoring, and the dedicated service credentials are local
responsibilities. This is acceptable for the stated scope. It would need to be
revisited before adding multiple hosts or API workers.

## One server, explicit workload modules

Use a single versioned API because users need one login and one "My Jobs" view
across workloads. Shared policy belongs in the control plane:

- cookie authentication, CSRF and exact-Origin validation;
- immutable public job IDs and submitter ownership;
- idempotent submissions and User, Tool, and Global active-job limits;
- common lifecycle, cancellation, error, and download behavior; and
- one durable audit/provenance record.

Workload-specific validation stays in explicit submodules. FastAPI's
`APIRouter` is designed for this application shape. [FastAPI: bigger applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/)

The current layout is:

```text
src/biomodals/service/
  api.py             unified app factory, auth and common routes
  config.py          host configuration
  auth.py            manual accounts and opaque browser sessions
  store.py           SQLite users, sessions and jobs
  jobs.py            common job view and workload registration
  artifacts.py       verified final-ZIP staging and cache
  gromacs/
    router.py        GROMACS request schema and submission route
    modal.py         Modal invocation, polling, cancellation and artifacts
```

New workloads export a router plus lifecycle hooks through an explicit
`WorkloadRegistration`; there is no import-time plugin discovery. App-specific
submission endpoints are clearer than a lowest-common-denominator
`POST /jobs?kind=...`, because a PDB upload, AlphaFold input, and workflow DAG
have different schemas.

The common routes are:

- `GET /api/v1/jobs`
- `GET /api/v1/jobs/{job_id}`
- `POST /api/v1/jobs/{job_id}/cancel`
- `GET /api/v1/jobs/{job_id}/download`

GROMACS submission is `POST /api/v1/gromacs/jobs`; AlphaFold3 will use its own
prefix.

Split a workload into a separate API server only when it develops a genuinely
different trust boundary, owner, region, release policy, availability target,
or ingress dependency. Different Modal images or GPU types are not sufficient
reasons: compute Functions already have independent containers.

## Private asynchronous jobs

The browser supplies a UUID `Idempotency-Key` when submitting a job. The key is
scoped to `(owner_user_id, workload)`. Repeating the same key and payload
returns the existing job; reusing it with different inputs returns `409`.
Admission and active-job limits are checked in one SQLite write transaction.
That transaction also rechecks that the User is enabled before returning an
existing same-payload admission, so a request racing with Disable cannot claim
an unsubmitted Job's provider lease.
Limits are non-negative integers and zero intentionally blocks new Submissions
within that scope. Lowering a User, Tool, or Global limit below its current
count never cancels admitted work; new Submissions are rejected until the
applicable count falls below the configured limit.

Counts are scoped to this service's SQLite database. Global therefore means all
Users and Tools admitted by one BioModals deployment, not a combined beta,
production, or Modal-account total. Separate deployments that target the same
Modal App do not coordinate admission. Pre-release examples set User, Tool,
and Global defaults to one; provider-level limits or shared coordination remain
outside this architecture.

Every list, inspect, cancel, and download lookup is constrained by both
`owner_user_id` and `job_id`. Looking up another user's job returns the same
`404` as a missing job, before the server resolves any Modal identifier.
Account administration does not grant access to employee jobs.

The submit route persists the Job, spawns preparation, stores its Function name
and call identifier internally, and returns `202`. SQLite keeps one durable
provider-call row per directly submitted operation. When calls complete, the
reconciler evaluates the fixed GROMACS dependencies and attaches every newly
ready call. Several direct stages may therefore be active at once. Per-stage
submission leases retain the existing restart and ambiguous-submission boundary
without introducing a remote coordinator.

`JobView.active_stages` exposes every active sanitized stage code and deployed
Function name. The singular `stage` remains as a compatibility summary of the
most recently started active stage, or the relevant terminal stage. SQLite also
retains an ordered `stage_history` timing record: a stage starts when its direct
call is durably attached and ends when the reconciler records its observed
terminal outcome. Parallel entries may have overlapping timestamps and may
finish in a different order from their table rows. Each entry has `started_at`,
nullable `ended_at`, and a nullable outcome of `completed`, `failed`, or
`cancelled`; active, state-unknown, and blocked work has no end or outcome.
Result packaging spans `finalizing` through archive publication. Modal call IDs,
App and Environment names, Volume names and paths, dashboard links, tracebacks,
and internal filesystem paths remain private.

The stable public GROMACS stage mapping is:

| Stage code | Display label | Associated Function |
| --- | --- | --- |
| `prepare_simulation` | Prepare simulation | `prepare_tpr_cpu` or `prepare_tpr_gpu` |
| `analyze_nvt` | Analyze NVT | `collect_traj_stats` |
| `analyze_npt` | Analyze NPT | `collect_traj_stats` |
| `run_production` | Run production | `production_run_cpu` or `production_run_gpu` |
| `analyze_production` | Analyze production | `collect_traj_stats` |
| `prepare_result` | Prepare result | none; local service work |

The preparation Function internally performs preparation, minimization, NVT,
and NPT execution. Those internal phases are not independent API-orchestrated
calls and therefore are not public stages. The three trajectory-analysis calls
share a Function name but remain distinct stages because their prefixes and
positions differ. Nested App implementation calls such as `postprocess_traj`
also remain outside the public timeline.

The GROMACS API dependency graph is:

```text
prepare_tpr_cpu|gpu
  |-> collect_traj_stats(traj_prefix="nvt_") -------------------|
  |-> collect_traj_stats(traj_prefix="npt_") -------------------|
  `-> production_run_cpu|gpu                                     |
        `-> collect_traj_stats(                                  |
              traj_prefix="production_",                         |
              save_processed_traj=true                           |
            ) ---------------------------------------------------|
                                                                  `-> service builds
                                                                      result.zip
```

The API calls the same established Functions with the same scientific
arguments as `submit_gromacs_task`, but overlaps production with NVT/NPT
analysis. The CLI keeps its existing, slightly more conservative ordering.
Preparation commits the shared inputs before fan-out, each branch writes
distinct prefixed files on the Modal Volume v2, and Result preparation is the
join barrier. `collect_traj_stats` remains free to use its established
implementation details, including its internal call to `postprocess_traj`; the
API does not duplicate or alter those details.

A definite branch failure prevents further dependent submissions and requests
cancellation of every still-running sibling. The Job remains active until those
calls are confirmed inactive, then becomes `failed`; this avoids hiding paid
remote work behind an early terminal state. If a sibling's status expires while
the service is trying to stop it, the Job becomes `state_unknown` and continues
to consume admission capacity until an Administrator resolves it.

There is deliberately no deployed `run_gromacs_job` Function wrapping these
calls. Such a wrapper adds a second Modal call layer, obscures which resource
stage is active, and makes the local durable Job record cease to be the
orchestration authority. Likewise, archive construction is control-plane work,
not a new Function added to the scientific App.

One in-process reconciler polls active Modal calls approximately every ten
seconds. Public states are intentionally coarse:

```text
queued -> running -> finalizing -> succeeded
                            |---> partial (downloadable, with warnings)
                            `---> failed
queued/running -> cancel_requested -> cancelled
queued/running/cancel_requested -> state_unknown -> failed (Admin resolution)
```

Cancellation is idempotent while it is pending. The adapter asks Modal to
cancel every recorded active direct call and any visible active descendants
with `terminate_containers=False`; the Job becomes `cancelled` only after all
call graphs are inactive. This cancels inputs without forcibly terminating
workers that may contain unrelated inputs. `cancel_requested_at` is persisted
before the provider request, transient failures are retried, and restart
resumes reconciliation. No successor stage may be spawned after the durable
request. Calls that complete first are recorded as complete while their active
siblings are cancelled; if a verified Result archive was already published,
the completed Result wins. Terminal jobs are preserved and there is no
job-delete endpoint in v1. [Modal: `FunctionCall`](https://modal.com/docs/sdk/py/latest/modal.FunctionCall)

The service does not infer successful Cancellation from a timeout. A pending
Cancellation continues consuming Active Job Limits while Modal still exposes a
reconcilable call. After 15 minutes the owner-facing view uses the persisted
timestamp to warn that Cancellation is taking longer than expected while
reconciliation continues. If any call status expires before cancellation can
be confirmed and no verified final Result can be recovered, the Job moves to
`state_unknown`, not falsely to `cancelled`.

`state_unknown` is the durable safety state for remote execution that may still
exist but can no longer be tracked automatically. It also applies when a
`.spawn()` request may have reached Modal but no call ID was durably recorded.
An explicit ambiguous SDK outcome enters it immediately; a process interruption
enters it after the short submission lease expires. It consumes User, Tool, and
Global Active Job Limits, is excluded from reconciliation, and is returned by
idempotent replay without submitting again.

The owner sees “Status unknown,” a generic explanation, and
`state_unknown_at`, but no provider detail. The Admin Modal page lists the safe
Job ID, workload, display name, run name, fixed reason, and timestamp needed for
manual Modal review. After checking Modal and stopping remote work there when
necessary, an Administrator may use the destructive `Mark failed` action. That
action records a safe `compute_failed` terminal failure and releases admission
capacity; it does not itself contact or cancel Modal. No automatic or owner
transition leaves `state_unknown`.

Initial submission uses a short SQLite lease and a stable run name made from a
sanitized display-name slug plus the full Job UUID, for example
`kinase-trial-<job UUID without hyphens>`. The deployed GROMACS App uses that
single value for both its Volume directory and scientific filenames, so the
UUID suffix prevents repeated display names from silently reusing another
Job's checkpoints. The API does not change the established App interface.
Legacy `api-<job UUID>` names remain valid for stored Result recovery.

An idempotent replay cannot create a second call while the lease is active. If
the process dies after claiming the Job but before storing a Modal call ID,
reconciliation leaves the Job queued until the lease expires and then moves it
to `state_unknown`. It does not automatically resubmit an operation whose
provider outcome cannot be proven, because doing so could duplicate paid
compute. An Administrator must review the remote state before marking it failed.

Every later stage submission takes the same kind of operation-scoped durable
lease before calling `.spawn()`. A returned call ID atomically attaches to that
operation and clears its lease without replacing sibling calls. If the process
dies or Modal's response is ambiguous before that attachment, the Job enters
`state_unknown` immediately for an
explicit ambiguous response or at lease expiry after a process interruption;
it does not spawn the stage again. Each established stage writes resume-aware
output under the stable run name, but resume behavior is not treated as a
provider idempotency guarantee. The required single API worker means routine
reconciliation passes cannot race. Eliminating this untracked-call limitation
requires provider-side idempotent submission or an external durable worker
before enabling multiple API processes.

The supported `FunctionCall` API does not expose a backend log stream. Live
stdout streaming is therefore deferred. A small service-generated `run.log`
records the completed job identity and status instead of integrating Modal's
CLI log command into the service.

## Artifact and storage contract

Each successfully completed job produces exactly one immutable,
browser-friendly ZIP. A workload may also use the generic `partial` terminal
state for a downloadable result with warnings; GROMACS v1 itself emits
`succeeded`. After the final established App call returns, the control plane
streams an explicit allowlist of files from `Gromacs-outputs` into a
deterministic local ZIP, validates it, then uploads `result.zip` followed by a
small completion marker under `api-results/<run_name>/`. The marker records the
request identity, byte size, and SHA-256 digest. The archive and marker are the
durable success boundary; the control plane does not mark a job complete until
the ZIP contract validates.

When the final deployed Function completes, the control plane atomically stores
`finalization_started_at` as it enters `finalizing`. Archive provenance uses
that persisted value rather than the time of an individual packaging attempt.
Packaging can therefore be repeated safely for the same completed run because
the member order, ZIP metadata, timestamps, and contents are deterministic for
the recorded Job. Unchanged Volume outputs reproduce the same archive bytes,
size, and SHA-256 digest. The single-worker control plane remains responsible
for preventing ordinary concurrent publication. No API-specific registry or
Function is added to the GROMACS App.

The ZIP has exactly three top-level entries or namespaces:

- `input.pdb` is the exact submitted structure;
- `outputs/` contains the processed no-PBC production trajectory, centered
  structure, production topology and parameters, plus RMSD,
  radius-of-gyration, and RMSF CSV/PNG pairs; and
- `metadata/` contains the normalized parameters, safe provenance,
  service-generated run log, manifest, checksums, and any other explicitly
  allowed debugging document that is not useful at the top level.

The `metadata/` allowlist is fixed rather than pattern-recursive:

- `parameters.json` stores normalized Submission parameters;
- `provenance.json` stores safe Job, Tool, deployed App, software-version, and
  persisted timestamp facts;
- `stages.json` stores public stage codes, deployed Function names,
  started/ended timestamps, and terminal outcomes;
- `run.log` is the service-generated lifecycle summary;
- `manifest.json` and `checksums.sha256` verify the archive; and
- `gromacs/` contains the existing minimization, NVT, NPT, and production text
  logs plus the small minimization/equilibration `.mdp` files.

The exact Input, required end-User outputs, every CSV/PNG analysis pair, and
service-generated metadata are mandatory. Missing any of them at initial
publication fails the Job with `result_invalid`. The allowlisted GROMACS logs
and `.mdp` files are optional diagnostics: package them when present and list
their exact membership in the manifest, but do not fail an otherwise valid
Result when they are absent.

No metadata document remains loose at the archive root. The allowlist excludes
the larger raw production trajectory and does not recursively package
equilibration outputs, the working directory, credentials, internal storage
paths, databases, raw provider exceptions, or large shared caches. It also
excludes equilibration trajectories, `.trr`, `.edr`, `.cpt`, and intermediate
`.tpr` and structure files; deeper diagnosis uses the authoritative Modal
Volume.

Once published, archive membership is immutable. Reconstruction may neither
add an optional diagnostic that was absent nor drop one that was present,
because either would violate the recorded size and SHA-256 identity.

Modal Volume storage is authoritative. Scientific cache files, databases, and
unselected intermediates remain on Modal. The selected final files stream
through staging below the configured local cache directory while the service
builds the ZIP. A successfully published staged ZIP becomes the local cache
entry instead of being downloaded back from Modal.

The local Result Cache has atomic publication and size/SHA-256 verification but
no automatic size-driven eviction:

1. a cache hit is served as a local file;
2. a miss restores or reconstructs the ZIP from the recorded Modal Volume;
3. every finalized Job records Result byte size and current local-cache
   presence;
4. active staging files and leased downloads are protected from cleanup; and
5. clearing local cache data never deletes the Modal source or reruns
   scientific compute.

A cache miss first restores the published Modal ZIP and completion marker. If
either is missing or corrupt, deterministic reconstruction reads the explicit
raw-output allowlist and uses the persisted provenance timestamp. The backend
republishes only if size and SHA-256 match the Job's recorded Result. A mismatch
never replaces the Result or replays scientific compute: the previously
completed Job becomes blocked under safe category `result_integrity` until the
exact archive is restored, then returns to its prior completed state.
Local staging and cache filesystem errors remain recoverable and eventually
block under `local_storage`; they are not classified as invalid scientific
Results.

Browser download uses a two-step contract. Idempotent
`POST /api/v1/jobs/{job_id}/prepare-download` fills and verifies the local cache
through the per-Job coordination lock and returns `204`; the following GET
streams the prepared file with normal browser download handling. Cache-fill
work is shared so cancellation of one HTTP waiter does not cancel or corrupt
the underlying fill. This adds neither scientific work nor an external task
queue.

The GET supplies a sanitized `Content-Disposition` filename of
`<display-name>-results.zip`, falling back to `gromacs-results.zip`. It never
exposes a Job UUID, Modal run name, or storage path. The filename is presentation
only; the immutable manifest, byte size, and SHA-256 remain authoritative.

The Admin Storage page shows published Result bytes, completed local cache
bytes, active staging bytes, filesystem free space, and a process/file
controlled soft-warning threshold. The threshold defaults to 1 TiB and warns
on actual local usage without rejecting or deleting Results. Startup reconciles
database cache-presence markers with actual files and cleans abandoned staging
files.

The frontend loads this snapshot on Storage-page entry, refetches on focus,
offers manual Refresh with a last-updated time, and invalidates it after cache
cleanup or a Result cache fill completed in the same browser session. It does
not poll storage periodically or introduce push updates. Opening Clear Result
Cache first obtains a fresh reclaimable count and byte estimate; the mutation
returns actual entries and bytes removed and triggers another snapshot load.

Whole-file ZIP validation, SHA-256 verification, cache reconciliation, and
large directory scans run through one bounded artifact worker thread rather
than blocking the single FastAPI event loop. Modal byte streams remain
asynchronous. Cache fills use per-job coordination so unrelated downloads do
not wait behind one large restore. This preserves the single-process
architecture without introducing another service or task queue.

The cache can be deleted or rebuilt without losing a job. Intermediate cleanup
is disabled when `BIOMODALS_INTERMEDIATE_RETENTION_DAYS` is unset or blank. A
positive number enables deletion of only the workload's `<run_name>/`
intermediate directory after the terminal ZIP has remained published for that
many days. Final archives and shared scientific caches are outside that cleanup
policy. [Modal: Volumes](https://modal.com/docs/guide/volumes)

## Local persistence and operations

The service exposes separate unauthenticated operational probes. `/api/v1/health`
is event-loop liveness only. `/api/v1/ready` reports success only after startup
and while SQLite, configured cache storage, the bounded artifact worker, and the
reconciler remain usable. A failed readiness response contains no path or
configuration detail. Neither probe contacts Modal; startup and Admin setting
preflight own deployed-resource validation. Both stable operations appear in
OpenAPI, and deployment waits for readiness before admitting Users.

Each HTTP request receives a server-generated request identifier returned in
the OpenAPI-documented `X-Request-ID` response header and attached to related
journald entries. Lifecycle records use consistent searchable fields such as
`event`, `job_id`, `workload`, public stage code, and safe Blocking Category.
They cover admission/idempotent replay, stage changes, Cancellation,
finalization retry/block/recovery, Runtime Setting mutations, Result Cache
cleanup, and readiness transitions. Password Links, session/CSRF values,
request bodies, PDB content, Modal credentials, and secrets are never logged.
No external metrics, tracing, error-reporting, or audit service is required for
the MVP.

SQLite is intentionally restricted to one FastAPI process on local disk. WAL,
foreign keys, transactional admission, and a busy timeout provide the required
durability and concurrency for this scale. Do not place the database on a
Modal Volume or network filesystem: Modal Volumes use filesystem consistency
semantics and are not a multi-writer relational database.
[Modal: Volume consistency](https://modal.com/docs/guide/volumes#filesystem-consistency)

The state directory and cache directory are separate. Company backups must
cover the state directory and use a SQLite-aware backup/snapshot procedure;
copying only the main database file while its WAL is active is insufficient.
The cache does not need backup.

Run Uvicorn with `--workers 1`. Multiple workers would each start a reconciler
and would invalidate the current single-process SQLite assumptions. If the
department later needs multiple API processes or hosts, first move job and
identity state to a service database and add leader election or an external
worker for reconciliation.

The Linux process runs as a dedicated service account. Its Modal profile uses
a dedicated Modal service user and an explicit Environment. Keep the Modal
token only on the backend; browser clients never receive it. More restrictive
Modal RBAC can be added later without changing the API contract.
[Modal: service users](https://modal.com/docs/guide/service-users)
[Modal: Environments](https://modal.com/docs/guide/environments)

## Configuration and start command

The application factory reads these settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `BIOMODALS_API_CONF_ENV` | unset | Optional path to a dotenv file; explicit process variables override file values |
| `MODAL_TOKEN_ID` | required | Dedicated Modal service-user token identifier |
| `MODAL_TOKEN_SECRET` | required | Dedicated Modal service-user token secret; never returned by the API or stored in SQLite |
| `BIOMODALS_STATE_DIR` | `.biomodals/state` | Durable SQLite directory |
| `BIOMODALS_CACHE_DIR` | `.biomodals/cache` | Rebuildable final-ZIP cache directory |
| `BIOMODALS_CACHE_WARNING_BYTES` | `1099511627776` | Soft warning threshold for local Result staging and cache usage (1 TiB) |
| `BIOMODALS_PUBLIC_URL` | `http://localhost:5173` | One public origin for links, exact-Origin checks, and same-origin browser access |
| `BIOMODALS_SECURE_COOKIES` | `false` | Use secure `__Host-` session cookies behind HTTPS |
| `BIOMODALS_MODAL_ENVIRONMENT` | `production` | Modal Environment default; configurable in Admin unless a process override is set |
| `BIOMODALS_GROMACS_APP` | `Gromacs` | Deployed GROMACS Modal App name |
| `BIOMODALS_GROMACS_APP_VERSION` | `1` | Exact GROMACS Modal deployment version used by new Jobs |
| `BIOMODALS_GROMACS_ACTIVE_LIMIT` | `2` | Workload-wide active Job limit default |
| `BIOMODALS_GLOBAL_ACTIVE_JOB_LIMIT` | `10` | Global active Job limit default |
| `BIOMODALS_DEFAULT_USER_ACTIVE_JOB_LIMIT` | `2` | Active Job limit assigned to new Users by default |
| `BIOMODALS_RECONCILE_SECONDS` | `10` | Modal reconciliation interval |
| `BIOMODALS_INTERMEDIATE_RETENTION_DAYS` | unset | Positive retention enables cleanup of published runs' intermediates |

The Admin API stores editable runtime overrides for the Modal Environment,
GROMACS App name, and Tool and Global limits in SQLite. Dotenv values are their
host defaults. An explicit process variable has highest precedence and makes
the corresponding Admin field read-only. Modal credentials remain process/file
configuration only, and the API refuses to start unless both are present.

That full validation belongs specifically to `biomodals api serve`. Offline
`biomodals api admin` account commands resolve the same configuration file,
state path, process overrides, and public URL, then validate only their own
dependencies. They neither require Modal credentials nor initialize the Modal
client, reconciler, or deployed-resource preflight. Password Link creation
still requires a valid public URL so its generated SPA route matches the
deployed frontend.

The configuration file and process environment are read once during startup;
there is no file watcher or hot reload. Changing a file-controlled value or
Modal credential requires restart, and the new process repeats configuration
validation and Modal preflight before becoming ready. SQLite-backed Admin
Runtime Settings retain their immediate behavior. Admin reads reflect the
running process's loaded values rather than independently reparsing the file.

Runtime-setting PATCH requests are field-specific. An omitted field remains
unchanged; an explicit JSON `null` removes only that field's SQLite override,
revealing its dotenv value or built-in default. Resetting one Tool setting
therefore cannot change the provenance or effective value of another setting.

Start the development server with:

```bash
uv run biomodals api serve
```

The command defaults to the development address `127.0.0.1:4144`, accepts
`--host` and `--port`, and keeps the required worker count at one. Production
service definitions explicitly select port `4100`.

Production uses the same factory and worker count behind the internal HTTPS
reverse proxy. See the root README for the complete local setup and account
provisioning example.

The SQLite schema has one current pre-release version. Schema 10 has one narrow
automatic migration to schema 11: it adds per-stage provider-call records and
preserves all Users, Sessions, settings, Jobs, and stage history. Encountering
any other version is a startup error: the service reports the configured
database location and never truncates or deletes it automatically. During
active development an Administrator may explicitly remove or relocate an
unsupported database while the service is stopped, then restart to initialize
a fresh schema. This reset policy ends at the first release.

Pre-release and production service definitions select distinct
`BIOMODALS_STATE_DIR` and `BIOMODALS_CACHE_DIR` values. Pre-release sets
`BIOMODALS_PUBLIC_URL=https://beta.biomodals.example.com`; production uses
`https://biomodals.example.com`. A pre-release reset or cache clear therefore targets
only isolated host-local data and never falls back to production configuration,
state, or cache. Modal Environment selection remains explicit and independent;
an intentionally authorized pre-release service may still target the Modal
`production` Environment.

## Alternatives considered

| Option | Decision |
| --- | --- |
| Modal webhook/`fastapi_endpoint` per operation | Rejected. It fragments one job lifecycle across URLs and adds Modal-specific ingress without adding compute capability. |
| One Modal `asgi_app` per scientific app | Rejected for this service. It duplicates login, policy, metadata, and frontend integration. |
| One Modal `asgi_app` including all compute Apps | Rejected. It creates broad atomic deployment coupling without sharing compute pools. |
| One Modal `asgi_app` with deployed-Function adapters | Viable future host, but unnecessary while the internal Linux host is the accepted ingress. |
| One external FastAPI server with explicit workload modules | **Selected.** It matches the network, scale, UX, and portability requirements. |
| Modal Dict as the job database | Rejected. Job history and ownership must outlive short Modal call/Dict retention. |
| Modal Dict as an atomic final-archive publication registry | Rejected. The single control-plane reconciler publishes deterministic archives directly, so another registry adds no useful authority. |
| PostgreSQL and multiple API workers | Deferred. It adds operations without benefit at the current scale; it becomes necessary before horizontal API scaling. |
| SSE/WebSockets or Modal log streaming | Deferred. Polling coarse SQLite state is enough for v1, and the public Modal call API has no supported log stream. |
