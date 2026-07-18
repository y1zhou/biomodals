# Biomodals API service architecture

Status: accepted for the department development service

Decision date: 2026-07-16

Scope: HTTP ingress and job orchestration for GROMACS, AlphaFold 3, and future
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
Function. The control plane mirrors the established `submit_gromacs_task`
local entrypoint by calling the existing preparation, trajectory-analysis, and
production Functions sequentially. After the last Function succeeds, the
control plane packages the established Volume outputs itself. The GROMACS App
and its command-line behavior remain unchanged. This keeps the HTTP contract
and account data local while preserving independent compute images, scaling,
and deployment lifecycles. [Modal: invoking deployed Functions](https://modal.com/docs/guide/trigger-deployed-functions)

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
  |                            |-> collect_traj_stats(nvt_)
  |                            |-> collect_traj_stats(npt_)
  |                            |-> production_run_{cpu,gpu}
  |                            |-> collect_traj_stats(production_)
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
- idempotent submissions and per-workload active-job limits;
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
  artifacts.py       verified final-ZIP LRU cache
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

GROMACS submission is `POST /api/v1/gromacs/jobs`; AlphaFold 3 will use its own
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
The initial configurable limits are two active GROMACS jobs and one future
AlphaFold 3 job per user.

Every list, inspect, cancel, and download lookup is constrained by both
`owner_user_id` and `job_id`. Looking up another user's job returns the same
`404` as a missing job, before the server resolves any Modal identifier.
Account administration does not grant access to employee jobs.

The submit route persists the job, spawns the first Modal stage, stores its
Function name and call identifier internally, and returns `202`. When that call
completes, the reconciler resolves the next deployed Function by name and
atomically replaces the stored active operation and call identifier. Exactly
one direct stage is the durable active operation at a time. `JobView` exposes a
sanitized stage code and the associated deployed Function name so the Job
detail page can show the current sequential step. Modal call IDs, App and
Environment names, Volume names and paths, dashboard links, tracebacks, and
internal filesystem paths remain private.

The GROMACS API sequence is:

```text
prepare_tpr_cpu|gpu
  -> collect_traj_stats(traj_prefix="nvt_")
  -> collect_traj_stats(traj_prefix="npt_")
  -> production_run_cpu|gpu
  -> collect_traj_stats(
       traj_prefix="production_",
       save_processed_traj=true
     )
  -> service builds and publishes result.zip
```

This is the same sequence and argument shape as the GROMACS App's established
`submit_gromacs_task` local entrypoint, except that the API deliberately waits
for each call before starting the next one. `collect_traj_stats` remains free to
use its own established implementation details, including its internal call to
`postprocess_traj`; the API does not duplicate or alter those details.

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
```

Cancellation is idempotent while it is pending. The adapter asks Modal to
cancel the currently recorded direct call and any visible active descendants
with `terminate_containers=False`; the job becomes `cancelled` only after the
call graph is inactive. If a verified result archive wins a cancellation race,
the completed result wins. Terminal jobs are preserved and there is no
job-delete endpoint in v1. [Modal: `FunctionCall`](https://modal.com/docs/sdk/py/latest/modal.FunctionCall)

If a submission may have reached Modal but no call ID was durably recorded,
the control plane cannot prove inactivity or issue a targeted cancellation. A
Cancellation request therefore remains pending through the submission lease
and becomes `failed`, not falsely `cancelled`, when that lease expires.

Initial submission uses a short SQLite lease and the stable run name
`api-<job UUID>`. An idempotent replay cannot create a second call while that
lease is active. If the process dies after claiming the Job but before storing
a Modal call ID, reconciliation leaves the Job queued until the lease expires
and then fails it with `compute_failed`. It does not automatically resubmit an
operation whose provider outcome cannot be proven, because doing so could
duplicate paid compute. The User must start a new Submission explicitly.

Every later stage transition takes the same kind of durable lease before
calling `.spawn()`. A returned call ID atomically replaces the prior completed
call and clears the lease. If the process dies or Modal's response is ambiguous
before that replacement, reconciliation waits for expiry and fails the Job; it
does not spawn the stage again. Each established stage writes resume-aware
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

Packaging can be repeated safely for the same completed run because the member
order, metadata, and contents are deterministic for the recorded job. The
single-worker control plane remains responsible for preventing ordinary
concurrent publication. No API-specific registry or Function is added to the
GROMACS App.

The ZIP contains an explicit allowlist of final outputs, including:

- the exact submitted PDB and normalized parameters;
- the processed no-PBC production trajectory, centered structure, production
  topology and parameters;
- RMSD, radius-of-gyration and RMSF data as CSV files and plots as PNG files;
- provenance such as the Modal App name and timestamps;
- a manifest and checksums; and
- a service-generated `run.log`.

It excludes the larger raw production trajectory and does not recursively
package equilibration outputs, the working directory, credentials, internal
storage paths, databases, or large shared caches.

Modal Volume storage is authoritative. Scientific cache files, databases, and
unselected intermediates remain on Modal. The selected final files stream
through a spooled temporary file on the Linux host only while the service builds
the ZIP; they are not added to its durable job database. The configurable local
directory caches final ZIPs only. It is a size-bounded LRU cache with atomic
publication and size/SHA-256 verification:

1. a cache hit is served as a local file;
2. a miss streams the ZIP from the recorded Modal Volume;
3. an archive larger than the cache target is downloaded to an unlinked
   temporary file, verified, streamed through that held descriptor, and not
   retained; and
4. eviction never deletes the Modal source.

The cache can be deleted or rebuilt without losing a job. Intermediate cleanup
is disabled when `BIOMODALS_INTERMEDIATE_RETENTION_DAYS` is unset or blank. A
positive number enables deletion of only the workload's `<run_name>/`
intermediate directory after the terminal ZIP has remained published for that
many days. Final archives and shared scientific caches are outside that cleanup
policy. [Modal: Volumes](https://modal.com/docs/guide/volumes)

## Local persistence and operations

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
| `BIOMODALS_CACHE_MAX_BYTES` | `10737418240` | Local cache target (10 GiB) |
| `BIOMODALS_PUBLIC_URL` | `http://localhost:5173` | One public origin for links, exact-Origin checks, and same-origin browser access |
| `BIOMODALS_SECURE_COOKIES` | `false` | Use secure `__Host-` session cookies behind HTTPS |
| `BIOMODALS_MODAL_ENVIRONMENT` | `production` | Modal Environment default; configurable in Admin unless a process override is set |
| `BIOMODALS_GROMACS_APP` | `Gromacs` | Deployed GROMACS Modal App name |
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

Start the development server with:

```bash
uv run biomodals api serve
```

The command defaults to `127.0.0.1:8000`, accepts `--host` and `--port`, and
keeps the required worker count at one.

Production uses the same factory and worker count behind the internal HTTPS
reverse proxy. See the root README for the complete local setup and account
provisioning example.

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
