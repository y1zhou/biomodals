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
plane. The adapter resolves a deployed Function with
`modal.Function.from_name()`, submits it with `.spawn()`, and reconciles the
detached `FunctionCall`. This keeps the HTTP contract and account data local
while preserving independent compute images, scaling, and deployment
lifecycles. [Modal: invoking deployed Functions](https://modal.com/docs/guide/trigger-deployed-functions)

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
  |-- /api/v1/gromacs/*   -> GROMACS adapter   -> deployed Gromacs Modal App
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

The submit route persists the job, spawns the Modal Function, stores its call
identifier internally, and returns `202`. Modal call IDs, Volume names and
paths, dashboard links, tracebacks, and internal filesystem paths are never
part of the public response.

One in-process reconciler polls active Modal calls approximately every ten
seconds. Public states are intentionally coarse:

```text
queued -> running -> finalizing -> succeeded
                            |---> partial (downloadable, with warnings)
                            `---> failed
queued/running/finalizing -> cancel_requested -> cancelled
```

Cancellation is idempotent while it is pending. The adapter asks Modal to
cancel the root and currently visible active descendants with
`terminate_containers=False`; the job becomes `cancelled` only after the call
graph is inactive. If a verified result archive wins a cancellation race, the
completed result wins. Terminal jobs are preserved and there is no job-delete
endpoint in v1. [Modal: `FunctionCall`](https://modal.com/docs/sdk/py/latest/modal.FunctionCall)

Submission uses a short SQLite lease and the stable run name
`api-<job UUID>`. An idempotent replay cannot create a second call while that
lease is active, and a replay after a process failure can safely retry the same
run. If the call was accepted before its ID reached SQLite, reconciliation can
recover the verified archive by stable run name.

The supported `FunctionCall` API does not expose a backend log stream. Live
stdout streaming is therefore deferred. A sanitized `run.log` is included in
completed or partial result archives instead of integrating Modal's CLI log
command into the service.

## Artifact and storage contract

Each successfully completed job produces exactly one immutable,
browser-friendly ZIP inside the Modal compute job. A workload may also use the
generic `partial` terminal state for a downloadable result with warnings;
GROMACS v1 itself emits `succeeded`. The GROMACS entrypoint writes its ZIP under
an opaque run directory on the `Gromacs-outputs` Volume and returns one archive
artifact with its filename, byte size, and SHA-256 digest. The archive is the
durable success boundary; the control plane does not mark a job complete until
that contract validates.

Retries can overlap only in the narrow failure window between Modal accepting a
call and SQLite recording its call ID. GROMACS therefore writes a unique
candidate, then uses atomic `Modal Dict.put(..., skip_if_exists=True)` to elect
one durable candidate per run name. All publishers copy only those elected
bytes to `result.zip`, so concurrent Volume commits cannot select different
archives. The Dict is a compute-side publication registry, not the job database.
[Modal: Dicts](https://modal.com/docs/guide/dicts)

The ZIP contains an explicit allowlist of final outputs, including:

- the exact submitted PDB and normalized parameters;
- production trajectory, structure, topology and analysis outputs;
- provenance such as app/version and timestamps;
- a manifest and checksums; and
- a sanitized `run.log`.

It does not recursively package the working directory, credentials, internal
storage paths, databases, or large shared caches.

Modal Volume storage is authoritative. Intermediate outputs, scientific cache
files, and databases remain on Modal and are not downloaded to the Linux host.
The configurable local directory caches final ZIPs only. It is a size-bounded
LRU cache with atomic publication and size/SHA-256 verification:

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
| `BIOMODALS_STATE_DIR` | `.biomodals/state` | Durable SQLite directory |
| `BIOMODALS_CACHE_DIR` | `.biomodals/cache` | Rebuildable final-ZIP cache directory |
| `BIOMODALS_CACHE_MAX_BYTES` | `10737418240` | Local cache target (10 GiB) |
| `BIOMODALS_FRONTEND_URL` | `http://localhost:5173` | Base URL written into one-time links |
| `BIOMODALS_ALLOWED_ORIGIN` | frontend URL | Exact accepted browser Origin |
| `BIOMODALS_SECURE_COOKIES` | `false` | Use secure `__Host-` session cookies behind HTTPS |
| `BIOMODALS_MODAL_ENVIRONMENT` | `main` | Explicit Modal Environment for lookup/storage |
| `BIOMODALS_GROMACS_APP` | `Gromacs` | Deployed GROMACS Modal App name |
| `BIOMODALS_GROMACS_ACTIVE_LIMIT` | `2` | Active jobs admitted per user |
| `BIOMODALS_RECONCILE_SECONDS` | `10` | Modal reconciliation interval |
| `BIOMODALS_INTERMEDIATE_RETENTION_DAYS` | unset | Positive retention enables cleanup of published runs' intermediates |

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
| Modal Dict as an atomic final-archive publication registry | **Selected.** It elects one compute-side candidate without moving identity or job state into Modal. |
| PostgreSQL and multiple API workers | Deferred. It adds operations without benefit at the current scale; it becomes necessary before horizontal API scaling. |
| SSE/WebSockets or Modal log streaming | Deferred. Polling coarse SQLite state is enough for v1, and the public Modal call API has no supported log stream. |
