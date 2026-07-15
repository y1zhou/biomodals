# Biomodals API service architecture

Status: research and provisional recommendation
Research date: 2026-07-15
Scope: HTTP ingress and job orchestration for GROMACS, AlphaFold 3, and future
Biomodals apps and workflows

## Decision summary

Biomodals does **not** need a Modal Web Function for every compute app. An
ordinary FastAPI process can look up a deployed Modal Function with
`modal.Function.from_name()`, submit it with `.spawn()`, and later poll or
cancel the resulting `FunctionCall`. Modal explicitly documents this pattern
for codebases outside Modal and for multiple loosely coupled Modal Apps with
distinct deployment lifecycles. Function lookup requires a *deployed* App; it
does not work against an ephemeral App created by `modal serve`.
[Modal: invoking deployed Functions](https://modal.com/docs/guide/trigger-deployed-functions)

If Modal hosts the HTTP ingress, Biomodals needs one Web Function for that
ingress. Because this is a multi-route job API, `@modal.asgi_app` is the
appropriate primitive. `@modal.fastapi_endpoint` is intended for a simple,
single handler; it replaced the deprecated `@modal.web_endpoint` name.
[Modal: `fastapi_endpoint`](https://modal.com/docs/sdk/py/latest/modal.fastapi_endpoint)
[Modal: `asgi_app`](https://modal.com/docs/sdk/py/latest/modal.asgi_app)

The provisional target is therefore:

1. one versioned, unified HTTP control plane, assembled from a separate
   `APIRouter` or router factory for each app or workflow;
2. separately deployed Modal compute Apps (`Gromacs`, `AlphaFold3`, and so on),
   reached through narrow backend adapters and deployed-function lookups; and
3. a replaceable ingress host: use one lean Modal `@asgi_app` deployment first,
   while retaining the same FastAPI application factory for external hosting.

This is a unified **API surface**, not one giant Modal App. It centralizes the
client contract, authentication, quotas, job metadata, and OpenAPI document
without coupling every scientific image and compute function to one atomic
deployment.

## What Modal Web Functions do and do not provide

Modal calls HTTP-exposed Functions “Web Functions.” They are an additional
ingress mechanism for clients that speak HTTP; they are not required for
Python code using the Modal client. `@modal.fastapi_endpoint` wraps one handler
in FastAPI, while `@modal.asgi_app` serves a complete ASGI application and is
the documented choice for multiple routes. Web Functions scale their
containers on demand and may cold-start on the first request; `@modal.concurrent`
allows one ASGI container to process multiple requests before Modal scales out.
[Modal: Web Functions](https://modal.com/docs/guide/webhooks)
[Modal: input concurrency](https://modal.com/docs/guide/concurrent-inputs)
[Modal: cold starts](https://modal.com/docs/guide/cold-start)

All Modal Web Function types have a 150-second maximum HTTP request timeout.
Modal's documented solution for long jobs is exactly the pattern already used
here: spawn a compute Function, return its call identifier immediately, and
poll another route. A GROMACS or AlphaFold request should never hold the HTTP
connection open for the scientific run.
[Modal: Web Function request timeouts](https://modal.com/docs/guide/webhook-timeouts)

`requires_proxy_auth=True` protects Web Functions at Modal's edge with
`Modal-Key` and `Modal-Secret` headers. Modal also documents implementing
ordinary Bearer authentication inside FastAPI. Proxy auth is convenient for a
small set of machine clients, but making those Modal-specific headers part of
the public contract increases lock-in. This note therefore treats proxy auth
as an optional edge layer, not as the long-term user or tenant identity model.
[Modal: proxy tokens](https://modal.com/docs/guide/webhook-proxy-auth)
[Modal: conventional token authentication](https://modal.com/docs/guide/webhooks#token-based-authentication)

An external control plane uses Modal API tokens instead. Those tokens select
the workspace and are read from the active Modal profile or
`MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`. They belong only on the trusted server;
API clients should never receive them.
[Modal: lookup authentication](https://modal.com/docs/guide/trigger-deployed-functions#authentication)

## Does a local FastAPI server suffice?

For development, a single-user workstation, or a private laboratory service:
yes. The current external factory can run under Uvicorn and invoke a deployed
Modal Function without exposing any Modal Web Function. The compute still runs
on Modal; only request validation, authentication, and job bookkeeping run
locally.

For an Internet-facing production service: a laptop-local process does not
suffice operationally. The FastAPI application itself is sufficient, but it
must run on durable infrastructure that supplies TLS, startup and restart
management, replication, monitoring, and a persistent metadata store. These
are the production concerns FastAPI identifies beyond starting Uvicorn.
[FastAPI: deployment](https://fastapi.tiangolo.com/deployment/)
[FastAPI: running a server manually](https://fastapi.tiangolo.com/deployment/manually/)

A non-Modal FastAPI deployment offers the least ingress lock-in and can remain
warm continuously. In exchange, Biomodals must operate that service and its
database. A Modal-hosted `@asgi_app` supplies HTTPS ingress, scale-to-zero,
horizontal scaling, logs, and optional proxy auth with less infrastructure,
but the control plane can cold-start and its wrapper, authentication, and
short-lived state primitives are Modal-specific.

## Current branch

The branch already has useful seams for either hosting choice:

- [`service/api.py`](../../src/biomodals/service/api.py) constructs shared
  FastAPI behavior without knowing about GROMACS or Modal.
- [`service/gromacs.py`](../../src/biomodals/service/gromacs.py) defines a
  provider-neutral `JobBackend` protocol and a Modal adapter. Its
  `create_deployed_app()` factory looks up a deployed Function for external
  Uvicorn hosting.
- [`gromacs_api_app.py`](../../src/biomodals/app/service/gromacs_api_app.py)
  exposes the same FastAPI routes through `@modal.asgi_app` and proxy auth.

There is one important coupling in the current Modal wrapper: it calls
`App.include(gromacs_app.app)` and directly captures `run_gromacs_job`. Modal
documents `App.include()` as merging another App's objects into the receiving
App. Because an App is the atomic deployment unit, a change to any Function
updates all Functions in that App. Each Function still has an independent
autoscaling container pool, so merging Apps does not produce a shared compute
pool; it mainly creates deployment and release coupling.
[Modal: `App.include`](https://modal.com/docs/sdk/py/latest/modal.App#include)
[Modal: Apps and independent Function scaling](https://modal.com/docs/guide/apps)
[Modal: App deployment unit](https://modal.com/docs/guide/managing-deployments#no-op-deployments-and-rollovers)

For a second backend such as AlphaFold 3, repeating this wrapper would preserve
isolation but fragment the public API. Including GROMACS, AlphaFold 3, and all
future compute Apps in one API App would unify the URL but create an
increasingly broad atomic deployment. Neither is necessary: deploy the compute
Apps separately and let the control plane look them up by configured App and
Function name.

## Decision matrix

| Option | HTTP and client experience | Scaling and cold starts | Deployment and failure boundary | Auth and lock-in | Job state | Fit |
|---|---|---|---|---|---|---|
| Local FastAPI plus deployed Function lookup | One API on the local/private network; no Modal URL | API stays warm while the process lives; compute Functions still autoscale and cold-start independently | Local process is a single failure point; compute Apps can deploy separately | Standard HTTP auth is easy; server still needs Modal SDK credentials | SQLite is viable only for one local process; Modal call output remains short-lived | Excellent development/private-lab option; not a public production deployment by itself |
| Externally hosted FastAPI plus deployed Function lookup | One stable domain and OpenAPI; hosting platform is replaceable | Determined by chosen host; Modal compute scaling is unchanged | Control plane and each compute App have independent releases and failures | Lowest client-facing Modal lock-in; server-side Modal adapter remains | Use a normal durable database, especially with multiple workers | Best when portability, enterprise auth, audit history, or existing infrastructure outweighs ops cost |
| One `fastapi_endpoint` per operation | Many small Modal URLs/handlers; fragmented documentation for a job lifecycle | Each handler has its own Function pool and potential cold start | Strong handler isolation but many deployments and contracts | Proxy tokens are simple but Modal-specific | Must build polling and registry behavior across handlers | Poor fit for a multi-route jobs API; useful only for isolated callbacks or tiny endpoints |
| One Modal `asgi_app` per scientific app/workflow (current direction) | Separate base URL and OpenAPI per app | Independent API pools and cold starts; independent route-level capacity | Good API failure isolation; repeated wrappers can still include and couple compute code | Proxy auth repeated at every service; clients learn multiple endpoints | Metadata and policies tend to fragment by service | Reasonable when apps have different owners, trust boundaries, or SLOs; costly as the default public surface |
| One Modal `asgi_app` that includes every compute App | One domain and OpenAPI | One lean API pool, but compute Functions still have independent pools | Largest atomic deploy; an incompatible API/build change has broad blast radius | Central auth, but ingress and client headers can be Modal-specific | Central registry is possible | Avoid: it confuses unified routing with unified deployment |
| **One unified Modal `asgi_app`, separate deployed compute Apps, lookup adapters** | **One domain, version, auth policy, and OpenAPI; per-app routers remain modular** | **One lean control-plane pool; each compute Function scales/cold-starts independently** | **One ingress failure domain, but independent compute deployments and images** | **Moderate lock-in now; keep standard auth and app factories so ingress can move later** | **One repository interface; Modal Dict for short MVP retention or SQL for durable history** | **Recommended starting architecture** |

## Unified API versus one server per app

The strongest reason for a unified API is consistency, not shared compute. One
control plane can own:

- a single versioned URL and OpenAPI document;
- one authentication, authorization, quota, and request-size policy;
- idempotent submission and globally unique public job IDs;
- common job status, cancellation, result-manifest, and error envelopes; and
- one place to record users, timestamps, provenance, retention, and audit data.

FastAPI's `APIRouter` is designed for this shape: routes can remain in separate
modules while the main app applies prefixes, tags, responses, and shared
dependencies such as authentication.
[FastAPI: bigger applications and `APIRouter`](https://fastapi.tiangolo.com/tutorial/bigger-applications/)

The strongest reasons for one API server per app are genuine isolation needs:
different owners, authentication realms, regulatory boundaries, release
cadences, availability targets, or an app-specific ingress dependency so large
that it should not enter the shared control-plane image. GROMACS and AlphaFold
having different GPU images is *not* by itself such a reason: those images
belong to their compute Functions, whose container pools scale independently
of the HTTP Function.

A unified router is a larger ingress failure domain. This can be mitigated by
keeping the image and route handlers thin, putting scientific execution behind
separate adapters, deploying with Modal's rolling strategy, and splitting out
only a service that later demonstrates a distinct SLO or trust boundary.
Modal identifies the App as the deployment unit and recommends rolling
deployment for production.
[Modal: managing deployments](https://modal.com/docs/guide/managing-deployments)

## Job execution and persistence

All hosting options can use the same detached-job flow:

1. validate and durably identify the request;
2. resolve the configured deployed Function;
3. call `.spawn()` and save its `FunctionCall.object_id` behind an opaque public
   job ID;
4. poll with `FunctionCall.from_id(...).get(timeout=0)`; and
5. cancel with `FunctionCall.cancel()`.

These operations are first-class Modal APIs, so a Web Function does not add job
execution capabilities; it only supplies HTTP ingress.
[Modal: detached invocation and polling](https://modal.com/docs/guide/trigger-deployed-functions#invocation-patterns)
[Modal: `FunctionCall`](https://modal.com/docs/sdk/py/latest/modal.FunctionCall)

Modal state is adequate only if the product explicitly accepts short history:
Function inputs and outputs are retained for up to seven days, and current
`modal.Dict` entries expire after seven days without a read or write. Queue
partitions default to 24 hours. Volumes remain until deleted and are suitable
for large scientific artifacts, but they do not replace a queryable job
database.
[Modal: data retention](https://modal.com/docs/guide/security#data-retention)
[Modal: `Dict` lifetime](https://modal.com/docs/sdk/py/latest/modal.Dict#lifetime-of-a-dict-and-its-items)
[Modal: `Queue` lifetime](https://modal.com/docs/sdk/py/latest/modal.Queue#lifetime-of-a-queue-and-its-partitions)

Do not put a multi-writer SQLite job database on a shared Modal Volume. Modal
warns that concurrent writes to the same files use last-write-wins semantics
and that Volumes are not a good fit for concurrent modification of one file.
Use SQLite only for a single-process local control plane; use a service database
such as PostgreSQL when the API has multiple containers or workers.
[Modal: Volume consistency](https://modal.com/docs/guide/volumes#filesystem-consistency)

Separating deployments introduces an API-contract risk: unpinned Function
lookups target the latest App version and may temporarily reach an older
version during a rolling deployment. Version-pinned lookup is limited to Team
and Enterprise plans. Keep each compute entrypoint backwards compatible,
validate return values in the adapter, and add contract tests across API and
compute deployments.
[Modal: version-pinned lookups](https://modal.com/docs/guide/trigger-deployed-functions#version-pinned-lookups)

## Provisional target shape

```text
client
  |
  v
one FastAPI control plane
  /v1/gromacs/jobs       -> Gromacs adapter    -> deployed Gromacs Modal App
  /v1/alphafold3/jobs    -> AlphaFold3 adapter -> deployed AlphaFold3 Modal App
  /v1/<workflow>/jobs    -> workflow adapter   -> deployed workflow/worker Apps
            |
            +-> job metadata repository
            +-> artifact manifests / signed download links
```

Start with app-specific route prefixes rather than a lowest-common-denominator
`POST /jobs?kind=...`. Share only the job envelope and lifecycle semantics that
are actually common; retain app-specific validated inputs and outputs. This
keeps one coherent API without erasing differences between a PDB upload,
AlphaFold JSON input, and a multi-step workflow.

Deploy this control plane as a small Modal `@asgi_app` first because the repo
already has the wrapper, asynchronous Modal adapter, and proxy-auth path. Stop
including scientific Apps in it; deploy those independently and use
`Function.from_name()` through configured adapters. Keep the external FastAPI
factory working and use ordinary Bearer/OIDC authentication at the application
layer if client portability matters. Treat Modal proxy auth as optional
defense-in-depth or an internal MVP credential.

Use `modal.Dict` only if “jobs disappear after a week of inactivity” is an
accepted product rule. Otherwise define a job repository boundary now and back
it with durable SQL before scaling the API past one container.

## Questions to resolve in the design grill

1. Who are the callers: one trusted operator, laboratory users, other backend
   services, or untrusted public users?
2. Must one credential access every scientific app, or are permissions and
   quotas app-specific?
3. How long must job history, input provenance, and audit events remain
   queryable: hours, seven days, months, or indefinitely?
4. Is a job submission required to be idempotent across client retries?
5. Do clients need artifact download through the API, signed external links, or
   only a manifest for later CLI retrieval?
6. Is cancellation best-effort, or must the API prove that all child Functions
   in a workflow stopped?
7. Must API and compute deployments be released atomically, or can entrypoints
   follow a backwards-compatible contract?
8. Which routes, if any, genuinely need a distinct trust boundary, owner, SLO,
   region, or custom domain?
9. Is avoiding Modal-specific client headers important enough to own OIDC or
   API-key lifecycle now?
10. What is the expected request rate and burst profile, and is control-plane
    cold-start latency acceptable?
