# BioModals

BioModals packages computational biology tools as independently deployable
[Modal](https://modal.com/) apps and composes them into durable workflows.

This repository also contains a reusable execution kernel and an optional
FastAPI service. The service powers the separate
[BioModals frontend](https://github.com/y1zhou/biomodals-frontend).

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/y1zhou/biomodals)

## What is in this repository?

BioModals supports three ways to run scientific work:

| Goal | Start here |
| --- | --- |
| Run one packaged tool from a terminal | `biomodals app` |
| Run a multi-tool DAG from a terminal | `biomodals workflow` |
| Offer a tool through the web interface | `biomodals api` and the frontend repository |

The same execution kernel provides durable task state, dependency scheduling,
Modal call tracking, cancellation, and recovery for all three paths.

## Architecture

There are two main execution paths:

```text
CLI -> deployed app/workflow coordinator -> biomodals.execution -> Modal functions

Browser -> static frontend -> FastAPI service -> biomodals.execution -> Modal functions
```

`biomodals.execution` is an embedded Python library, not a central scheduler
service. Each coordinator embeds it and owns the database for the work it
coordinates.

| Component | Responsibility |
| --- | --- |
| `biomodals.app` | Independently deployed scientific Modal apps |
| `biomodals.workflow` | DAGs that compose app functions into larger pipelines |
| `biomodals.execution` | Shared task, DAG, SQLite, scheduling, and Modal-call mechanics |
| `biomodals.service` | Accounts, web Jobs, administration, Results, and HTTP routes |

The kernel owns execution state only. Scientific cache validation, input and
output formats, and publication rules remain with each app or workflow. User
accounts and other non-execution service data remain in `biomodals.service`.

## Quick start

BioModals requires Python 3.12 or newer, a Modal account, and configured Modal
credentials. The examples below use [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/y1zhou/biomodals.git
cd biomodals
uv sync
uv run biomodals --help
```

`pip install .` is also supported when uv is not available.

Discover the installed apps and workflows before running paid compute:

```bash
uv run biomodals app list
uv run biomodals app help <app-name>

uv run biomodals workflow list
uv run biomodals workflow help <workflow-name>
```

## Apps and workflows

An **app** packages one scientific tool and its Modal functions. Apps can be
deployed and run independently.

```bash
uv run biomodals app deploy <app-name> --env <environment>
uv run biomodals app run <app-name> \
  --environment <environment> -- <app-specific-options>
```

A **workflow** constructs a scientific DAG from one or more tools. Workflow
flags also follow `--` so the BioModals CLI can separate them from its own
options.

```bash
uv run biomodals workflow deploy <workflow-name> --env <environment>
uv run biomodals workflow run <workflow-name> \
  --environment <environment> -- <workflow-specific-options>
```

A workflow deployment includes the callable functions declared by its
dependency apps. Those apps do not need separate deployments for that workflow.
Deploy them separately only when they must also run as standalone apps.

Use `--dry-run` to build and print a workflow DAG without resolving a Modal
deployment or starting remote work:

```bash
uv run biomodals workflow run ppiflow --dry-run -- \
  --task-yaml examples/data/ppiflow_workflow_task.yaml \
  --steps-yaml examples/data/ppiflow_workflow_steps.yaml
```

Normal app and workflow runs target a deployed coordinator. The CLI resolves
an exact deployment version before admitting work, even when the latest
version is selected implicitly.

Pass `--development` to run current app or workflow source through an
ephemeral Modal app. Development runs are useful while editing code, but they
do not provide durable cross-command recovery.

An app entrypoint that has not yet adopted a deployment coordinator fails
closed in normal mode. Use `--development` explicitly for that source-backed
path.

## Durable execution

A coordinated launch prints the identity needed to inspect the run from a
different process:

```text
Deployment Identity: <environment>/<deployment-name>/v<version>
Execution Run ID: <uuid>
Coordinator FunctionCall ID: <provider-call-id>
```

Use the deployment identity and Execution Run ID with the lifecycle commands:

```bash
biomodals run status \
  --environment <environment> \
  --deployment-name <deployment-name> \
  --deployment-version <version> \
  --execution-run-id <uuid>
```

The same locator options apply to `biomodals run cancel` and
`biomodals run resume`.

`resume` continues a suspended run. It never retries a task that failed
conclusively. To retry missing work, create a linked successor run:

```bash
biomodals run restart \
  --environment <predecessor-environment> \
  --deployment-name <predecessor-deployment-name> \
  --deployment-version <predecessor-version> \
  --execution-run-id <predecessor-uuid> \
  --target-environment <target-environment> \
  --target-deployment-name <target-deployment-name> \
  --target-deployment-version <target-version>
```

Restart verifies that the result-affecting plan is unchanged. It reuses valid
publications and schedules only conclusively missing work. Repeating the
original launch without `--restart-from` creates a separate root run.

Durable state follows the coordinator that owns the run:

| Caller | Execution database |
| --- | --- |
| Direct CLI app run | One remote, per-run app ledger |
| CLI workflow run | One remote, per-run workflow ledger |
| FastAPI service | Shared execution tables inside `service.sqlite3` |

The local CLI does not create an execution database. Provider workers also do
not write SQLite; their owning coordinator records observations and outcomes.

For the complete model, see
[ADR 0006](docs/adr/0006-unified-execution-kernel.md) and the
[unified task scheduler specification](docs/specs/unified-task-scheduler.md).

## Web API

The optional FastAPI service is a single-host control plane for the BioModals
web interface. It currently exposes the GROMACS MD simulation and is designed
to add more workload adapters without changing account or Job routes.

The service owns Users, Sessions, Jobs, runtime settings, Result staging, and
the mapping from a web Job to its kernel Execution Run. It calls established,
separately deployed Modal functions; it is not a Modal web endpoint.

The browser application lives in the
[biomodals-frontend repository](https://github.com/y1zhou/biomodals-frontend).
It uses same-origin `/api` requests and generated OpenAPI types.

### Local API development

Install the API dependencies and deploy the GROMACS app in the Modal
Environment selected by the development configuration:

```bash
uv sync --extra api
uv run biomodals app deploy gromacs --env production
```

Copy [`.env.example`](.env.example), replace both Modal token placeholders,
and keep the private copy out of Git:

```bash
install -m 600 .env.example .env
export BIOMODALS_API_CONF_ENV="$PWD/.env"
uv run biomodals api serve
```

The backend refuses to start unless `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` are present. The secret remains process-only and is never
stored in SQLite or returned by the API.

Relative state and cache paths are resolved from the directory containing the
configured `.env` file. The development example keeps them under the
repository-root `.biomodals/` directory.

Explicit process environment variables override values from the file.
Administrator-editable runtime settings resolve in this order: process
environment, database value, `.env` value, then built-in default.

`BIOMODALS_PUBLIC_URL` is the single browser origin used for Password Links
and mutation Origin checks. The development server listens on
`127.0.0.1:4144` by default and uses one worker.

Configure the frontend development server to proxy `/api` to
`http://127.0.0.1:4144`. The browser can then use same-origin, HTTP-only
Session cookies.

`BIOMODALS_SECURE_COOKIES` defaults to `false` for local HTTP. Set it to
`true` behind the production HTTPS reverse proxy.

### Create the first administrator

There is no public registration. The first User must be created as an
Administrator:

```bash
uv run biomodals api admin create-user alice@example.com \
  --display-name "Alice Example" --admin
```

The command prints a one-hour Password Link for delivery through a trusted
internal channel. It must use the same `BIOMODALS_API_CONF_ENV` or explicit
state configuration as the API process.

Related commands include `reset-password`, `disable-user`, `enable-user`,
`promote-user`, and `demote-user`. The final active Administrator cannot be
disabled or demoted.

### Pre-release database transition

The execution kernel intentionally does not migrate pre-release Job history.
If startup reports service database version 3, stop the API and run:

```bash
uv run biomodals api transition-execution-state --yes
```

The transition preserves Users, Password Links, Sessions, and service and
workload settings. It discards legacy Jobs and their local execution history.

Remote Modal Volumes and workload publications are unchanged. The command
rejects an unexpected source schema instead of guessing how to rewrite it.

## Production deployment

Production examples cover a native systemd service, Docker Compose, and a
Podman Quadlet. They bind one API process to `127.0.0.1:4100` behind a
same-origin HTTPS reverse proxy.

Start with the [deployment examples](deploy/README.md). Use the
[MVP deployment runbook](docs/deployment/mvp-runbook.md) for readiness,
cross-repository checks, manual smoke tests, and rollback guidance.

Run the API as a dedicated Linux and Modal service user. Keep SQLite state
separate from the rebuildable local Result cache. Modal Volume storage remains
authoritative for Results and scientific intermediates.

## Repository map

| Path | Contents |
| --- | --- |
| `src/biomodals/app/` | Scientific Modal apps |
| `src/biomodals/workflow/` | Multi-app scientific workflows |
| `src/biomodals/execution/` | Shared execution kernel |
| `src/biomodals/service/` | FastAPI control plane and workload adapters |
| `src/biomodals/schema/` | Shared scientific input and output schemas |
| `examples/` | Example app and workflow inputs |
| `deploy/` | Production deployment examples |
| `docs/` | Decisions, specifications, research, and runbooks |

Use the documentation by purpose:

- [`docs/adr/`](docs/adr/) records accepted architecture decisions. ADR 0001
  is explicitly superseded by ADR 0006.
- [`docs/specs/unified-task-scheduler.md`](docs/specs/unified-task-scheduler.md)
  is the implementation contract for durable execution.
- [`docs/specs/mvp-release-readiness.md`](docs/specs/mvp-release-readiness.md)
  is the backend and frontend MVP contract.
- [`docs/research/api-service-architecture.md`](docs/research/api-service-architecture.md)
  explains the service boundary and identifies the execution sections
  superseded by ADR 0006.
- [`docs/agents/app-development.md`](docs/agents/app-development.md) and
  [`docs/agents/workflow-development.md`](docs/agents/workflow-development.md)
  route contributors to the current development guidance.

## Project history

This repository is heavily refactored from the
[upstream BioModals project](https://github.com/hgbrian/biomodals). The
`*_app.py` suffix distinguishes the maintained app modules introduced by this
fork from inherited modules.
