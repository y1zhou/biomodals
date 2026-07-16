# biomodals

Bioinformatics tools running on modal.
Note that this repository is heavily refactored from [the upstream repository](https://github.com/hgbrian/biomodals).
All new apps have the `_app.py` suffix to distinguish from the original ones.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/y1zhou/biomodals)

## Installation

```bash
git clone https://github.com/y1zhou/biomodals.git
cd biomodals
pip install .
biomodals --help
```

Or alternatively, use [uv](https://github.com/astral-sh/uv), e.g.:

```bash
git clone https://github.com/y1zhou/biomodals.git
cd biomodals
uv run biomodals --help
```

## Getting started

To see a list of all available commands, run:

```bash
biomodals --help
```

To list and inspect apps:

```bash
biomodals app list
biomodals app help <app-name>
```

To list and inspect workflows:

```bash
biomodals workflow list
biomodals workflow help <workflow-name>
```

To run a workflow, pass workflow-specific flags after `--`:

```bash
uv run biomodals workflow run ppiflow --dry-run -- \
  --task-yaml examples/data/ppiflow_workflow_task.yaml \
  --steps-yaml examples/data/ppiflow_workflow_steps.yaml
```

## API server

The API is one local FastAPI control plane for GROMACS and future workloads.
It submits scientific work to separately deployed Modal Apps; it is not a
Modal web endpoint.

Install the API dependencies and deploy the GROMACS compute App in the Modal
Environment that the server will use:

```bash
uv sync --extra api
MODAL_ENVIRONMENT=main uv run biomodals app deploy gromacs
```

For local frontend development, keep durable SQLite state and the final-result
cache in separate directories, then start exactly one Uvicorn worker:

```bash
export BIOMODALS_STATE_DIR="$PWD/.biomodals/state"
export BIOMODALS_CACHE_DIR="$PWD/.biomodals/cache"
export BIOMODALS_FRONTEND_URL="http://localhost:5173"
export BIOMODALS_ALLOWED_ORIGIN="http://localhost:5173"
export BIOMODALS_MODAL_ENVIRONMENT="main"

uv run biomodals api serve
```

The server listens on `127.0.0.1:8000` by default. Use `--host` or `--port`
to override either value; the command keeps the required worker count at one.

Configure the frontend development server to proxy `/api` to
`http://127.0.0.1:8000`. The browser then uses ordinary same-origin,
HTTP-only session cookies. `BIOMODALS_SECURE_COOKIES` defaults to `false` for
local HTTP; set it to `true` behind the production HTTPS reverse proxy.

There is no public registration. An administrator provisions an employee and
delivers the printed one-hour password link through a trusted internal channel:

```bash
uv run biomodals-admin create-user alice@example.com \
  --display-name "Alice Example"
```

The admin command must use the same `BIOMODALS_STATE_DIR` and
`BIOMODALS_FRONTEND_URL` as the server. Related commands are
`biomodals-admin reset-password EMAIL` and
`biomodals-admin disable-user EMAIL`.

On the department server, run this command under a dedicated Linux and Modal
service user, set an explicit `BIOMODALS_MODAL_ENVIRONMENT`, and supervise the
single process with systemd. The local cache contains only final ZIP files;
Modal Volume storage remains authoritative for results and intermediates.
