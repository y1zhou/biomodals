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

### Local development

Install the API dependencies and deploy the GROMACS compute App in the Modal
Environment used by your development server:

```bash
uv sync --extra api
MODAL_ENVIRONMENT=production uv run biomodals app deploy gromacs
```

Copy [`.env.example`](.env.example), fill in a Modal service-user token, and
point the API at the private file. The copied `.env` is ignored by Git and
must remain readable only by its owner:

```bash
install -m 600 .env.example .env
# Edit .env and replace both Modal token placeholders.
export BIOMODALS_API_CONF_ENV="$PWD/.env"

uv run biomodals api serve
```

Explicit process environment variables override values in the file. Relative
state and cache paths are resolved from the directory containing `.env`.
The example keeps both under the repository-root `.biomodals/` directory.
The backend refuses to start unless both `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` are present. The token secret remains process-only and is
never stored in SQLite or returned by the API. `BIOMODALS_PUBLIC_URL` is the
single browser origin used for Password Links and mutation Origin checks.

The development server listens on `127.0.0.1:4144` by default. Use `--host`
or `--port` to override either value; the command keeps the required worker
count at one.

Configure the frontend development server to proxy `/api` to
`http://127.0.0.1:4144`. The browser then uses ordinary same-origin,
HTTP-only session cookies. `BIOMODALS_SECURE_COOKIES` defaults to `false` for
local HTTP; set it to `true` behind the production HTTPS reverse proxy.

There is no public registration. An administrator provisions an employee and
delivers the printed one-hour password link through a trusted internal channel:

```bash
uv run biomodals api admin create-user alice@example.com \
  --display-name "Alice Example" --admin
```

The admin command must use the same `BIOMODALS_API_CONF_ENV` or explicit state
configuration as the server. Related commands include `reset-password`,
`disable-user`, `enable-user`, `promote-user`, and `demote-user`. The final
active administrator cannot be disabled or demoted, and the first User must be
created with `--admin`.

Administrators can manage Users, per-Tool Job Log visibility, and non-secret
live Modal configuration in the web interface. Runtime setting precedence is
process environment, database Admin value, configured `.env` file, then
built-in default. Process-controlled fields are read-only in the Admin
interface.

### Production deployment

The [production deployment examples](deploy/README.md) cover a native systemd
service, Docker Compose, and a Podman Quadlet. Each example binds the API only
to `127.0.0.1:4100` for a same-origin HTTPS reverse proxy. Copy and review an
example before use; never commit real Modal credentials.

Run the API as one process under a dedicated Linux and Modal service user. Only
boot-critical host values belong in the service definition. Leaving the Modal
Environment, deployed App name, and admission limits out of the process
environment keeps those fields editable in the Admin interface.

The local cache contains rebuildable Result ZIP files; Modal Volume storage
remains authoritative for Results and intermediates. See the
[MVP deployment runbook](docs/deployment/mvp-runbook.md) for readiness,
change-aware manual checks, and rollback guidance.
