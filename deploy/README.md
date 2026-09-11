# Production API deployment examples

These files document three alternative production shapes. They are examples
only: review and copy one through the host's normal deployment process. Do not
install a unit, start a container, or replace the live service merely by
checking out this repository.

All three examples:

- expose the API only on `127.0.0.1:4100` for a same-origin HTTPS reverse
  proxy;
- use `https://biomodals.example.com` as the browser origin;
- keep durable SQLite state separate from the rebuildable Result cache; and
- contain only boot-critical process settings. Modal App names are startup-only
  configuration and require a restart after changes. Unless an explicit
  process override makes a field read-only, the Admin interface can change the
  Modal Environment, exact App deployment versions, admission limits, and
  per-Tool Job-log visibility.

The service host or container also needs GNU `tar` and `zstd` to rebuild
AlphaFold3 Result archives from the remote output Volume. The included
`Containerfile` installs both; native systemd deployments must install them
through the host package manager.

Replace every Modal token placeholder before starting a service. The backend
refuses to start when either credential is missing.

A fresh database initially uses `Gromacs`, `AlphaFold3`, and
`HumanizationWorkflow` at version `1`. These are defaults, not a guarantee that
those historical deployments support the current API. Startup resolves the
exact configured deployments of all three Tools before reporting ready.
Identify deployments built from the matching backend release, then save
their positive versions in Admin → Modal → Tools:

```console
modal app history Gromacs --env <environment> --json
modal app history AlphaFold3 --env <environment> --json
modal app history HumanizationWorkflow --env <environment> --json
```

The production examples deliberately do not set those deployment versions as
process environment variables, because doing so would make the Admin fields
read-only. Supply matching versions from a private `BIOMODALS_API_CONF_ENV`
file for first boot; do not depend on version `1` merely existing. Set
`BIOMODALS_GROMACS_APP_VERSION`, `BIOMODALS_ALPHAFOLD3_APP_VERSION`, and
`BIOMODALS_HUMANIZATION_APP_VERSION` there, along with the intended
`BIOMODALS_MODAL_ENVIRONMENT`. The file must be owned by the service user with
mode `0600`; container deployments must mount it privately and use its
in-container path. These file values remain database-overridable defaults.
The humanization workflow deployment includes its four dependent apps; it
does not require separate deployments of those generators.

## First-administrator bootstrap

For unattended first deployment, generate an Argon2id password hash locally:

```console
uv run biomodals api admin hash-password
```

The command prompts twice with input hidden, applies the normal password
policy, and prints only the hash to stdout. It does not access the database,
load service configuration, or contact Modal. Do not pass the password as a
command argument or put it in shell history.

Place the email and generated hash in the private file selected by
`BIOMODALS_API_CONF_ENV`:

```dotenv
BIOMODALS_DEFAULT_ADMIN_EMAIL=admin@example.com
BIOMODALS_DEFAULT_ADMIN_PASSWORD_HASH='paste-generated-argon2id-hash-here'
```

Replace the hash placeholder with the complete output, including its `$`
characters. Use single quotes when assigning it in a shell. For Compose,
mount the private configuration file and point `BIOMODALS_API_CONF_ENV` at its
container path rather than embedding a hash in interpolated YAML. Restrict
the file to its service owner with mode `0600`; hashes still allow offline
password guessing if stolen. No plaintext-password setting is supported.

When the API starts, it creates an enabled administrator named
`Administrator`, using the configured default per-User Job limit, only if no
administrator exists. Log in through the ordinary `/login` page with the
original password, not the hash. No Password Link or Session is created by
bootstrap. Remove the bootstrap values after successful setup; changing them
later never resets an existing password or recovers a disabled account.

Any existing administrator, including disabled or pending-setup accounts,
causes both values to be ignored. Without an administrator, omitting both
values preserves manual provisioning; supplying only one, an invalid email,
or an invalid hash stops startup. An existing non-admin with the same email
is an error, not an implicit promotion. Use the existing offline admin
commands for recovery. No database migration is needed.

## Native systemd

Install the project and API dependencies in the path used by
`biomodals-api.service`, using Python 3.12+ and the uv version range declared
in `pyproject.toml` (currently `>=0.12,<0.13`):

```console
cd /opt/biomodals/current/biomodals
uv sync --locked --no-dev --extra api
```

Repeat the locked sync after updating the checkout; restarting an old virtual
environment does not install newly declared dependencies. The runtime now
includes `zstandard` and requires Polars 1.44.2 or newer. Install the API extra
even if a development checkout previously worked through its dev dependencies.
The build backend accepts `uv_build>=0.9,<1.0`; let the package build resolve
that requirement rather than retaining the former `<0.10.0` constraint.
For containers, rebuild the image rather than only restarting it.

Copy `biomodals-api.service` to `/etc/systemd/system/`, replace its token
placeholders, and restrict the installed unit to root before reloading systemd.
Create the `biomodals` user and the two configured data directories first.

## Docker Compose

`compose.yaml` builds `Containerfile` from the repository root and uses
named volumes for state and cache. Supply the two Modal credentials in the
invoking environment:

```console
export MODAL_TOKEN_ID=replace-with-service-token-id
export MODAL_TOKEN_SECRET=replace-with-service-token-secret
docker compose -f deploy/compose.yaml up --detach --build
```

Compose validates that both variables are present before creating the
container. Do not store real values in this tracked file.

## Podman Quadlet

Build the local image from the repository root:

```console
podman build --file deploy/Containerfile \
  --tag localhost/biomodals-api:latest .
```

Create the two host directories owned by UID and GID `10001`, copy
`biomodals-api.container` to `/etc/containers/systemd/`, and replace its
token placeholders. After `systemctl daemon-reload`, systemd generates
`biomodals-api.service`; the Quadlet's `[Install]` section starts it at boot.
Do not run `systemctl enable` on the generated service.

For all three options, wait for
`http://127.0.0.1:4100/api/v1/ready` before exposing a release. The
[deployment runbook](../docs/deployment/mvp-runbook.md) covers the frontend
proxy, pre-release schema transition, isolation, verification, and rollback.
