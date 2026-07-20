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
- contain only boot-critical process settings, so the Modal Environment, App
  name, and admission limits remain editable in the Admin interface.

Replace every Modal token placeholder before starting a service. The backend
refuses to start when either credential is missing.

## Native systemd

Install the project and API dependencies in the path used by
`biomodals-api.service`:

```console
cd /opt/biomodals/current/biomodals
uv sync --locked --no-dev --extra api
```

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
proxy, pre-release isolation, verification, and rollback.
