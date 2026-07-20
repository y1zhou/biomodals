# MVP deployment runbook

This is documentation and example configuration only. It does not authorize an
agent or script to edit the live Caddyfile, install a systemd unit, start a
production service, deploy a Modal App, or copy frontend files into
`/srv/biomodals.example.com`.

## Host layout

Run the API as a dedicated `biomodals` Linux user. The recommended boundaries
are:

- application checkout: `/opt/biomodals/current/biomodals`;
- native production unit: `/etc/systemd/system/biomodals-api.service`;
- production SQLite state: `/var/lib/biomodals/production/state`;
- production Result cache: `/var/cache/biomodals/production`;
- pre-release SQLite state: `/var/lib/biomodals/prerelease/state`;
- pre-release Result cache: `/var/cache/biomodals/prerelease`; and
- production frontend files: `/srv/biomodals.example.com`.

The tracked files under `deploy/` are production examples, not live
configuration. Copy the selected systemd or Quadlet definition through an
administrator-approved path, replace its credential placeholders, and restrict
the installed file to root. Docker Compose receives credentials from its
invoking environment. The top-level `.env.example` is only for local
development.

Production and pre-release must never select the same service definition,
database, or cache directory. A pre-release copy must use
`https://beta.biomodals.example.com`, loopback port `4144`, and the pre-release
paths above. An intentionally shared Modal Environment does not change this
requirement.

## API service example

[`deploy/README.md`](../../deploy/README.md) documents three alternatives:
native systemd, Docker Compose, and a Podman Quadlet. The native unit keeps all
boot-critical environment values in the unit and documents one FastAPI worker,
a fixed working directory, private umask, restart-on-failure, and journald
output. Review and install only one alternative. A pre-release service must use
its own copied configuration, host paths, public URL, and port.

Before directing a browser to the production API, verify it on port `4100`.
Use port `4144` for the development or pre-release service:

```console
curl --fail http://127.0.0.1:4100/api/v1/health
curl --fail http://127.0.0.1:4100/api/v1/ready
```

Liveness only proves that the event loop responds. Readiness additionally
proves that startup preflight finished and local SQLite, cache storage, the
artifact worker, and reconciliation are usable. Readiness does not contact
Modal on each request.

Use `journalctl -u <unit-name>` to inspect startup failures and request IDs.
Never paste Password Links, cookies, PDB data, or Modal secrets into a support
record.

## Static frontend and reverse proxy

Build the frontend repository with its committed Bun lockfile, then stage the
contents of `dist/` into `/srv/biomodals.example.com` as one reviewed release. The
production browser always calls same-origin `/api/*`; it does not embed a
separate API hostname.

The live Caddy configuration remains host-owned. A reviewed production
configuration needs the equivalent behavior shown below, adapted by the
Administrator rather than copied blindly:

```caddyfile
biomodals.example.com {
	encode zstd gzip

	@api path /api/* /docs* /openapi.json /redoc*
	handle @api {
		reverse_proxy 127.0.0.1:4100
	}

	header {
		Content-Security-Policy "default-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
		X-Content-Type-Options nosniff
		Referrer-Policy strict-origin-when-cross-origin
		Permissions-Policy "camera=(), microphone=(), geolocation=()"
		Strict-Transport-Security "max-age=31536000"
	}

	@assets path /assets/*
	header @assets Cache-Control "public, max-age=31536000, immutable"
	@document path / /index.html
	header @document Cache-Control "no-cache"

	root * /srv/biomodals.example.com
	try_files {path} /index.html
	file_server
}
```

Do not add HSTS `includeSubDomains` or preload until every affected subdomain is
audited. Confirm that production serves hashed static assets and never Vite HMR,
React Refresh, or source modules.

## Update and rollback checks

Before every deployment, record exactly two last-deployed commit hashes: one
for this backend repository and one for the frontend repository. Verify both
locally and generate a change-aware manual checklist from those baselines.
When there is no prior deployment, treat the complete candidate as changed.

The checklist must order these actions:

1. Run the automated backend, frontend, OpenAPI, and Playwright gates against
   the intended pair of revisions.
2. Review configuration-file ownership/mode and the effective public URL,
   cookie mode, state path, cache path, Modal Environment, App name, and limits.
3. Deploy a changed GROMACS App before an API that calls its changed contract;
   otherwise explicitly mark that paid deployment step irrelevant.
4. Start or restart the candidate API and wait for readiness before exposing
   the frontend candidate.
5. Run only the manual smoke checks relevant to the changed surfaces. If the
   App, adapter, or effective Modal configuration changed, an Administrator
   must approve the cost and submit one smallest valid, unmistakably named
   real-Modal Job.
6. Verify one invocation of each expected deployed Function, exact public stage
   transitions, finalization, prepared download, and the Result ZIP allowlist.
7. Stop on any unexpected invocation, stage, archive member, permission error,
   or configuration error. Capture the Job ID, request ID, safe stage, and
   timestamps before retrying; do not blindly resubmit paid work.
8. Publish the reviewed static frontend only after the API checks pass, then
   verify SPA fallback, login/logout, and same-origin API access.

Rollback restores the prior backend checkout and frontend static release as a
pair, restores the prior service or container configuration if it changed,
restarts the one API process, and waits for readiness. Do not point a
rolled-back binary at an
incompatible newer database. During the current disposable-state phase, stop
the pre-release service and explicitly remove only the exact pre-release
database named by its configuration when a schema reset is required. Never
apply that reset instruction to production state or cache.
