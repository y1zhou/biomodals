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

Keep paid-work limits deliberately small during pre-release validation:

```dotenv
BIOMODALS_DEFAULT_USER_ACTIVE_JOB_LIMIT=1
BIOMODALS_GROMACS_ACTIVE_LIMIT=1
BIOMODALS_ALPHAFOLD3_ACTIVE_LIMIT=1
BIOMODALS_HUMANIZATION_ACTIVE_LIMIT=1
BIOMODALS_GLOBAL_ACTIVE_JOB_LIMIT=1
```

These values constrain only the selected service database. They do not
coordinate with another BioModals deployment that targets the same Modal App.

## API service example

[`deploy/README.md`](../../deploy/README.md) documents three alternatives:
native systemd, Docker Compose, and a Podman Quadlet. The native unit keeps all
boot-critical environment values in the unit and documents one FastAPI worker,
a fixed working directory, private umask, restart-on-failure, and journald
output. Review and install only one alternative. A pre-release service must use
its own copied configuration, host paths, public URL, and port.

Before directing a browser to the production API, verify its production unit
on port `4100`:

```console
curl --fail http://127.0.0.1:4100/api/v1/health
curl --fail http://127.0.0.1:4100/api/v1/ready
```

Verify a development or pre-release unit on its separate port `4144`:

```console
curl --fail http://127.0.0.1:4144/api/v1/health
curl --fail http://127.0.0.1:4144/api/v1/ready
```

Liveness only proves that the event loop responds. Readiness additionally
proves that startup preflight finished and local SQLite, cache storage, the
artifact worker, and reconciliation are usable. Readiness does not contact
Modal on each request.

Use `journalctl -u <unit-name>` to inspect startup failures and request IDs.
Never paste Password Links, cookies, PDB data, or Modal secrets into a support
record.

### Pre-release service schema

The service has no migration command or compatibility reader for pre-release
database schemas. If startup rejects a version, stop the API process and first
confirm the selected configuration and exact database path. Point the new build
at a new empty, pre-release-only state directory; retain the old database
separately if an Administrator needs to inspect or copy Users and settings.

Changing host-local service state does not change Modal Volumes or workload
publications. Never resolve an unsupported pre-release schema by deleting or
repointing production state.

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

	header {
		Content-Security-Policy "default-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
		X-Content-Type-Options nosniff
		Referrer-Policy strict-origin-when-cross-origin
		Permissions-Policy "camera=(), microphone=(), geolocation=()"
		Strict-Transport-Security "max-age=31536000"
	}

	@assets path /assets/*
	header @assets Cache-Control "public, max-age=31536000, immutable"
	handle @api {
		reverse_proxy 127.0.0.1:4100
	}
	handle /assets/* {
		root * /srv/biomodals.example.com
		file_server
	}
	handle {
		root * /srv/biomodals.example.com
		route {
			try_files {path} /index.html
			header /index.html Cache-Control "no-cache"
			file_server
		}
	}
}
```

Do not add HSTS `includeSubDomains` or preload until every affected subdomain is
audited. Confirm that production serves hashed static assets and never Vite HMR,
React Refresh, or source modules.

The separate handlers keep API requests out of the SPA fallback and return a
real 404 for missing assets. The document cache header runs **after** the
rewrite, so deep links receive it too. See Caddy's
[SPA routing guidance](https://caddyserver.com/docs/caddyfile/patterns#single-page-apps-spas).

### Password Links and missing browser routes

`/set-password` is a frontend route, not a file or a FastAPI GET endpoint.
A Password Link has the form `/set-password#token=...`; the fragment stays in
the browser. Caddy must serve `index.html` for `/set-password`, then the
frontend submits the password to `POST /api/v1/auth/set-password`. Do not add
an API GET route or move the token into a query parameter to work around a 404.

Check the actual public origin without a token or cookies:

```console
curl --silent --show-error --output /dev/null --write-out '%{http_code} %{content_type}\n' https://biomodals.example.com/
curl --silent --show-error --output /dev/null --write-out '%{http_code} %{content_type}\n' https://biomodals.example.com/set-password
curl --silent --show-error --output /dev/null --write-out '%{http_code} %{content_type}\n' https://biomodals.example.com/login
```

All three should return HTTP 200 HTML. If `/` works but the two deep links
return 404, check that the **loaded production site block**, not only the
development proxy, contains `try_files {path} /index.html` in its static
handler. A `root` plus `file_server` alone does not provide SPA routing.
Keep the `/api/*` handler unchanged and do not use `handle_path` to strip the
API prefix. Validate the administrator-reviewed Caddyfile before an authorized
reload; no frontend rebuild is needed for a routing-only correction.

If HTML returns 200 but the page is blank or shows a client-side not-found
screen, inspect browser console errors and the referenced `/assets/*` files.
Confirm Caddy's root contains the matching release's `index.html` and assets
(the contents of `dist/`, not its parent directory). Use the same public
origin in `BIOMODALS_PUBLIC_URL` when issuing Password Links. A no-token
`/set-password` page can show a missing-link message; that is not a routing
failure. Never share a real Password Link in diagnostic output.

## Jobs stuck before startup

A rejected startup request can leave an older API showing `Running` or
`Cancellation requested` even though no scientific work began. Update and
restart the API to enable failed-startup recovery; the next background pass
closes a confirmed uninitialized Job as `Failed` or `Cancelled`. This recovery
uses the Job's original pinned deployment and does not require a scientific
redeployment, a new submission, or a database edit.

The original deployment must still be reachable. If the root call's outcome
or remote state cannot be confirmed, the API retains uncertainty and admission
ownership. Do not stop the entire shared Modal deployment to clear one Job's
website status, or assume that cancelling its root call stops child work.

## Update and rollback checks

Before every deployment, record exactly two last-deployed commit hashes: one
for this backend repository and one for the frontend repository. Verify both
locally and generate a change-aware manual checklist from those baselines.
When there is no prior deployment, treat the complete candidate as changed.
Record the full 40-character candidate hashes as well. Run the frontend
repository's manually dispatched `Cross-repository checks` workflow with those
candidate hashes and retain its successful workflow URL; branches, tags, and
abbreviated hashes are not acceptable release evidence.

The checklist must order these actions:

1. Run the automated backend and frontend merge gates, then the exact-SHA
   OpenAPI and Playwright gate against the intended candidate pair.
2. Review configuration-file ownership/mode and the effective public URL,
   cookie mode, state path, cache path, Modal Environment, App name, exact App
   deployment version, and limits. Confirm the version against `modal app
   history <app> --env <environment> --json` before any real Submission.
   If the durable operation plan changed, confirm that no active Job from the
   prior plan remains or that the candidate explicitly supports its plan
   version.
3. Deploy every changed Tool App before an API that calls its changed contract;
   otherwise explicitly mark that paid deployment step irrelevant. Confirm
   that the API pins each intended exact deployment version.
4. Start or restart the candidate API and wait for readiness before exposing
   the frontend candidate.
5. Run only the manual smoke checks relevant to the changed surfaces. If the
   App, adapter, or effective Modal configuration changed, an Administrator
   must approve the cost and submit one smallest valid, unmistakably named
   real-Modal Job.
6. For a changed GROMACS path, verify one invocation of each expected deployed
   Function: preparation first; NVT analysis, NPT analysis, and production
   overlapping; production analysis after production; and Result preparation
   only after all analyses. Verify the public timestamps, prepared download,
   and ZIP schema. For a changed AlphaFold3 path, verify server validation and
   confirmation, Prepare environment, optional MSA and template searches,
   prediction, Prepare results, and the downloadable `.tar.zst` archive.
   For humanization, verify four independent generator stages, union/evaluation,
   useful partial outcomes, bounded table reads and CSV/ZIP downloads. Match the
   containing workflow's scientific identity to the API before submission; older
   archives retain their published schema and ranking.
7. Stop on any unexpected invocation, stage, archive member, permission error,
   or configuration error. Capture the Job ID, request ID, safe stage, and
   timestamps before retrying; do not blindly resubmit paid work.
8. Publish the reviewed static frontend only after the API checks pass, then
   verify SPA fallback, login/logout, and same-origin API access.

Rollback restores the prior backend checkout and frontend static release as a
pair, restores the prior service or container configuration if it changed,
restarts the one API process, and waits for readiness. Do not point a
rolled-back binary at an incompatible newer database. The documented
pre-release transition is one-way.

During the current disposable-state phase, an Administrator may stop the
pre-release service and remove only its exact configured database when an
unsupported schema requires a reset. This also deletes its Users and settings.
Never apply that reset instruction to production state or cache.
