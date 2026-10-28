# Local tools without Modal credentials

Status: implemented after the final shared-understanding confirmation.
Investigated 2026-10-27 across `biomodals` and `biomodals-frontend`;
decisions accepted and implementation verified 2026-10-28.

## Requested outcome

- The API starts without Modal credentials so local scientific tools remain
  usable.
- The frontend displays only local tools when those credentials are absent.
- Keep modifications minimal and settle implementation details through the
  grill-with-docs interview before changing application behavior.

## Vocabulary and current boundaries

The startup credentials are `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. They
authorize the service's Modal access and are separate from a User's BioModals
login. Neither the API's session implementation nor the static frontend needs
an external session-signing secret.

Antibody Sequence Analysis runs on the API host and creates no Service Job.
The frontend glossary reserves Client Operation for work performed entirely in
the browser. Public therapeutic-reference downloads can still occur during
analysis, so local execution does not promise operation without a network.

Both repositories already use `CONTEXT.md` for canonical vocabulary. Keep
resolved domain terms there rather than introducing a competing glossary.

## Verified baseline implementation findings

| Boundary | Baseline behavior and source |
| --- | --- |
| Credential loading | `service/config.py:312` normalizes empty values to absent; process environment overrides the configured private file. |
| API startup | `service/api.py:233` unconditionally calls `install_modal_credentials`; `service/config.py:332` rejects an incomplete or absent pair. |
| Remote startup work | `service/api.py:94` preflights all five remote registrations; line 107 always starts reconciliation. |
| Container startup | `deploy/compose.yaml:7` independently requires both credentials during Compose interpolation. |
| Local analysis | `service/antibody_sequence_analysis/router.py:27` registers independently of remote Tool registrations, with login and mutation Origin/CSRF checks. |
| Readiness | `service/operations_api.py:39` already tolerates a missing reconciler task; the application must explicitly initialize that state. |
| Existing Jobs | `service/jobs_api.py:61` needs Tool configuration to render Jobs; `service/tool_runtime.py:142` needs registrations to advance them. Removing registrations wholesale is insufficient. |
| Remote request paths | Tool routers, shared Jobs, Result restoration, logs, Modal administration and billing can invoke provider operations. Guarding submission or preflight alone is insufficient. |
| Modal administration | `service/runtime_config.py:141` assumes a token ID exists; `service/admin_api.py:214` requires it in the response. |
| Frontend startup | `vite.config.ts:20` in the frontend reads public origins and proxy configuration, not Modal credentials. `src/auth.tsx:44` reports unavailable when the API cannot answer the session query. |
| Frontend discovery | `src/tools.ts:155` contains a static six-tool catalog. Antibody Sequence Analysis alone has `createsJobs: false`. |
| Frontend routing | `src/App.tsx:213` mounts remote overview routes publicly; submission and Job routes are separately protected by login. Catalog filtering alone does not gate direct links. |
| Frontend navigation | `src/AppShell.tsx:112` always exposes My Jobs to signed-in Users. `src/pages/admin/ModalAdminPage.tsx:111` automatically queries billing when its cost section mounts. |

Backend `service/` references above are relative to `src/biomodals/`.
Frontend references are relative to the sibling `biomodals-frontend` repository.

Local endpoints inside a larger remote Tool do not make that entire Tool local:
Protein Sequence Optimization has a local review endpoint but its full workflow
creates remote Jobs. Antibody Sequence Analysis is the currently available
end-to-end local Tool.

## Accepted decisions, 2026-10-28

1. **Authentication scope.** Only Modal credentials become optional. Preserve
   BioModals accounts, login and Origin/CSRF checks.
2. **Credential availability.** An absent or incomplete pair selects local
   operation. A complete pair retains existing strict startup preflight,
   including errors for rejected credentials, unreachable Modal or missing
   deployments. Availability changes require an API restart.
3. **Existing remote Jobs.** Preserve durable state, suspend remote
   reconciliation, and hide/block Jobs and Modal administration while
   credentials are absent. Already-running remote computations are not
   cancelled by this change. Restore access and normal reconciliation after
   credentials return and the API restarts. Local deletion cleanup continues
   in both modes.
4. **Capability contract and discovery.** A public
   `GET /api/v1/capabilities` returns `remote_execution_enabled: boolean`.
   Expose no credentials. Keep frontend metadata static and use its existing
   `createsJobs` classification, which matches the current domain: Jobs refer
   to remote computations. Local preparation within a remote Tool does not
   make that whole Tool local. Generate frontend types from OpenAPI and use
   one shared capability query.
5. **Unavailable routes and administration.** Gated API routes
   return `503 remote_execution_unavailable` before running their handlers.
   Keep every route registered so OpenAPI remains the same in both modes.
   Frontend direct links show an unavailable explanation with Back to tools;
   remote cards, My Jobs and Modal administration remain hidden. Users and
   Storage administration remain usable with their existing authorization.
   An unavailable-route response may precede authentication errors because
   availability is public; enabled routes retain all existing authentication,
   Origin and CSRF checks.
6. **Loading, failures and refresh.** Show known local entries
   while capability discovery is pending or fails; indicate lookup failure
   with a retry action rather than claiming credentials are absent. Remote
   routes wait for a confirmed enabled capability. Refresh on page load and
   window focus, plus after a remote-unavailable API response; add no periodic
   polling. A failed refresh must not keep advertising remote availability
   solely from an earlier successful response.
7. **Acceptance evidence.** Write the browser E2E scenario before implementation
   against the real app factory with no Modal credentials. Retain a downloaded
   analysis CSV and test report, and cover preserved existing Jobs and zero
   provider calls. Retain credentialed regression coverage and exercise the
   credential combinations and restart transition described below. Do not add
   unit tests unless the E2E test has absolutely no way to exercise the smaller
   module. Convenience, runtime or easier isolation do not justify an exception.
   If an exception is unavoidable, document why E2E cannot reach the behavior
   and write down the isolated module's failure modes before its test/code.

The user accepted all seven decisions, with the stricter E2E requirement in
decision 7. No design questions remain open.

## Guard boundaries verified after the first round

Keep remote registrations and local cache/pending/validation stores in both
modes. The Storage view reads those stores, SQLite and the local filesystem;
its cache cleanup invokes no Modal operations. Users administration is also
local. Blocked-Job counts belong to the Modal view and disappear with it.

A shared availability dependency can guard all remote Tool routers, the common
Jobs router, the separate Job logs router, and the independently mounted
Protein Sequence Optimization review router. Within the mixed Admin router,
only its five Modal endpoints need the dependency: configuration, costs,
state-unknown resolution, environment changes and Tool-setting changes. Users
and Storage endpoints retain their existing behavior. This avoids splitting
the Admin router or changing its response types merely to represent absence
of a Modal token ID.

The implementation retains existing remote registrations and
configuration, conditions credential installation, preflight and remote Job
advancement on one startup capability, and gates remote API routes and frontend
routes. The bounded reconciliation loop retains local deletion cleanup in both
modes.
It does not require a backend-owned Tool Catalog or a database schema change.

## Implementation sequence

1. Write the credential-free E2E scenario using temporary state and the real
   app factory. Demonstrate the current startup refusal before changing
   application behavior.
2. Add the startup capability, public capability endpoint and shared remote
   route guards. Skip credential installation, provider preflight and remote
   reconciliation when the credential pair is incomplete. Preserve local
   service initialization and readiness.
3. Make Compose's credential interpolation optional so container startup
   follows the same policy as direct API startup.
4. Generate frontend API types, add the shared capability query, filter the
   existing catalog and navigation, and gate remote routes with the agreed
   loading/error/unavailable behavior.
5. Update existing startup/deployment documentation and frontend ADR 0006,
   then run the E2E acceptance scenarios and relevant existing checks. Preserve
   the downloaded CSV and test report as repeatable verification artifacts.

## Verification and documentation follow-through

The existing browser fixture supplies fake Modal credentials and adapters
(`tests/browser_server.py:796`). It cannot alone prove that the real service
factory starts without credentials.

Required E2E acceptance: start the real API with isolated temporary state and no
Modal credential pair, authenticate according to the agreed policy, verify
local-only discovery and remote direct-link/API rejection, run real antibody
numbering, and retain a downloaded CSV plus a test report. Include an existing
remote Job to verify that its durable execution state is not changed and that
no provider operations are attempted. Retain the credentialed fixture's
existing regression coverage.

Also cover absent, whitespace-only and incomplete credential pairs; complete
credentials with failed preflight; capability loading/failure; and restarting
with credentials restored. No paid computation is needed for these checks.

Update the existing startup promises in the backend
README/deployment documentation and frontend ADR 0006. Frontend ADR 0001 keeps
Tool metadata frontend-owned and should remain satisfied. Add a new ADR only
if the chosen design makes a consequential architectural trade-off that needs
its own rationale.

The investigation phase changed no application code. The pre-existing
`pyproject.toml` modification was left untouched during implementation.

## Implementation evidence

The new browser scenario was written before production edits and first failed
at the real factory's mandatory-credential check. It now passes:

```sh
# From the sibling biomodals-frontend repository, after installing dependencies:
bun run test:e2e:local
```

It starts the real API factory with isolated temporary state, traps outbound
network connections, and substitutes only provider responses and the public
therapeutic-reference download. Actual antibody analysis, numbering, accounts,
SQLite and HTTP routing remain exercised. The retained `local-analysis.csv`
and `local-tools-report.json` artifacts are under the Playwright output
directory (`BIOMODALS_BROWSER_ROOT/test-results`). The report records all 49
guarded remote HTTP operations, an unchanged running Job and zero provider
calls in local mode. It also verifies absent, partial and whitespace pairs,
rejected complete credentials, successful credentialed restart, and the
return to local mode in an already-open browser tab. Pending discovery and a
failed refresh of previously enabled capabilities are exercised in the browser.

The existing cross-repository workflow runs this scenario after the existing
browser suite. Existing browser mocks now declare their remote capability.
No new unit tests were added; the obsolete startup-refusal test was removed.

Frontend lint, 149 existing tests, production build and OpenAPI type comparison
pass. The existing browser suite reports 120 passes and one timeout in the
native protein-optimization publication scenario; that scenario passes on a
targeted rerun with no code changes. Backend `prek` passes. The existing service suite reports 489 passes and
five failures in antibody/nanobody analysis cleanup under Python 3.14.7; the
same five event-loop failures reproduce against an untouched HEAD snapshot.
`ty` reports the same 13 diagnostics on changed service files as on HEAD
(existing Pydantic field overrides and an async-context-manager annotation).
