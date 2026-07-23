# FastAPI identity design for Biomodals

Status: superseded by `api-service-architecture.md`; retained as decision
research for the implemented identity boundary

Decision date: 2026-07-16

Scope: browser login and private job ownership for a few dozen employees

## Decision

Use administrator-provisioned company email accounts, Argon2id passwords, and
opaque server-side browser sessions stored in the service's local SQLite
database. There is no public registration, email delivery integration,
personal API token, JWT, or external identity provider in v1.

End users interact with a separate website, not the API directly. In
production, the reverse proxy serves the frontend and `/api` on one internal
HTTPS origin. The frontend development server likewise proxies `/api` to local
FastAPI. This permits a conventional login form and host-only cookies without
making Modal credentials or Modal-specific proxy headers part of the browser
contract.

The implementation intentionally uses a small repo-owned identity boundary
plus one cryptographic dependency:

- Python `secrets` creates high-entropy opaque tokens;
- only SHA-256 token digests are stored;
- [`pwdlib`](https://frankie567.github.io/pwdlib/) supplies its recommended
  Argon2 password hasher and transparent rehash support; and
- FastAPI dependencies turn an authenticated session into a minimal
  `Principal(user_id, email, display_name)`.

This was selected over FastAPI Users after the browser and lifecycle
requirements became concrete. FastAPI Users is a credible general account
package, but its built-in strategies do not directly provide the chosen
sliding idle timeout, absolute lifetime, session-wide revocation, CSRF binding,
and atomic one-time reset semantics. Supplying those pieces would retain most
of the custom security-sensitive code while also adding an ORM, async SQLite
driver, and a maintenance-mode framework.
[FastAPI Users: features and maintenance notice](https://fastapi-users.github.io/fastapi-users/latest/)

## User and ownership model

Each employee has an immutable random UUID. Normalized company email is a
unique login identifier and display metadata; it is not the ownership key.
Disabled users and their jobs remain in the database. User deletion is outside
v1.

User Status is `pending_setup`, `enabled`, or `disabled`. Provisioning
creates a pending-setup User; successful Password Setup enables them. A disabled
User cannot authenticate, consume a Password Link, or submit a Job. Re-enabling
returns a User with an existing password to enabled and one without a password
to pending setup. Administrator role is independent, but only an enabled
Administrator satisfies the final-administrator safeguard.

Every job stores the submitter's UUID. Every list, detail, cancellation, and
download query constrains both `owner_user_id` and `job_id`. A job belonging to
another user produces the same `404` as a nonexistent job, and the service
performs that check before resolving any Modal call or Volume path. Clients
cannot submit an owner field. Account administrators can provision, reset, and
disable accounts, but that role does not grant job inspection.

```text
browser session cookie
        |
        v
opaque-token digest -> session -> Principal(user UUID)
                                      |
                                      v
                           owner-scoped job query
                                      |
                                      v
                           internal Modal adapter
```

This `Principal` boundary also leaves a clean migration path: company OIDC or
a trusted identity proxy can produce the same principal later without changing
workload routers or job ownership.

## Manual account lifecycle

An administrator runs the CLI with the same `BIOMODALS_API_CONF_ENV` (or
equivalent explicit `BIOMODALS_STATE_DIR` and `BIOMODALS_PUBLIC_URL`) as the
server:

```bash
uv run biomodals api admin create-user alice@example.com \
  --display-name "Alice Example"

uv run biomodals api admin reset-password alice@example.com
uv run biomodals api admin disable-user alice@example.com
```

`create-user` and `reset-password` print one URL exactly once, accompanied by
its absolute expiration time. An administrator delivers it through company
chat or in person after identifying the employee. The link expires after one
hour and has this shape:

```text
https://biomodals.example.com/set-password#token=<random-token>
```

The secret is in the URL fragment, which is not sent to the web server or
included in ordinary proxy request logs. Merely opening the link does not
consume it, so link previewers and security scanners are harmless. The
frontend reads the fragment and submits the token with the chosen password to
`POST /api/v1/auth/set-password`; that successful POST consumes the token.
Expired, reused, and unknown links receive the same error.

The equivalent create-user and replacement-link HTTP responses declare both
the one-time URL and `expires_at` in OpenAPI. The frontend localizes the
timestamp and adds "Valid for approximately one hour" without a live countdown.
Closing the handoff dialog still destroys the only frontend copy of the URL;
the non-secret expiration metadata cannot be used to retrieve it again.

A successful password setup or reset atomically:

1. stores the new Argon2id hash;
2. deletes all setup/reset links for that user; and
3. revokes all of that user's sessions.

Disabling an account likewise revokes its sessions and password links. There
is no automated "Forgot password" request endpoint: the website tells the
employee to contact an administrator. This avoids account enumeration and
email infrastructure in the first version.

## Password policy

Passwords contain 15 to 128 Unicode characters. Spaces and passphrases are
allowed, and password managers work normally. There are no required uppercase,
digit, symbol, or periodic-rotation rules. A small local denylist rejects
obvious common values. Passwords are never logged or stored in plaintext.

This favors length and compromised/common-password screening over composition
rules, consistent with current NIST guidance. The local denylist is deliberately
small for v1 and can be replaced with a larger offline corpus without changing
the API. [NIST SP 800-63B](https://pages.nist.gov/800-63-4/sp800-63b.html)

Login performs a dummy Argon2 verification for unknown or unactivated users so
timing and messages reveal less account state. Password verification happens
before a short `BEGIN IMMEDIATE` transaction; the transaction rechecks the
stored hash and User Status before issuing a session, preventing a concurrent
reset or disable from losing its revocation guarantee.

Job admission likewise rechecks enabled User Status inside the same write
transaction that applies idempotency and Active Job Limits. A session
authenticated before an Administrator disables the User cannot admit paid work
after that disable commits.

Argon2 work runs through one process-local bounded executor: two operations may
run and eight more may wait. Further login or Password Setup requests receive
`503 authentication_busy` and a `Retry-After` header. Password Setup first
checks the high-entropy Password Link digest before spending an Argon2 slot,
then atomically rechecks and consumes the link while committing the password
and replacement Session. This bound protects API responsiveness; it is not an
account-lockout or per-User throttling policy.

## Browser sessions and CSRF

Login returns two independent random values as cookies:

- the session token is `HttpOnly`, host-only, `SameSite=Lax`, and `Path=/`;
- the CSRF token is readable by the frontend and must be echoed in
  `X-CSRF-Token` for state-changing requests.

With `BIOMODALS_SECURE_COOKIES=true`, the session cookie is named
`__Host-biomodals-session` and has the `Secure` attribute. Local HTTP
development uses `biomodals-session` with secure cookies disabled. Neither
cookie sets a `Domain` attribute.

Unsafe routes require both the session-bound CSRF value and an exact match to
the origin configured by `BIOMODALS_PUBLIC_URL`. Login and password setup also
require that exact Origin. CORS is not enabled; the frontend uses the
same-origin `/api` proxy. These controls remain required even though the
service is internal.

The server stores session and CSRF digests, never the bearer values. A session
expires after 30 days without use or 90 days after login, whichever occurs
first. Successful authentication slides only the idle deadline. Logout revokes
one session; password reset and account disable revoke every session
immediately.

## Login throttling decision

There is deliberately no login throttling or lockout in v1. The service is
reachable only on the company network, has a few dozen manually provisioned
users, and the product owner explicitly preferred avoiding lockout and
throttling machinery for this department server.

This is a documented risk acceptance, not a general recommendation for
password services. Failed logins still use a generic response and are recorded
without credentials. Revisit rate limiting before exposing the service to a
larger network, adding automated clients, or observing password-guessing
activity. OWASP otherwise recommends controls against automated attacks.
[OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)

## Persistence and backup

The same local SQLite database stores users, password-link digests, session
digests, and private job metadata. The essential tables are:

| Table | Important fields |
| --- | --- |
| `users` | immutable UUID and normalized unique login email, Administrator-editable display name, Argon2id hash, User Status, timestamps |
| `password_tokens` | SHA-256 token digest, user UUID, expiry |
| `sessions` | SHA-256 session digest, user UUID, CSRF digest, created/last-seen/absolute-expiry timestamps |
| `jobs` | owner UUID, public job UUID, workload, state and internal provider/artifact metadata |

SQLite runs in WAL mode on local disk with foreign keys and a busy timeout. Run
one FastAPI worker. The state directory is separate from the rebuildable
artifact cache. Pre-release state is currently disposable; before real
production Users are onboarded, the state directory must be included in
company backups. Backups must use a SQLite-aware backup or snapshot: copying
only `service.sqlite3` while its WAL is active can omit committed data.

## Options considered

| Option | Assessment for this service |
| --- | --- |
| **Repo-owned opaque sessions + `pwdlib`** | **Selected.** It implements the exact cookie, reset, revocation and SQLite transaction semantics with one focused dependency and no client-facing vendor contract. |
| FastAPI Users | Best-known FastAPI account package and supports cookie/database strategies, but is in maintenance mode and still requires custom session/reset/CSRF behavior for this design. |
| Long-lived personal API tokens | Smaller backend, but rejected for the primary UX because employees use a website and should not copy or store secrets manually. May be added separately for technical automation later. |
| JWT access/refresh tokens (`AuthX`, FastAPI-Login, PyJWT) | Stateless validation is attractive, but immediate reset/disable revocation and browser refresh-token handling add complexity without benefit at this scale. |
| `keyshield` API keys | Useful API-key lifecycle package, but it manages keys rather than the interactive employee accounts and cookie sessions required here. [keyshield](https://pypi.org/project/keyshield/) |
| `oauth2-proxy` | Good companion to an existing OIDC provider, but there is no usable company IdP for this development server. FastAPI would still own per-job authorization. [oauth2-proxy](https://oauth2-proxy.github.io/oauth2-proxy/) |
| Authelia | Viable shared file/LDAP identity service for several internal apps, but another service and reverse-proxy policy layer is excessive for this API alone. [Authelia](https://www.authelia.com/) |
| authentik | Full user-management and OIDC product with the best admin UX of the evaluated self-hosted options, but substantially more operational machinery than a few dozen manual accounts. [authentik](https://goauthentik.io/) |
| Authlib/company OIDC | Preferred future direction when a supported company identity provider exists. It avoids maintaining passwords locally, but does not provide a local account database by itself. [Authlib](https://docs.authlib.org/) |

The selected design is intentionally scoped to this private, single-host
department service. An Internet-facing or multi-application identity platform
should adopt a maintained external IdP rather than extending this code into a
general authentication server.
