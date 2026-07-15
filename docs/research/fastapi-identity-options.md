# FastAPI identity options for Biomodals

Status: research and provisional recommendation
Research date: 2026-07-15
Scope: local identities for a department-only API used by a few dozen employees

## Recommendation

For the stated department-only development server, provision each employee an
immutable user UUID and an individually revocable **opaque API token**. Keep
users, token digests, and job ownership in local SQLite. Authenticate the
`Authorization: Bearer` token through a shared repo-owned `Principal`
dependency or middleware. Provision users and tokens through an admin CLI; do
not expose public registration.

This is deliberately smaller than a username/password system: there is no
password policy, reset flow, login throttling, JWT signing key, refresh-token
flow, or browser cookie/CSRF policy to operate. A token should contain a public
lookup identifier plus at least 32 random secret bytes, be shown once, and be
stored only as a digest. HTTPS is still required because it is a bearer
credential. FastAPI's security helpers parse the header, but the token store
and principal mapping remain application responsibilities. [FastAPI security
reference](https://fastapi.tiangolo.com/reference/security/) · [OWASP REST
security](https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html)

If employees instead need username/password login, use **FastAPI Users with its
SQLAlchemy adapter and SQLite**, behind the same `Principal` boundary. Include
its Bearer login router but omit public registration, and provision or disable
accounts through an admin CLI using `UserManager`. Prefer its revocable
database-token strategy to long-lived JWTs. FastAPI Users is the most complete
FastAPI-native account package found, but it is officially in maintenance mode
and is materially more machinery than personal tokens.

FastAPI Users supplies authentication and basic account state, not job
authorization. On submission, the server must set `owner_id` from the
authenticated principal. Every read, list, cancellation, and artifact query
must constrain by both `job_id` and `owner_id` (returning 404 for another
owner). Modal call IDs and client-supplied owner fields are not authorization.

This requirement strengthens the case for one unified API control plane.
Separate public servers would otherwise duplicate the user database and token
policy, or force the project to deploy a separate identity provider now.

## Comparison

| Option | Authentication and API tokens | Users and provisioning | Authorization | Maintenance and fit |
| --- | --- | --- | --- | --- |
| **Repo-owned opaque tokens** | FastAPI parses Bearer credentials; `secrets`, `hashlib`/`hmac`, and SQLite provide generation, digest storage, lookup, and revocation. No JWT or password flow. | Admin CLI creates stable UUID users and one or more named tokens per user. | The token resolves to a `Principal`; per-job ownership remains application code. | **Best fit if users accept issued tokens.** Smallest surface and no new runtime dependency, but its focused token schema and tests are ours to maintain. [FastAPI security](https://fastapi.tiangolo.com/reference/security/) · [Python `secrets`](https://docs.python.org/3/library/secrets.html) |
| **keyshield** | Versioned API keys, Argon2/bcrypt hashing with a pepper, expiry, revocation, scopes, FastAPI dependencies, and a Typer CLI. | Manages keys, not employee accounts; Biomodals must associate each key with a stable user. Its current SQLAlchemy repository adds an ORM and async driver. | Returns an API-key record; the app must map it to a principal and enforce job ownership. | Closest third-party personal-token implementation, but young (**2.0.0, 2026-03-06**) and larger than the required token table. Worth a pilot, not an automatic choice. [documentation and PyPI](https://pypi.org/project/keyshield/) · [source](https://github.com/Athroniaeth/keyshield) |
| **FastAPI Users** | Bearer or cookie transport; JWT, database, and Redis strategies. Database tokens are revocable; JWTs are not individually revocable. | SQLAlchemy and Beanie adapters; password hashing; login, registration, reset, verification, and user routers. Users can be created programmatically, so the registration router can remain absent. | Dependencies support active, verified, and superuser checks. Per-job ownership remains application code. | **Best fit if users require password login.** PyPI lists **15.0.5 (2026-03-27)**, but the project is explicitly in maintenance mode. [features and maintenance notice](https://fastapi-users.github.io/fastapi-users/latest/) · [authentication strategies](https://fastapi-users.github.io/fastapi-users/latest/configuration/authentication/) · [programmatic provisioning](https://fastapi-users.github.io/fastapi-users/latest/cookbook/create-user-programmatically/) · [PyPI](https://pypi.org/project/fastapi-users/) |
| **FastAPI-Login** | Thin JWT issuer/validator with header or cookie lookup and optional scopes. No built-in refresh or token revocation. | The application writes credential checking and a `user_loader`; it supplies all storage and provisioning. | Route dependency and scopes only; ownership is application code. | Too little benefit over the built-in baseline. Latest PyPI release is **1.10.3 (2024-12-14)**. [official usage](https://github.com/maxrdu/fastapi_login#usage) · [PyPI](https://pypi.org/project/fastapi-login/) |
| **AuthX** | JWT access/refresh tokens, several token locations, blocklists, scopes, and policy hooks. | Official examples make the application validate passwords and load users; no account database or admin provisioning is supplied. | Useful token/scopes toolkit, but resource ownership remains application code. | Actively released (**1.7.1, 2026-06-27**), but it does not solve the missing identity store. [official docs](https://authx.yezz.me/) · [basic example](https://github.com/yezz123/authx/blob/main/docs/get-started/basic-usage.md) · [PyPI](https://pypi.org/project/authx/) |
| **Authlib** | Standards-focused OAuth/OIDC client, server, resource-server, and JOSE toolkit. FastAPI support is chiefly an OAuth client integration. | No local account database or provisioning workflow. Building an authorization server still requires those pieces. | OAuth scopes are available, but app resources remain app-owned. | Strong future choice for company OIDC, not a local identity implementation. **1.7.2 (2026-05-06)**. [client roles](https://docs.authlib.org/en/latest/oauth2/client/index.html) · [FastAPI client](https://docs.authlib.org/en/latest/client/fastapi.html) · [PyPI](https://pypi.org/project/Authlib/) |
| **oauth2-proxy** | Reverse-proxy sessions against an OAuth/OIDC provider; can additionally use an htpasswd file and validate issuer-backed JWTs. It forwards trusted identity headers upstream. | No real local user-management plane; htpasswd administration is file based. | Coarse proxy rules only; FastAPI must still enforce job ownership and must trust headers only from the proxy. | Poor fit without an IdP and awkward for CLI clients. **7.15.2 (2026-04-14)** fixed several critical authentication bypasses, so patch discipline is essential. [configuration](https://oauth2-proxy.github.io/oauth2-proxy/configuration/overview/) · [release](https://github.com/oauth2-proxy/oauth2-proxy/releases/tag/v7.15.2) |
| **Authelia** | Reverse-proxy cookies/basic auth and an OIDC provider with authorization-code, device-code, client-credentials, and bearer-token support. | File or LDAP users; file users use modern password hashes but are configuration-managed rather than managed by FastAPI. | Proxy access rules plus user/group headers; FastAPI still owns object authorization. | Viable shared identity service if several internal applications need it, but extra service/proxy configuration is excessive for this API alone. Latest release shown is **4.39.20 (2026-05-26)**. [file users](https://www.authelia.com/configuration/first-factor/file/) · [proxy identity](https://www.authelia.com/integration/proxies/introduction/) · [OIDC grants](https://www.authelia.com/integration/openid-connect/introduction/) · [releases](https://github.com/authelia/authelia/releases) |
| **authentik** | Full OAuth/OIDC provider and proxy, including device-code flow for CLIs and client credentials. | Admin UI/API for users, groups, roles, passwords, sessions, and service accounts. | Policies and application bindings; FastAPI still enforces individual job ownership. | Most complete and most operationally expensive. Consider if the department wants a reusable IdP for several services. Latest release is **2026.5.2 (2026-05-28)**. [users](https://docs.goauthentik.io/users-sources/user/user_basic_operations/) · [OAuth/OIDC flows](https://docs.goauthentik.io/add-secure-apps/providers/oauth2/) · [service accounts](https://docs.goauthentik.io/sys-mgmt/service-accounts/) · [releases](https://github.com/goauthentik/authentik/releases) |
| **FastAPI security + pwdlib + PyJWT** | FastAPI documents OAuth2 password login and JWT Bearer validation; `pwdlib` provides recommended Argon2 hashing and PyJWT signs/verifies tokens. | The project must build the user table, provisioning, password policy, disable/reset flows, login throttling, and migrations. | Everything beyond token decoding is application code. | Least package-level lock-in for password login, but the largest amount of security-sensitive account code to own. Prefer FastAPI Users if passwords are required. [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/) · [pwdlib](https://frankie567.github.io/pwdlib/reference/pwdlib/) · [PyJWT](https://pyjwt.readthedocs.io/en/stable/) |

## Proposed boundary

Keep the scientific routers independent of the selected identity package:

```text
Authorization: Bearer ...
        |
identity adapter (local opaque tokens initially)
        |
Principal(id, username, is_admin)
        |
app/workflow router -> owner-scoped job registry -> Modal compute app
```

This makes migration to password login or company OIDC an adapter change:
FastAPI Users, Authlib, or an upstream proxy can produce the same `Principal`,
while job ownership and HTTP contracts remain unchanged. User credentials and
bearer tokens stay on the internal server; only the server's Modal credential
crosses into the Modal SDK.

## Decision still required

Choose the client login experience before implementation:

1. **Personal API tokens (recommended):** an admin verifies the employee out of
   band and issues one or more named, long-lived tokens that can be revoked
   independently. This is simplest for curl, Python clients, and unattended
   scripts, but it has no interactive login or self-service recovery.
2. **Session tokens:** users post username/password, receive an expiring bearer
   token, and repeat login when it expires. This is friendlier for an
   interactive UI but makes the server responsible for password and login
   lifecycle. FastAPI Users is the preferred implementation for this branch.
