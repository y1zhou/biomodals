# Test selection by failure mode

The browser and deployed E2E flows exercise ordinary user journeys. Keep an
isolated test only when it can detect a concrete failure that those journeys are
unlikely to force or diagnose. Review tests against these failure modes before
adding another fixture or assertion:

| System | Failures worth isolating |
| --- | --- |
| Execution ledger and coordinator | Duplicate submission, lost provider response, restart with stale state, cancellation race, partial task result, wrong resource admission, corrupted publication, and schema incompatibility. |
| Artifact and cache boundaries | Same-size content corruption, incomplete marker, stale generation, path escape, interrupted write, and a cache key that omits scientific input. |
| Scientific input and output transforms | Invalid or colliding identifiers, malformed chemistry or structure, missing result columns, wrong row association, and scientific parameter drift. |
| Service security and admission | Cross-user data access, missing CSRF protection, replay changing ownership or input, invalid configuration starting a public service, and admission accepting work beyond limits. |
| Modal packaging and workflow wiring | Missing source in the remote image, wrong resource class, child coordinator duplication, and workflow dependency or deployment resolution failure. |

CLI list/help smoke tests and E2E journeys already reveal catalog entries and
ordinary path layout. Avoid tests that merely repeat an existing name, module
path, enum declaration, helper alias, or constructor output. Prefer a test that
forces one of the failures above and checks the resulting behavior.

This is a selection guide, not a claim that a passing local suite substitutes
for a deployed Modal or browser run. When a new isolated test is necessary,
record its failure mode first, then add the smallest fixture that causes it.
