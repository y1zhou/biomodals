# Preserve completed simulations when extending production

Status: accepted, 2026-09-16.

A GROMACS production continuation creates a new linked Job and Execution Run,
leaving the completed source and its downloadable Result unchanged. It
inherits the source's checkpoint state and physical settings and publishes
the cumulative production trajectory. This costs additional storage and
analysis but preserves historical results and avoids concurrent writers to a
completed simulation. Adding time changes the scientific plan, so continuation
is not the kernel's same-plan Successor recovery; its fingerprint checks stay
intact. The website, API and CLI share the app-owned implementation described
in the [continuation spec](../specs/gromacs-continuation.md).
