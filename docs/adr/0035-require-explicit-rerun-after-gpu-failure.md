# Require explicit rerun after GPU failure

Status: accepted.

GPU inference relies on Modal's infrastructure handling for container and
platform crashes, but the app adds no retry loop for surfaced upstream
exceptions, timeouts, or deterministic failures. A failed worker writes compact
diagnostics, leaves successful sibling Seed Completion Markers intact, and
makes its failed claim generations eligible for replacement.

The coordinator reports the completed and still-unmarked requested seed sets.
It may refresh the accumulated run summary for newly completed siblings, but it
does not publish a successful Inference Request Result or Request Retrieval
Archive while any requested seed remains unmarked.

A later explicit invocation performs normal cache reconciliation, reuses every
marked seed, and claims only the unmarked set. This keeps partial progress while
requiring the caller to authorize another potentially costly GPU attempt rather
than repeatedly charging for a deterministic failure.
