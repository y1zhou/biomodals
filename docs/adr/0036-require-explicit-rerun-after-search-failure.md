# Require explicit rerun after search failure

Status: accepted.

MSA and template workers rely on Modal's infrastructure handling for container
and platform crashes, but the app adds no retry loop for surfaced HMMER,
template-tool, timeout, or deterministic failures. Every successfully published
Raw Database MSA or Template Search Result remains canonical and reusable, and
failed claim generations become eligible for replacement.

The coordinator reports the exact sequence/database and template tasks that
remain incomplete. It does not publish dependent combined MSAs, construct an
Enriched AlphaFold Input, or start inference while required search evidence is
missing.

A later explicit invocation performs normal field and cache reconciliation and
schedules only missing database or template work. This preserves the
one-database retry boundary while requiring the caller to authorize another
potentially costly CPU attempt.
