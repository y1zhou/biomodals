# Humanization publication and identifier boundaries

Status: accepted. Consolidates the original mutation-format (0010) and
identifier (0011) decisions with the schema 3 workflow download decision.
The [workflow spec](../specs/humanization-workflow.md#result-publication) owns
current filenames, fields and compatibility behavior; this record owns why.

## Typed mutation evidence

Mutation data grows with parents, candidates and iterations. Use Parquet for
standalone-app mutation tables and the workflow's common IMGT table to retain
types and reduce transfer size. This does not standardize native coordinates:
Sapiens retains iterative mutation history, while the other apps retain their
own endpoint and coordinate semantics. Compact standalone sequence, summary
and attempt tables may remain CSV for inspection.

This originally advanced standalone bundle manifest schemas to 2. Workflow
result versions and standalone app protocol versions are separate identities.

## Identifiers

FASTA exports replace each whitespace character in a parental or candidate ID
with `_` before adding the chain suffix. CSV/Parquet retain original IDs, as does
pair-seed derivation. Reject normalized-ID collisions such as `clone 1` and
`clone_1` before dispatch. Filesystem normalization must not silently alter
scientific identity or random streams.

p-AbNatiV2 uses short ASCII names during isolated upstream structure work.
Standalone collected structures use `structures/pair_NNNN/`; their manifest's
`structure_ids` mapping restores original parental IDs. This avoids upstream
dot truncation and filesystem byte limits without constraining user IDs.

The identifier correction originally advanced output/wrapper protocols to 3
without changing sequence-selection algorithms or standalone manifest schema 2.
Subsequent scientific changes can advance those identities again; their current
values live in app code, not this historical rollout description.

## Lean workflow downloads

The workflow download is for candidate selection and detailed cross-evaluation,
not a copy of every execution artifact. Keep final scientific tables and compact
generation accounting; leave native task publications under operational retention.
Generation-time structures, profiles, alignments and displacement metrics are
deliberately outside the download boundary, not falsely described as already
consolidated.

Evidence: the schema 2 archive for job
`4a0e44bd-9fbb-49ae-8fc6-84b72619aa1b` contained 29 selection rows and was
26,237,632 bytes. Six consolidated tables used 1,238,516 bytes, but 26 native
generation publications used 24,810,234 bytes. Repeated p-AbNatiV2 input/final
PDB strings alone used 14,977,544 bytes. The 180,768-byte manifest also repeated
87 scoring publications. Per-candidate copies made metadata scale with yield
and could exceed the service's bounded manifest reader.

Therefore record shared scientific identities once and hash delivered members.
A separate generation ledger preserves all origins, including no-ops, deduped
origins, rejected attempts and failed calls, without repeating sequences or
embedding per-candidate scoring manifests. This narrows the terminal publication
boundary and requires a new result/scientific identity. Old archives remain
immutable; neither old publications nor native task evidence are deleted.

## Why no-op evidence matters

A read-only audit of earlier job `a9085e43-5ad9-4234-a933-863e801ab869`
reconstructed all ten union rows and verified the archive hashes. Humatch made
zero edits because its parent already exceeded all configured targets: VH/VL/
pairing probabilities were approximately 0.99988/0.99899/0.98998 (target 0.95),
and germline likeness approximately 0.41236/0.41908 (target 0.4).
Its native summary reported success. Sapiens and p-AbNatiV2 each contributed
one changed pair; HuDiff contributed seven distinct valid pairs from ten
attempts, with three rejected for changed grid occupancy.

A null parental generation label is consequently not missing execution.
Candidate provenance must survive exact deduplication, while method labels
remain presentation rather than ranking inputs. More seed replicates do not
guarantee novel candidates or balance method yields.

## Alternatives not retained

Copying all native bundles preserved diagnostics but obscured useful outputs
and inflated metadata/transfer costs. Dropping them without generation accounting
would lose no-op, duplicate and rejected-attempt evidence. Increasing the manifest
memory guard would conceal duplication rather than remove it. Debug downloads,
PDB delivery and persistent experimental-panel selections remain separate future
product decisions, not speculative interfaces in this release.
