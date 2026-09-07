# Implement Sapiens humanization without the BioPhi runtime

Status: accepted.

## Context

The standalone Sapiens package exposes residue scores but BioPhi defines the
user-facing iterative humanization procedure: choose the canonical-amino-acid
argmax at every position and restore parental CDRs after each pass by default.
BioPhi also contains the separate OASis humanness evaluator and its large
repertoire database. OASis does not participate in Sapiens mutation selection
or residue scoring.

Installing the full BioPhi web/runtime stack would add unrelated services,
dependencies, and data distribution to a narrow sequence operation. Calling
floating Hugging Face repository names would make the scientific result
mutable even when the Sapiens package version stayed fixed.

## Decision

Implement BioPhi's narrow iterative Sapiens algorithm using pinned
`sapiens==1.1.0`, BioPhi's exact AbNumber/ANARCI versions, and immutable local
VH, VL, and tokenizer snapshots. Do not install BioPhi or OASis in the Sapiens
image. Record package, dependency, model, tokenizer, numbering, CDR, and
iteration identities in every result.

Treat OASis as a potential separate, method-independent evaluation app. It is
not part of Sapiens scientific identity and is deferred from the initial
antibody-humanization stack.

## Consequences

- The app can reproduce BioPhi humanization without carrying its web server,
  task queue, database, or OASis data.
- Exact sequence, mutation, numbering, and CDR behavior must match the pinned
  BioPhi oracle. Complete floating-point matrices require tight numerical
  comparison before changing an inference dependency.
- A dependency, checkpoint, tokenizer, or intentional algorithm change must
  update scientific identity even if the user-facing operation is unchanged.
- The initial stack has no common post-hoc repertoire humanness metric. A later
  OASis app can fill that role without coupling it to individual humanizers.
