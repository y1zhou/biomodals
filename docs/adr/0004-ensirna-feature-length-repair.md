# Repair ENsiRNA feature-length mismatches

Status: accepted for the pinned ENsiRNA wrapper.

## Context

Pinned upstream ENsiRNA commit
`028824341635903f3c661f5d1cc737de106493d5` assumes that RNAplex-derived
position and chain arrays have this length:

```text
61 + len(anti sequence) + len(sense sequence) + 2 sentinels
```

`Data_Prepare.get_anti_start()` returns `None` when the length differs. The
resulting null features are removed by upstream's later `dropna()`, so the
candidate does not reach the prepared JSON, RNA-FM preprocessing, model
inference, or the result workbook. Upstream also passes RNAplex-derived
secondary-structure strings to Rosetta without first fitting them to their
sequence lengths; a mismatch can prevent publication of that candidate's PDB.

This is not an upstream-equivalence repair. There is no upstream or external
oracle that defines the missing positions or secondary-structure states.

## Decision

Biomodals intentionally patches the pinned helper to rescue only candidates
whose derived feature lengths do not fit the model contract.

- A short position array is extended by decrementing the last position for
  each missing slot. Each added chain value is `3`, the upstream code for the
  siRNA strands.
- A long position array and its chain array are truncated together at the
  expected length.
- A short secondary-structure string is extended on the right with `.`, which
  represents an unpaired nucleotide.
- A long secondary-structure string preferentially loses unpaired `.` values
  from the left and then the right. Any remaining excess is truncated on the
  right.

These added positions, chain assignments, and unpaired states are invented
features. They satisfy the model and Rosetta shape contracts; they are not
claimed to be experimentally observed or uniquely implied by RNAplex.

The fitting branches do not modify correctly sized position, chain, or
secondary-structure features. ENsiRNA inference is record-oriented, so the
intended outcome is additional scored candidates, not changed raw scores for
already valid candidates. Candidate counts, row membership, output ordering,
and any downstream rank or selection can nevertheless change because rescued
candidates are now present. The repaired candidates' scores depend partly on
the invented features and must be interpreted accordingly.

The repair affects every stage downstream of candidate preparation:

- prepared candidate identities and row counts;
- JSON and PDB artifacts;
- RNA-FM processed shards;
- inference rows and the final XLSX workbook; and
- any downstream ranking or filtering over that workbook.

## Cache identity

The behavior is part of the prepared-cache scientific identity. The current
namespace includes app version `3`, pinned upstream commit, and wrapper patch
version `safe-ids-resumable-rnafm-v2`; inference results live below that same
content-addressed preparation key. Any change to the fitting policy must bump
the wrapper patch version (and the app version when appropriate) before old
prepared artifacts or scores may be reused. Marker schema version `3` validates
publication shape but does not replace that semantic identity.

## Evidence and tests

The source rewrite is guarded by the SHA-256 of the exact pinned upstream file
and the image build compiles the patched module. The contract test fixture
stores that upstream file losslessly. The behavioral boundary test applies the
production patch and demonstrates that:

- pinned upstream returns no positional features for short and long cases;
- the patched helper emits the required length for both cases;
- an invented short-case position continues the upstream decrement and uses
  chain `3`;
- an exactly sized candidate returns the same features before and after the
  patch; and
- secondary-structure fitting preserves exact input, pads short input with
  unpaired states, and trims long boundary states.

These tests establish control flow and boundary semantics, not scientific
validity. A live paired run can compare overlapping candidate scores and count
rescued rows, but upstream supplies no score for a dropped candidate and thus
cannot validate the rescued score. No independent experimental or model-author
study currently justifies one invented-feature policy over another.

## Removal criteria

Remove or replace this patch when the pinned upstream version adopts an
explicit feature-length policy, the model authors publish a canonical mapping,
or independent evidence shows that rejecting mismatches is safer than scoring
invented features. Such a change requires a new cache identity, boundary tests
for the replacement policy, and paired comparison of overlapping candidates;
it must not silently reuse scores produced under this decision.
