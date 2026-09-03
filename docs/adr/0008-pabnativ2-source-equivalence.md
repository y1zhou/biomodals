# Target p-AbNatiV2 source equivalence

Status: accepted.

## Context

The p-AbNatiV2 paper describes averaging relative solvent accessibility over
ten predicted structures when selecting mutable residues. The pinned AbNatiV
2.0.8 source instead hard-codes four ABodyBuilder3 predictions. Changing this
count can change eligible positions and the final greedy humanization result.

## Decision

The initial Biomodals app targets the pinned AbNatiV 2.0.8 source behavior,
including its four-prediction RASA ensemble. Record the source commit, package,
model checkpoints, structure checkpoint, ensemble count, and all humanization
parameters in each result manifest. Record hardware only as operational
telemetry and exclude it from scientific fingerprints. Describe this as source
equivalence, not paper-protocol equivalence.

## Consequences

- Direct upstream 2.0.8 runs provide the sequence and score oracle.
- The four-prediction behavior remains explicit despite not being a public app
  control.
- A ten-prediction implementation requires a new scientific identity and its
  own validation; it cannot silently replace this behavior.
