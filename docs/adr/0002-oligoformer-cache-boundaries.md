# Split OligoFormer immutable assets from run intermediates

OligoFormer performance work will keep immutable upstream assets in the standard
model volume and store run-specific cached intermediates in the app output
volume. RNA-FM weights and full-human off-target references belong in the model
volume. The converted TargetScan references carry both declared source metadata
and SHA-256 digests of their actual converted bytes; that identity is mirrored
into the output volume so input preparation can derive a fail-closed evidence
key without mounting the model volume. Setup commits the converted bytes and
their marker to the model volume before publishing the matching identity to the
output volume. Identity publication, RNAplfold invalidation, and RNAplfold
construction share one stable global reference-state generation, so different
reference versions cannot write the same cache tree concurrently. The derived
full-human RNAplfold cache belongs in the
OligoFormer output volume, is keyed by the converted UTR content digest, and
uses Modal Volume v2 for the stage's high number of concurrent distinct-file
writers. Efficacy, off-target, toxicity, and packaging intermediates also belong
in the output volume.

This preserves the model volume as a shared read-mostly asset store while
allowing reruns to reuse expensive per-input work without baking user inputs or
derived results into an image or model cache. Three cache identities separate
the reusable stages:

1. The efficacy key covers app and upstream versions, input mRNA and optional
   siRNA FASTA content, and functionality-filter behavior.
2. The evidence key builds on the efficacy key and adds off-target mode,
   `top_n`, candidate-binding semantics, and either custom-reference content or
   the persisted converted-human-reference identity.
3. The final-table key builds on the evidence key and adds toxicity mode plus
   PITA, TargetScan, and toxicity thresholds.

This lets reference or `top_n` changes reuse GPU efficacy, and lets threshold
exploration reuse both efficacy and compact merged off-target evidence. The
implementation does not auto-prune output-volume cache entries; retention
policy can be added after the runtime split has real usage data.

`force` runs receive an isolated cache generation that participates in
the efficacy key and therefore all downstream keys. They rebuild without
deleting or overwriting cache trees that concurrent normal runs may be
producing or consuming. A repeated preparation within the same submitted force
run reuses its generation, including when all-human model setup requires
re-planning.

The run plan carries all three keys, the frozen semantic configuration used to
derive them, and the exact converted-human reference digest when applicable.
Efficacy and post-processing functions reject caller settings that differ from
the prepared plan. Full-human evidence construction revalidates the pinned
digest and holds the global reference-state generation until evidence is
committed, preventing a concurrent reference refresh from writing new-reference
scores beneath an old evidence key.

The first split preserves OligoFormer's standalone tarball return contract. The
output-volume cache is internal until a workflow consumer needs durable
`AppRunResult` outputs.

GPU efficacy and CPU post-processing will hand off through upstream-shaped files
in the output volume rather than an in-memory Python result object. This keeps
the cache inspectable, preserves upstream output formats, and avoids invasive
changes inside `scripts/infer.py`.

The wrapper may patch upstream minimally to add explicit stage selection, such
as efficacy-only execution and CPU post-processing from existing efficacy
outputs. That is preferable to running the full upstream pipeline and then
reverse-engineering partial state because it gives Modal a clear GPU/CPU
boundary while keeping upstream file formats as the contract.

The local entrypoint will orchestrate the GPU efficacy function and the CPU
post-processing function as separate remote calls. GPU functions should not own
downstream CPU scheduling in the first split; keeping orchestration local makes
each function single-purpose and keeps retry behavior visible at the app
boundary. For full-human off-target runs, the local entrypoint starts reusable
RNAplfold reference preparation and GPU efficacy concurrently after model and
reference readiness is established. It waits for both before starting
post-processing. If efficacy or merged human off-target evidence is already
ready, the entrypoint skips the corresponding model or reference setup.

Concurrent evidence variants can share an efficacy key, and concurrent
final-table variants can share an evidence key. Both efficacy and evidence
generation—and identical final-table generation—therefore acquire
stage-specific distributed build generations with an atomic Modal Dict
insertion. The same mechanism serializes the global RNAplfold reference cache.
A writer commits all output files and the completion marker before recording its
generation complete; waiters then reload and recheck the output volume. Failed
or timed-out generations advance through append-only status records, avoiding
unsafe compare-then-delete lock recovery.

When final output files are already present in the output-volume cache, reruns
should skip GPU efficacy and CPU post-processing compute. They still need a
small remote packaging path so the standalone entrypoint can return fresh
tarball bytes without exposing output-volume paths as public API.

Long-running model/reference setup, efficacy, reference preparation—including
TargetScan batch-shard preparation—off-target worker, branch, reducer, and
post-processing functions use Modal's 24-hour maximum timeout. Short run-plan
input preparation and packaging functions retain the app's normal timeout so
failures in those stages remain prompt.
